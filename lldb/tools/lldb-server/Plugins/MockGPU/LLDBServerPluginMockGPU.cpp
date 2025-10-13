//===-- LLDBServerPluginMockGPU.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginMockGPU.h"
#include "ProcessMockGPU.h"
#include "lldb/Host/common/TCPSocket.h"
#include "llvm/Support/Error.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

enum {
  BreakpointIDInitialize = 1,
  BreakpointIDShlibLoad = 2,
  BreakpointIDThirdStop = 3,
  BreakpointIDResumeAndWaitForResume = 4,
  BreakpointIDWaitForStop = 5
};

LLDBServerPluginMockGPU::LLDBServerPluginMockGPU(
  LLDBServerPlugin::GDBServer &native_process, MainLoop &main_loop)
    : LLDBServerPlugin(native_process, main_loop) {
  m_process_manager_up.reset(new ProcessMockGPU::Manager(m_main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      m_main_loop, *m_process_manager_up, "mock-gpu.server"));

  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() faking launch...");
  ProcessLaunchInfo info;
  info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                      eLaunchFlagDisableASLR);
  Args args;
  args.AppendArgument("/pretend/path/to/mockgpu");
  args.AppendArgument("--option1");
  args.AppendArgument("--option2");
  args.AppendArgument("--option3");
  info.SetArguments(args, true);
  info.GetEnvironment() = Host::GetEnvironment();
  m_gdb_server->SetLaunchInfo(info);
  Status error = m_gdb_server->LaunchProcess();
  if (error.Fail()) {
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() failed to launch: %s", error.AsCString());
  } else {
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::LLDBServerPluginMockGPU() launched successfully");
  }
}

LLDBServerPluginMockGPU::~LLDBServerPluginMockGPU() {
  CloseFDs();
}

llvm::StringRef LLDBServerPluginMockGPU::GetPluginName() {
  return "mock-gpu";
}

void LLDBServerPluginMockGPU::CloseFDs() {
  if (m_fds[0] != -1) {
    close(m_fds[0]);
    m_fds[0] = -1;
  }
  if (m_fds[1] != -1) {
    close(m_fds[1]);
    m_fds[1] = -1;
  }
}

int LLDBServerPluginMockGPU::GetEventFileDescriptorAtIndex(size_t idx) {
  if (idx != 0)
    return -1;
  if (m_fds[0] == -1) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, m_fds) == -1) {
      m_fds[0] = -1;
      m_fds[1] = -1;
    }
  }
  return m_fds[0];
}


bool LLDBServerPluginMockGPU::HandleEventFileDescriptorEvent(int fd) { 
  if (fd == m_fds[0]) {
    char buf[1];
    // Read 1 bytes from the fd
    read(m_fds[0], buf, sizeof(buf));
    return true;
  }
  return false;
}


std::optional<GPUPluginConnectionInfo> LLDBServerPluginMockGPU::CreateConnection() {
  std::lock_guard<std::mutex> guard(m_connect_mutex);
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __PRETTY_FUNCTION__);
  if (m_is_connected) {
    LLDB_LOGF(log, "%s already connected", __PRETTY_FUNCTION__);
    return std::nullopt;
  }
  if (m_is_listening) {
    LLDB_LOGF(log, "%s already listening", __PRETTY_FUNCTION__);
    return std::nullopt;
  }
  m_is_listening = true;
  LLDB_LOGF(log, "%s trying to listen on port 0", __PRETTY_FUNCTION__);
  llvm::Expected<std::unique_ptr<TCPSocket>> sock = 
      Socket::TcpListen("localhost:0", 5);
  if (sock) {
    GPUPluginConnectionInfo connection_info;
    const uint16_t listen_port = (*sock)->GetLocalPortNumber();
    connection_info.connect_url = llvm::formatv("connect://localhost:{}", 
                                                listen_port);
    connection_info.copy_cpu_breakpoints_during_attaching = true;
    LLDB_LOGF(log, "%s listening to %u", __PRETTY_FUNCTION__, listen_port);
    
    m_listen_socket = std::move(*sock);
    m_is_connected = false;
    auto res = m_listen_socket->Accept(m_main_loop, [this](std::unique_ptr<Socket> socket) {
      std::unique_ptr<Connection> connection_up(new ConnectionFileDescriptor(std::move(socket)));
      this->m_gdb_server->InitializeConnection(std::move(connection_up));
      m_is_connected = true;
    });
    if (res) {
      m_read_handles = std::move(*res);
    } else {
      LLDB_LOGF(log, "MockGPU failed to accept: %s", llvm::toString(res.takeError()).c_str());
    }
    
    return connection_info;
  } else {
    std::string error = llvm::toString(sock.takeError());
    LLDB_LOGF(log, "%s failed to listen to localhost:0: %s", 
              __PRETTY_FUNCTION__, error.c_str());
  }
  m_is_listening = false;
  return std::nullopt;
}

std::optional<GPUActions> LLDBServerPluginMockGPU::NativeProcessIsStopping() {
  NativeProcessProtocol *native_process = m_native_process.GetCurrentProcess();
  // Show that we can return a valid GPUActions object from a stop event.
  if (native_process->GetStopID() == 3) {
    GPUActions actions = GetNewGPUAction();
    GPUBreakpointInfo bp;
    bp.identifier = BreakpointIDThirdStop;
    bp.name_info = {"a.out", "gpu_third_stop"};
    actions.breakpoints.emplace_back(std::move(bp));
    return actions;
  }
  return std::nullopt;
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginMockGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  const auto bp_identifier = args.breakpoint.identifier;
  Log *log = GetLog(GDBRLog::Plugin);
  if (log) {
    std::string json_string;
    llvm::raw_string_ostream os(json_string);
    os << toJSON(args);
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::BreakpointWasHit(%u):\nJSON:\n%s", 
              bp_identifier, json_string.c_str());
  }
  NativeProcessProtocol *gpu_process = m_gdb_server->GetCurrentProcess();
  GPUPluginBreakpointHitResponse response(GetNewGPUAction());
  if (bp_identifier == BreakpointIDInitialize) {
    response.disable_bp = true;
    LLDB_LOGF(log, "LLDBServerPluginMockGPU::BreakpointWasHit(%u) disabling breakpoint", 
              bp_identifier);
    response.actions.connect_info = CreateConnection();

    // We asked for the symbol "gpu_shlib_load" to be delivered as a symbol
    // value when the "gpu_initialize" breakpoint was set. So we will use this
    // to set a breakpoint by address to test setting breakpoints by address.
    std::optional<uint64_t> gpu_shlib_load_addr = 
        args.GetSymbolValue("gpu_shlib_load");
    if (gpu_shlib_load_addr) {
      GPUBreakpointInfo bp;
      bp.identifier = BreakpointIDShlibLoad;
      bp.addr_info = {*gpu_shlib_load_addr};
      bp.symbol_names.push_back("g_shlib_list");
      bp.symbol_names.push_back("invalid_symbol");
      response.actions.breakpoints.emplace_back(std::move(bp));
    }
  } else if (bp_identifier == BreakpointIDShlibLoad) {
    // Tell the native process to tell the GPU process to load libraries.
    response.actions.load_libraries = true;
  } else if (bp_identifier == BreakpointIDThirdStop) {
    // Tell the native process to tell the GPU process to load libraries.
    response.actions.load_libraries = true;
  } else if (bp_identifier == BreakpointIDResumeAndWaitForResume) {
    response.actions.resume_gpu_process = true;
    response.actions.wait_for_gpu_process_to_resume = true;
  } else if (bp_identifier == BreakpointIDWaitForStop) {
    // Update the stop ID to make sure it reflects the fact that we will need
    // to stop at the next stop ID.
    response.actions.stop_id = gpu_process->GetNextStopID();
    response.actions.wait_for_gpu_process_to_stop = true;
    // Simulate a long wait for the GPU process to stop.
    std::thread resume_after_delay_thread([gpu_process]{
      sleep(5);
      gpu_process->Halt();
    });
    resume_after_delay_thread.detach();
  }

  return response;
}

GPUActions LLDBServerPluginMockGPU::GetInitializeActions() {
  GPUActions init_actions = GetNewGPUAction();
  {
    GPUBreakpointInfo bp;
    bp.identifier = BreakpointIDInitialize;
    bp.name_info = {"a.out", "gpu_initialize"};  
    bp.symbol_names.push_back("gpu_shlib_load");
    init_actions.breakpoints.emplace_back(std::move(bp));
  }
  {
    GPUBreakpointInfo bp;
    bp.identifier = BreakpointIDResumeAndWaitForResume;
    bp.name_info = {"a.out", "gpu_resume_and_wait_for_resume"};
    init_actions.breakpoints.emplace_back(std::move(bp));
  }
  {
    GPUBreakpointInfo bp;
    bp.identifier = BreakpointIDWaitForStop;
    bp.name_info = {"a.out", "gpu_wait_for_stop"};
    init_actions.breakpoints.emplace_back(std::move(bp));
  }
  return init_actions;
}
