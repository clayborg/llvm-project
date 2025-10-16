//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginNVGPU.h"
#include "../Utils/Utils.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessNVGPU.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

/// Internal identifier for the gpu initialization breakpoint.
static constexpr uint32_t kGpuInitializationBreakpoint = 1;

/// Helper function to set environment variables with logging.
///
/// Checks if the environment variable already exists and either uses the
/// existing value or sets it to the specified value.
///
/// \param[in] env_var_name
///     Name of the environment variable to set.
///
/// \param[in] cmake_value
///     Value to use as default.
///
/// \param[in] log
///     Log instance for debug output.
static void SetEnvVar(const char *env_var_name, const char *value, Log *log) {
  if (!sys::Process::GetEnv(env_var_name)) {
    setenv(env_var_name, value, 1);
    LLDB_LOG(log, "Set {}={}", env_var_name, value);
  } else {
    LLDB_LOG(log, "Using existing {} from environment", env_var_name);
  }
}

LLDBServerPluginNVGPU::LLDBServerPluginNVGPU(
    LLDBServerPlugin::GDBServer &native_process, MainLoop &main_loop)
    : LLDBServerPlugin(native_process, main_loop) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU initializing...");

  // We set this variable to avoid JITing, which simplifies module loading.
  SetEnvVar("CUDA_MODULE_LOADING", "EAGER", log);
  // Set environment variables from CMake configuration if they were defined
#ifdef CMAKE_NVGPU_CUDBG_INJECTION_PATH
  SetEnvVar("CUDBG_INJECTION_PATH", CMAKE_NVGPU_CUDBG_INJECTION_PATH, log);
#endif
#ifdef CMAKE_NVGPU_CUDA_VISIBLE_DEVICES
  SetEnvVar("CUDA_VISIBLE_DEVICES", CMAKE_NVGPU_CUDA_VISIBLE_DEVICES, log);
#endif
#ifdef CMAKE_NVGPU_CUDA_DEVICE_ORDER
  SetEnvVar("CUDA_DEVICE_ORDER", CMAKE_NVGPU_CUDA_DEVICE_ORDER, log);
#endif
#ifdef CMAKE_NVGPU_CUDA_LAUNCH_BLOCKING
  SetEnvVar("CUDA_LAUNCH_BLOCKING", CMAKE_NVGPU_CUDA_LAUNCH_BLOCKING, log);
#endif

  m_process_manager_up.reset(new ProcessNVGPU::Manager(main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      main_loop, *m_process_manager_up, "nvgpu.server"));

  // During initialization, there might be no cubins loaded, so we don't have
  // anything tangible to use as the identifier or file for the GPU process.
  // Thus, we create a fake process and we pretend we just launched it.
  ProcessLaunchInfo info;
  info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                      eLaunchFlagDisableASLR);
  Args args;
  args.AppendArgument("/pretend/path/to/NVGPU");
  info.SetArguments(args, true);
  info.GetEnvironment() = Host::GetEnvironment();
  m_gdb_server->SetLaunchInfo(info);
  Status error = m_gdb_server->LaunchProcess();
  m_gpu = static_cast<ProcessNVGPU *>(m_gdb_server->GetCurrentProcess());

  // The GPU process is fake and shouldn't fail to launch. Let's abort if we see
  // an error.
  if (error.Fail())
    logAndReportFatalError("Failed to launch the GPU process. {}", error);
}

llvm::StringRef LLDBServerPluginNVGPU::GetPluginName() { return "nvgpu"; }

std::optional<GPUActions> LLDBServerPluginNVGPU::NativeProcessIsStopping() {
  return {};
}

void LLDBServerPluginNVGPU::AcceptAndMainLoopThread(
    std::unique_ptr<TCPSocket> listen_socket_up) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread spawned");
  Socket *socket = nullptr;
  Status error = listen_socket_up->Accept(std::chrono::seconds(30), socket);
  // Scope for lock guard.
  {
    // Protect access to m_is_listening and m_is_connected.
    std::lock_guard<std::mutex> guard(m_connect_mutex);
    m_is_listening = false;
    if (error.Fail())
      logAndReportFatalError(
          "LLDBServerPluginNVGPU::AcceptAndMainLoopThread error "
          "returned from Accept(): {}",
          error);
    m_is_connected = true;
  }

  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread initializing "
                "connection");
  std::unique_ptr<Connection> connection_up(
      new ConnectionFileDescriptor(std::unique_ptr<Socket>(socket)));
  m_gdb_server->InitializeConnection(std::move(connection_up));
  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread running main "
                "loop");
  m_main_loop_status = m_main_loop.Run();
  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread main loop "
                "exited!");
  if (m_main_loop_status.Fail()) {
    logAndReportFatalError(
        "LLDBServerPluginNVGPU::AcceptAndMainLoopThread main loop "
        "exited with an error: {}",
        m_main_loop_status);
  }
  // Protect access to m_is_connected.
  std::lock_guard<std::mutex> guard(m_connect_mutex);
  m_is_connected = false;
}

Expected<GPUPluginConnectionInfo> LLDBServerPluginNVGPU::CreateConnection() {
  std::lock_guard<std::mutex> guard(m_connect_mutex);
  Log *log = GetLog(GDBRLog::Plugin);
  if (m_is_connected) {
    return createStringError("LLDBServerPluginNVGPU::CreateConnection error: "
                             "already connected");
  }
  if (m_is_listening) {
    return createStringError("LLDBServerPluginNVGPU::CreateConnection error: "
                             "already listening");
  }
  m_is_listening = true;
  // The following variables help us to establish connections for remote
  // platforms. It should be possible to automate them, but that requires
  // exposing the connection information of lldb-platform, which is a
  // good amount of work. Let's do that only when we really need it.
  const uint16_t listen_to_port =
      std::stoi(sys::Process::GetEnv("NVGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT")
                    .value_or("0"));
  std::string listen_to_host =
      sys::Process::GetEnv("NVGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST")
          .value_or("localhost");
  std::string remote_host =
      sys::Process::GetEnv("NVGPU_DEBUGGER_REMOTE_HOST").value_or("localhost");

  std::string listen_to_host_and_port =
      llvm::formatv("{}:{}", listen_to_host, listen_to_port);
  llvm::Expected<std::unique_ptr<TCPSocket>> sock =
      Socket::TcpListen(listen_to_host_and_port, 5);
  if (sock) {
    GPUPluginConnectionInfo connection_info;
    connection_info.copy_cpu_breakpoints_during_attaching = true;
    connection_info.should_step_over_breakpoints_on_resume = false;
    // connection_info.exe_path = "/pretend/path/to/NVGPU";
    connection_info.triple = "nvptx-nvidia-cuda";
    const uint16_t listen_port = (*sock)->GetLocalPortNumber();
    connection_info.connect_url =
        llvm::formatv("connect://{}:{}", remote_host, listen_port);
    LLDB_LOG(log, "LLDBServerPluginNVGPU::CreateConnection listening to {}",
             listen_port);
    std::thread t(&LLDBServerPluginNVGPU::AcceptAndMainLoopThread, this,
                  std::move(*sock));
    t.detach();
    return connection_info;
  }
  m_is_listening = false;
  return createStringErrorFmt("LLDBServerPluginNVGPU::CreateConnection error: "
                              "failed to listen to localhost:0: {}",
                              llvm::toString(sock.takeError()));
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  std::string library_name = *args.breakpoint.name_info->shlib;
  // This method is invoked when a CPU breakpoint is hit signaling that the
  // driver is initializing. This is the perfect time to initialize the debugger
  // API. We are assuming that no kernels will run until we resume the CPU.
  Expected<CUDADebuggerAPI> api_or = CUDADebuggerAPI::Initialize(
      args, library_name, *m_native_process.GetCurrentProcess());
  if (!api_or)
    return api_or.takeError();

  m_cuda_api = std::move(*api_or);
  this->m_gpu->SetDebuggerAPI(*m_cuda_api);

  // We are registering the event notifier in the GPU main loop. We might want
  // to use the CPU main loop at some point if needed.
  Expected<MainLoopEventNotifierUP> main_loop_event_notifier =
      MainLoopEventNotifier::CreateForEventCallback(
          "CUDA Debugger API event notifier", m_main_loop,
          [this]() { OnDebuggerAPIEvent(); });
  if (!main_loop_event_notifier)
    return main_loop_event_notifier.takeError();
  m_main_loop_event_notifier_up = std::move(*main_loop_event_notifier);

  CUDBGResult res = (*m_cuda_api)
                        ->setNotifyNewEventCallback31(
                            [](void *data) {
                              Log *log = GetLog(GDBRLog::Plugin);
                              LLDB_LOG(
                                  log,
                                  "CUDA Debugger API event notifier callback");
                              static_cast<LLDBServerPluginNVGPU *>(data)
                                  ->m_main_loop_event_notifier_up->FireEvent();
                            },
                            this);
  if (res != CUDBG_SUCCESS)
    return createStringError(
        "Failed to set the event callback for the CUDA Debugger API. {}",
        cudbgGetErrorString(res));

  GPUPluginBreakpointHitResponse response(GetPluginName(), kGpuInitializationBreakpoint);

  Expected<GPUPluginConnectionInfo> connection_info = CreateConnection();
  if (!connection_info)
    return connection_info.takeError();

  response.actions.connect_info = std::move(*connection_info);
  response.disable_bp = true;
  return response;
}

GPUActions LLDBServerPluginNVGPU::GetInitializeActions() {
  GPUActions init_actions(GetPluginName());

  init_actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetInitializationBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME));
  init_actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetInitializationBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME_ALT));
  return init_actions;
}

void LLDBServerPluginNVGPU::OnDebuggerAPIEvent() {
  Log *log = GetLog(GDBRLog::Plugin);
  CUDBGEvent event;
  CUDBGResult res;
  CUDADebuggerAPI &cuda_api = *m_cuda_api;
  LLDB_LOG(log, "LLDBServerPluginNVGPU::OnDebuggerAPIEvent");

  res = cuda_api->getNextEvent(CUDBGEventQueueType::CUDBG_EVENT_QUEUE_TYPE_SYNC,
                               &event);
  if (res == CUDBGResult::CUDBG_ERROR_NO_EVENT_AVAILABLE) {
    // We shouldn't be getting spurious calls to this function, so all
    // invocations should have a corresponding event.
    logAndReportFatalError(
        "We didnt' get an event from the CUDA Debugger API queue. {}",
        cudbgGetErrorString(res));
  }

  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("Failed to get the next CUDA Debugger API event. {}",
                           cudbgGetErrorString(res));
  }

  switch (event.kind) {
  case CUDBG_EVENT_ELF_IMAGE_LOADED: {
    LLDB_LOG(log, "CUDBG_EVENT_ELF_IMAGE_LOADED");
    // When we get an elf file, we report a dyld stop to the client.
    // We hold ack'ing this event until we have gotten the autoresume from the
    // client. This will need to be changed once we support multiple contexes.
    m_gpu->OnElfImageLoaded(event.cases.elfImageLoaded);
    m_gpu->ReportDyldStop();
    return;
  }
  case CUDBG_EVENT_KERNEL_READY: {
    LLDB_LOG(log, "CUDBG_EVENT_KERNEL_READY");
    break;
  }
  case CUDBG_EVENT_KERNEL_FINISHED: {
    LLDB_LOG(log, "CUDBG_EVENT_KERNEL_FINISHED");
    break;
  }
  case CUDBG_EVENT_INTERNAL_ERROR: {
    LLDB_LOG(log, "CUDBG_EVENT_INTERNAL_ERROR");
    break;
  }
  case CUDBG_EVENT_CTX_PUSH: {
    LLDB_LOG(log, "CUDBG_EVENT_CTX_PUSH");
    break;
  }
  case CUDBG_EVENT_CTX_POP: {
    LLDB_LOG(log, "CUDBG_EVENT_CTX_POP");
    break;
  }
  case CUDBG_EVENT_CTX_CREATE: {
    LLDB_LOG(log, "CUDBG_EVENT_CTX_CREATE");
    break;
  }
  case CUDBG_EVENT_CTX_DESTROY: {
    LLDB_LOG(log, "CUDBG_EVENT_CTX_DESTROY");
    break;
  }
  case CUDBG_EVENT_TIMEOUT: {
    LLDB_LOG(log, "CUDBG_EVENT_TIMEOUT");
    break;
  }
  case CUDBG_EVENT_ATTACH_COMPLETE: {
    LLDB_LOG(log, "CUDBG_EVENT_ATTACH_COMPLETE");
    break;
  }
  case CUDBG_EVENT_DETACH_COMPLETE: {
    LLDB_LOG(log, "CUDBG_EVENT_DETACH_COMPLETE");
    break;
  }
  case CUDBG_EVENT_ELF_IMAGE_UNLOADED: {
    LLDB_LOG(log, "CUDBG_EVENT_ELF_IMAGE_UNLOADED");
    break;
  }
  case CUDBG_EVENT_FUNCTIONS_LOADED: {
    LLDB_LOG(log, "CUDBG_EVENT_FUNCTIONS_LOADED");
    break;
  }
  case CUDBG_EVENT_ALL_DEVICES_SUSPENDED: {
    LLDB_LOG(log, "CUDBG_EVENT_ALL_DEVICES_SUSPENDED {0:x} {1:x}",
             event.cases.allDevicesSuspended.brokenDevicesMask,
             event.cases.allDevicesSuspended.faultedDevicesMask);
    auto log_to_client_callback = [this](llvm::StringRef message) {
      // The structured data packet can only be sent when the client is waiting
      // for the stop reply packet. Otherwise, it might think that this is the
      // response to a pending query packet. Creating the callback at this point
      // is safe because we are about to report that the state is stopped, which
      // means that we are running.
      if (m_gpu->GetState() != lldb::eStateRunning) {
        logAndReportFatalError(
            "Logging to client is only supported when the GPU is running.");
      }

      m_gdb_server->SendStructuredDataPacket(
          llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                               {"subtype", "log"},
                                               {"message", message}}));
    };
    // Order is important here. We need to suspend the GPU first before the
    // native process so that the CPU's GPUActions can hit the GPU server.
    m_gpu->OnAllDevicesSuspended(event.cases.allDevicesSuspended,
                                 log_to_client_callback);
    // Do not suspend the native process here. We do not want to force
    // the two processes to be sync'ed unless we have to. Allow for a non-stop
    // mode for the native process.
    break;
  }
  case CUDBG_EVENT_INVALID: {
    LLDB_LOG(log, "CUDBG_EVENT_INVALID");
    break;
  }
  default:
    LLDB_LOG(log, "Unknown event kind: {}", event.kind);
    break;
  }

  LLDB_LOG(log, "Done servicing CUDA API events");

  // Handled all pending events. Acknowledge them.
  res = cuda_api->acknowledgeSyncEvents();
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("Failed to acknowledge CUDA Debugger API events. {}",
                           cudbgGetErrorString(res));
  }
}

void LLDBServerPluginNVGPU::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  if (m_gpu)
    m_gpu->OnNativeProcessExit(exit_status);
}
