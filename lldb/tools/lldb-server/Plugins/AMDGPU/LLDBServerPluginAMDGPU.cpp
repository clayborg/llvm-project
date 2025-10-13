//===-- LLDBServerPluginAMDGPU.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginAMDGPU.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessAMDGPU.h"
#include "ThreadAMDGPU.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"

#include "Plugins/Utils/Utils.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <cinttypes>
#include <cstdint>
#include <optional>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

static amd_dbgapi_status_t amd_dbgapi_client_process_get_info_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_client_process_info_t query, size_t value_size, void *value) {
  LLDBServerPluginAMDGPU *debugger =
      reinterpret_cast<LLDBServerPluginAMDGPU *>(client_process_id);
  lldb::pid_t pid = debugger->GetNativeProcess()->GetID();
  LLDB_LOGF(GetLog(GDBRLog::Plugin),
            "amd_dbgapi_client_process_get_info_callback callback, with query "
            "%d, pid %lu",
            query, (unsigned long)pid);
  switch (query) {
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID: {
    if (value_size != sizeof(amd_dbgapi_os_process_id_t))
      return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;
    *static_cast<amd_dbgapi_os_process_id_t *>(value) = pid;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE: {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  }
  return AMD_DBGAPI_STATUS_SUCCESS;
}

// Internal name used to identify the gpu loader breakpoint.
// We match this name in the GPUBreakpointInfo passed into the
// `BreakpointWasHit` callback to identify which breakpoint was hit.
static constexpr uint32_t kGpuLoaderBreakpointIdentifier = 1;

// Set the internal gpu breakpoint by symbol name instead of using the address
// passed into the `insert_breakpoint` callback. The ROCdbgapi library
// uses the `amd_dbgapi_insert_breakpoint_callback` to communicate the address
// where the breakpoint should be set to catch all changes to the loaded code
// objects (e.g. when a new kernel is loaded). The callback is triggered during
// the processing of the call to the `amd_dbgapi_process_next_pending_event`
// that handles the AMD_DBGAPI_EVENT_KIND_RUNTIME event type when the runtime
// is first loaded. Instead of trying to set the breakpoint on demand at a
// time when the cpu is running, it is easier to set the breakpoint when we
// create the connection to a known symbol name. Otherwise, we have to halt the
// cpu process which shows a public stop to the user. See PR for more discussion
// on the issues and alternatives:
// https://github.com/clayborg/llvm-project/pull/20
static const char *kSetDbgApiBreakpointByName =
    "_ZN4rocr19_loader_debug_stateEv"; // rocr::_loader_debug_state

static amd_dbgapi_status_t amd_dbgapi_insert_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t address,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "insert_breakpoint callback at address: 0x%" PRIx64, address);

  LLDBServerPluginAMDGPU *debugger =
      reinterpret_cast<LLDBServerPluginAMDGPU *>(client_process_id);
  LLDBServerPluginAMDGPU::GPUInternalBreakpoinInfo bp_info;
  bp_info.addr = address;
  bp_info.breakpoind_id = breakpoint_id;
  debugger->m_gpu_internal_bp.emplace(std::move(bp_info));

  if (kSetDbgApiBreakpointByName) {
    LLDB_LOGF(log,
              "ignoring breakpoint address 0x%" PRIx64
              " and using name '%s' instead",
              address, kSetDbgApiBreakpointByName);
  } else {
    debugger->GetNativeProcess()->Halt();
    debugger->m_wait_for_gpu_internal_bp_stop = true;
  }
  return AMD_DBGAPI_STATUS_SUCCESS;
}

/* remove_breakpoint callback.  */

static amd_dbgapi_status_t amd_dbgapi_remove_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "remove_breakpoint callback for %" PRIu64,
            breakpoint_id.handle);
  return AMD_DBGAPI_STATUS_SUCCESS;
}

/* xfer_global_memory callback.  */

static amd_dbgapi_status_t amd_dbgapi_xfer_global_memory_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t global_address, amd_dbgapi_size_t *value_size,
    void *read_buffer, const void *write_buffer) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "xfer_global_memory callback");

  return AMD_DBGAPI_STATUS_SUCCESS;
}

static void amd_dbgapi_log_message_callback(amd_dbgapi_log_level_t level,
                                            const char *message) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "ROCdbgapi [%d]: %s", level, message);
}

static amd_dbgapi_callbacks_t s_dbgapi_callbacks = {
    malloc,
    free,
    amd_dbgapi_client_process_get_info_callback,
    amd_dbgapi_insert_breakpoint_callback,
    amd_dbgapi_remove_breakpoint_callback,
    amd_dbgapi_xfer_global_memory_callback,
    amd_dbgapi_log_message_callback,
};

LLDBServerPluginAMDGPU::LLDBServerPluginAMDGPU(
    LLDBServerPlugin::GDBServer &native_process, MainLoop &main_loop)
    : LLDBServerPlugin(native_process, main_loop) {
  m_process_manager_up.reset(new ProcessManagerAMDGPU(main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      m_main_loop, *m_process_manager_up, "amd-gpu.server"));

  Status error = InitializeAmdDbgApi();
  if (error.Fail()) {
    logAndReportFatalError("{} Failed to initialize debug library: {}",
                           __FUNCTION__, error);
  }
}

LLDBServerPluginAMDGPU::~LLDBServerPluginAMDGPU() { }

llvm::StringRef LLDBServerPluginAMDGPU::GetPluginName() { return "amd-gpu"; }

Status LLDBServerPluginAMDGPU::InitializeAmdDbgApi() {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "%s called", __FUNCTION__);

  // Initialize AMD Debug API with callbacks
  amd_dbgapi_status_t status = amd_dbgapi_initialize(&s_dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return Status::FromErrorStringWithFormat(
        "Failed to initialize AMD debug API: %d", status);
  }

  m_amd_dbg_api_state = AmdDbgApiState::Initialized;
  return Status();
}

Status LLDBServerPluginAMDGPU::AttachAmdDbgApi() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __FUNCTION__);
  // Attach to the process with AMD Debug API
  amd_dbgapi_status_t status = amd_dbgapi_process_attach(
      reinterpret_cast<amd_dbgapi_client_process_id_t>(
          this), // Use pid_ as client_process_id
      &m_gpu_pid);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return HandleAmdDbgApiAttachError(
        "Failed to attach to process with AMD debug API", status);
  }
  m_amd_dbg_api_state = AmdDbgApiState::Attached;

  // Get the process notifier
  status =
      amd_dbgapi_process_get_info(m_gpu_pid, AMD_DBGAPI_PROCESS_INFO_NOTIFIER,
                                  sizeof(m_notifier_fd), &m_notifier_fd);

  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return HandleAmdDbgApiAttachError("Failed to get process notifier", status);
  }
  LLDB_LOGF(log, "Process notifier fd: %d", m_notifier_fd);

  Status error = InstallAmdDbgApiNotifierOnMainLoop();
  if (error.Fail()) {
    return HandleAmdDbgApiAttachError(error.AsCString(),
                                      AMD_DBGAPI_STATUS_ERROR);
  }

  // TODO: read the architecture from the attached agent.
  amd_dbgapi_architecture_id_t architecture_id;
  const uint32_t elf_amdgpu_machine = 0x04C;
  status = amd_dbgapi_get_architecture(elf_amdgpu_machine, &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return HandleAmdDbgApiAttachError("Failed to get architecture", status);
  }
  m_architecture_id = architecture_id;

  // The process from LLDB’s PoV is the entire portion of logical GPU bound to
  // the CPU host process. It does not represent a real process on the GPU. The
  // The GPU process is fake and shouldn't fail to launch. Let's abort if we see
  // an error.
  error = CreateGpuProcess();
  if (error.Fail()) {
    return HandleAmdDbgApiAttachError(error.AsCString(),
                                      AMD_DBGAPI_STATUS_ERROR);
  }

  // Process all pending events
  LLDB_LOGF(log, "%s Processing any pending dbgapi events", __FUNCTION__);
  amd_dbgapi_event_id_t eventId;
  amd_dbgapi_event_kind_t eventKind;
  if (amd_dbgapi_process_next_pending_event(m_gpu_pid, &eventId, &eventKind) ==
      AMD_DBGAPI_STATUS_SUCCESS) {
    GetGPUProcess()->handleDebugEvent(eventId, eventKind);

    // Mark event as processed
    amd_dbgapi_event_processed(eventId);
  }

  LLDB_LOGF(log, "%s successfully attached to debug library", __FUNCTION__);
  return Status();
}

Status
LLDBServerPluginAMDGPU::HandleAmdDbgApiAttachError(llvm::StringRef error_msg,
                                                   amd_dbgapi_status_t status) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __FUNCTION__);
  // Cleanup any partial initialization
  if (m_amd_dbg_api_state == AmdDbgApiState::Attached) {
    Status error = DetachAmdDbgApi();
    if (error.Fail()) {
      LLDB_LOGF(log, "%s: failed DetachAmdDbgApi: %s", __FUNCTION__,
                error.AsCString());
    }
  }

  // Finalize the AMD Debug API
  amd_dbgapi_status_t finalize_status = amd_dbgapi_finalize();
  if (finalize_status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_finalize failed: %d",
              finalize_status);
  }

  // Reset the state to error state
  m_amd_dbg_api_state = AmdDbgApiState::Error;

  // Return a status with the error message and additional context
  return Status::FromErrorStringWithFormat(
      "AMD Debug API attach failed: %s (status: %d)", error_msg.data(), status);
}

Status LLDBServerPluginAMDGPU::DetachAmdDbgApi() {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "%s called", __FUNCTION__);

  if (m_amd_dbg_api_state == AmdDbgApiState::Detached) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "%s -- already detached from process",
              __FUNCTION__);
    return Status();
  }

  amd_dbgapi_status_t status = amd_dbgapi_process_detach(m_gpu_pid);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return Status::FromErrorStringWithFormat(
        "Failed to detach from process: %d", status);
  }
  m_gpu_pid = AMD_DBGAPI_PROCESS_NONE;
  m_notifier_fd = INVALID_NOTIFIER_ID;

  m_amd_dbg_api_state = AmdDbgApiState::Detached;
  return Status();
}

bool LLDBServerPluginAMDGPU::processGPUEvent() {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "processGPUEvent");
  char buf[256];
  ssize_t bytesRead = 0;
  bool result = false;
  do {
    do {
      bytesRead = read(m_notifier_fd, buf, sizeof(buf));
    } while (bytesRead <= 0);

    auto *process = GetGPUProcess();
    process->m_wave_ids.clear();
    amd_dbgapi_status_t status = amd_dbgapi_process_set_progress(
        m_gpu_pid, AMD_DBGAPI_PROGRESS_NO_FORWARD);
    assert(status == AMD_DBGAPI_STATUS_SUCCESS);
    AmdDbgApiEventSet events = process_event_queue(AMD_DBGAPI_EVENT_KIND_NONE);
    if (events.HasWaveStopEvent()) {
      for (auto wave_id : process->m_wave_ids) {
        process->AddThread(wave_id);
      }
      process->Halt();
    }
    status =
        amd_dbgapi_process_set_progress(m_gpu_pid, AMD_DBGAPI_PROGRESS_NORMAL);
    assert(status == AMD_DBGAPI_STATUS_SUCCESS);
    break;
  } while (true);
  return result;
}

bool LLDBServerPluginAMDGPU::HandleEventFileDescriptorEvent(int fd) {
  return processGPUEvent();
}

std::optional<GPUPluginConnectionInfo>
LLDBServerPluginAMDGPU::CreateConnection() {
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
    connection_info.connect_url =
        llvm::formatv("connect://localhost:{}", listen_port);
    LLDB_LOGF(log, "%s listening to %u", __PRETTY_FUNCTION__, listen_port);

    // Store the socket in the member variable to keep it alive
    m_listen_socket = std::move(*sock);
    auto extra_args =
        llvm::formatv("gpu-url:connect://localhost:{};", listen_port);
    m_is_connected = false;
    llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> res =
        m_listen_socket->Accept(
            m_main_loop, [this](std::unique_ptr<Socket> socket) {
              Log *log = GetLog(GDBRLog::Plugin);
              LLDB_LOGF(log,
                        "LLDBServerPluginAMDGPU::CreateConnection() "
                        "initializing connection");
              std::unique_ptr<Connection> connection_up(
                  new ConnectionFileDescriptor(std::move(socket)));
              this->m_gdb_server->InitializeConnection(
                  std::move(connection_up));
              m_is_connected = true;
            });
    if (res) {
      m_read_handles = std::move(*res);
    } else {
      LLDB_LOGF(
          log,
          "LLDBServerPluginAMDGPU::GetConnectionURL() failed to accept: %s",
          llvm::toString(res.takeError()).c_str());
    }

    connection_info.synchronous = true;
    return connection_info;
  } else {
    std::string error = llvm::toString(sock.takeError());
    LLDB_LOGF(log, "%s failed to listen to localhost:0: %s",
              __PRETTY_FUNCTION__, error.c_str());
  }
  m_is_listening = false;
  return std::nullopt;
}

Status LLDBServerPluginAMDGPU::InstallAmdDbgApiNotifierOnMainLoop() {
  Status error;
  LLDBServerPluginAMDGPU *amdGPUPlugin = this;
  m_gpu_event_io_obj_sp = std::make_shared<GPUIOObject>(m_notifier_fd);
  m_gpu_event_read_up = m_main_loop.RegisterReadObject(
      m_gpu_event_io_obj_sp,
      [amdGPUPlugin](MainLoopBase &) {
        amdGPUPlugin->HandleEventFileDescriptorEvent(
            amdGPUPlugin->m_notifier_fd);
      },
      error);
  return error;
}

Status LLDBServerPluginAMDGPU::CreateGpuProcess() {
  ProcessManagerAMDGPU *manager =
      (ProcessManagerAMDGPU *)m_process_manager_up.get();
  manager->m_debugger = this;

  // During initialization, there might be no code objects loaded, so we don't
  // have anything tangible to use as the identifier or file for the GPU
  // process. Thus, we create a fake process and we pretend we just launched it.
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s faking launch...", __FUNCTION__);
  ProcessLaunchInfo info;
  info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                      eLaunchFlagDisableASLR);
  Args args;
  args.AppendArgument("/pretend/path/to/amdgpu");
  info.SetArguments(args, true);
  info.GetEnvironment() = Host::GetEnvironment();
  info.SetProcessID(m_gpu_pid.handle);
  m_gdb_server->SetLaunchInfo(info);

  return m_gdb_server->LaunchProcess();
}

bool LLDBServerPluginAMDGPU::ReadyToSendConnectionRequest() {
  // We are ready to send a connection request if we are attached to the debug
  // library and we have not yet sent a connection request.
  // The GPUActions are ignored on the initial stop when the process is first
  // launched so we wait until the second stop to send the connection request.
  const bool ready = m_amd_dbg_api_state == AmdDbgApiState::Attached &&
                     !m_is_connected && !m_is_listening &&
                     GetNativeProcess()->GetStopID() > 1;

  LLDB_LOGF(GetLog(GDBRLog::Plugin),
            "%s - ready: %d dbg_api_state: %d, connected: %d, listening: %d, "
            "native-stop-id: %d",
            __FUNCTION__, ready, (int)m_amd_dbg_api_state, m_is_connected,
            m_is_listening, GetNativeProcess()->GetStopID());
  return ready;
}

bool LLDBServerPluginAMDGPU::ReadyToAttachDebugLibrary() {
  return m_amd_dbg_api_state == AmdDbgApiState::Initialized;
}

bool LLDBServerPluginAMDGPU::ReadyToSetGpuLoaderBreakpointByAddress() {
  return m_wait_for_gpu_internal_bp_stop && m_gpu_internal_bp.has_value();
}

GPUActions LLDBServerPluginAMDGPU::SetConnectionInfo() {
  GPUActions actions;
  actions.plugin_name = GetPluginName();
  actions.connect_info = CreateConnection();
  return actions;
}

GPUActions LLDBServerPluginAMDGPU::SetGpuLoaderBreakpointByAddress() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s Requesting gpu breakpoint at 0x%p", __FUNCTION__,
            (void *)m_gpu_internal_bp->addr);
  assert(!kSetDbgApiBreakpointByName);
  GPUActions actions;
  actions.plugin_name = GetPluginName();

  GPUBreakpointByAddress bp_addr;
  bp_addr.load_address = m_gpu_internal_bp->addr;

  GPUBreakpointInfo bp;
  bp.identifier = kGpuLoaderBreakpointIdentifier;
  bp.addr_info.emplace(bp_addr);

  std::vector<GPUBreakpointInfo> breakpoints;
  breakpoints.emplace_back(std::move(bp));

  actions.breakpoints = std::move(breakpoints);
  m_wait_for_gpu_internal_bp_stop = false;
  return actions;
}

std::optional<GPUActions> LLDBServerPluginAMDGPU::NativeProcessIsStopping() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __FUNCTION__);

  if (ReadyToAttachDebugLibrary()) {
    Status error = AttachAmdDbgApi();
    if (error.Fail()) {
      LLDB_LOGF(log, "%s failed to attach debug library: %s", __FUNCTION__,
                error.AsCString());
      return std::nullopt;
    }
  }

  if (ReadyToSendConnectionRequest()) {
    return SetConnectionInfo();
  }

  if (ReadyToSetGpuLoaderBreakpointByAddress()) {
    return SetGpuLoaderBreakpointByAddress();
  }
  return std::nullopt;
}

bool LLDBServerPluginAMDGPU::HandleGPUInternalBreakpointHit(
    const GPUInternalBreakpoinInfo &bp) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "Hit %u at address: 0x%" PRIx64,
            kGpuLoaderBreakpointIdentifier, bp.addr);
  amd_dbgapi_breakpoint_id_t breakpoint_id{bp.breakpoind_id};
  amd_dbgapi_breakpoint_action_t action;

  auto status = amd_dbgapi_report_breakpoint_hit(
      breakpoint_id, reinterpret_cast<amd_dbgapi_client_thread_id_t>(this),
      &action);

  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "amd_dbgapi_report_breakpoint_hit failed: %d", status);
    return false;
  }

  if (action == AMD_DBGAPI_BREAKPOINT_ACTION_RESUME) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "AMD_DBGAPI_BREAKPOINT_ACTION_RESUME");
    return true;
  } else if (action == AMD_DBGAPI_BREAKPOINT_ACTION_HALT) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "AMD_DBGAPI_BREAKPOINT_ACTION_HALT");

    AmdDbgApiEventSet events =
        process_event_queue(AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);
    assert(events.HasBreakpointResumeEvent());
    return true;
  } else {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Unknown action: %d", action);
    return false;
  }
  return true;
}

AmdDbgApiEventSet LLDBServerPluginAMDGPU::process_event_queue(
    amd_dbgapi_event_kind_t until_event_kind,
    EventBoundaryType event_boundary) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "Processing event queue");
  AmdDbgApiEventSet events;
  while (true) {
    amd_dbgapi_event_id_t event_id;
    amd_dbgapi_event_kind_t event_kind;
    amd_dbgapi_status_t status = amd_dbgapi_process_next_pending_event(
        m_gpu_pid, &event_id, &event_kind);

    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "amd_dbgapi_process_next_pending_event failed: %d", status);
      break;
    }

    // We will handle this event if it is not the target event kind or if
    // it is the target event kind and we were told to handle it inclusively.
    const bool handle_this_event = event_kind != until_event_kind ||
                                   event_boundary == ProcessEventInclusive;
    events.SetLastEvent(event_id, event_kind);
    if (handle_this_event) {
      GetGPUProcess()->handleDebugEvent(event_id, event_kind);
    }

    if (event_kind == until_event_kind)
      break;
  }

  LLDB_LOGF(GetLog(GDBRLog::Plugin), "Processed events: %s",
            events.ToString().c_str());
  return events;
}

void LLDBServerPluginAMDGPU::FreeDbgApiClientMemory(void *mem) {
  s_dbgapi_callbacks.deallocate_memory(mem);
}

void LLDBServerPluginAMDGPU::GpuRuntimeDidLoad() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "%s called", __FUNCTION__);
  if (m_amd_dbg_api_state == AmdDbgApiState::RuntimeLoaded) {
    LLDB_LOGF(log, "%s -- runtime loaded event already handled", __FUNCTION__);
    return;
  }

  m_amd_dbg_api_state = AmdDbgApiState::RuntimeLoaded;
}

bool LLDBServerPluginAMDGPU::SetGPUBreakpoint(uint64_t addr,
                                              const uint8_t *bp_instruction,
                                              size_t size) {
  struct BreakpointInfo {
    uint64_t addr;
    std::vector<uint8_t> original_bytes;
    std::vector<uint8_t> breakpoint_instruction;
    std::optional<amd_dbgapi_breakpoint_id_t> gpu_breakpoint_id;
  };

  BreakpointInfo bp;
  bp.addr = addr;
  bp.breakpoint_instruction.assign(bp_instruction, bp_instruction + size);
  bp.original_bytes.resize(size);
  bp.gpu_breakpoint_id =
      std::nullopt; // No GPU breakpoint ID for ptrace version

  // TODO: use memory read/write API from native process instead of ptrace
  // directly.
  auto pid = GetNativeProcess()->GetID();
  // Read original bytes word by word
  std::vector<long> original_words;
  for (size_t i = 0; i < size; i += sizeof(long)) {
    long word = ptrace(PTRACE_PEEKDATA, pid, addr + i, nullptr);
    assert(word != -1 && errno == 0);

    original_words.push_back(word);
    // Copy bytes from the word into our original_bytes
    size_t bytes_to_copy = std::min(sizeof(long), size - i);
    memcpy(&bp.original_bytes[i], &word, bytes_to_copy);
  }

  // Write breakpoint instruction word by word
  for (size_t i = 0; i < size; i += sizeof(long)) {
    long word = original_words[i / sizeof(long)];
    size_t bytes_to_copy = std::min(sizeof(long), size - i);
    memcpy(&word, &bp_instruction[i], bytes_to_copy);

    auto ret = ptrace(PTRACE_POKEDATA, pid, addr + i, word);
    assert(ret != -1 && errno == 0);
  }
  return true;
}

bool LLDBServerPluginAMDGPU::CreateGPUBreakpoint(uint64_t addr) {
  // Get breakpoint instruction
  const uint8_t *bp_instruction;
  amd_dbgapi_status_t status = amd_dbgapi_architecture_get_info(
      m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
      sizeof(bp_instruction), &bp_instruction);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION failed");
    return false;
  }

  // Get breakpoint instruction size
  size_t bp_size;
  status = amd_dbgapi_architecture_get_info(
      m_architecture_id,
      AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE, sizeof(bp_size),
      &bp_size);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(
        GetLog(GDBRLog::Plugin),
        "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE failed");
    return false;
  }

  // Now call SetGPUBreakpoint with the retrieved instruction and size
  return SetGPUBreakpoint(addr, bp_instruction, bp_size);
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginAMDGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::string json_string;
  const auto bp_identifier = args.breakpoint.identifier;
  llvm::raw_string_ostream os(json_string);
  os << toJSON(args);
  LLDB_LOGF(log, "LLDBServerPluginAMDGPU::BreakpointWasHit(\"%d\"):\nJSON:\n%s",
            bp_identifier, json_string.c_str());

  GPUPluginBreakpointHitResponse response(GetNewGPUAction());
  if (bp_identifier == kGpuLoaderBreakpointIdentifier) {
    // Make sure the breakpoint address matches the expected value when we set
    // it by name.
    if (kSetDbgApiBreakpointByName &&
        args.symbol_values[0].value != m_gpu_internal_bp->addr) {
      LLDB_LOGF(
          log,
          "Breakpoint %s (0x%" PRIx64
          ") does not match expected breakpoint address value: 0x%" PRIx64,
          args.breakpoint.symbol_names[0].c_str(),
          args.symbol_values[0].value.value_or(0), m_gpu_internal_bp->addr);

      return llvm::createStringError(
          "Breakpoint address does not match expected value from ROCdbgapi");
    }

    bool success = HandleGPUInternalBreakpointHit(m_gpu_internal_bp.value());
    assert(success);
    if (GetGPUProcess()->HasDyldChangesToReport()) {
      auto *process = m_gdb_server->GetCurrentProcess();
      if (process->IsRunning()) {
        response.actions.wait_for_gpu_process_to_resume = true;
        response.actions.stop_id = GetGPUProcess()->GetNextStopID();
        ThreadAMDGPU *thread = (ThreadAMDGPU *)process->GetCurrentThread();
        thread->SetStopReason(lldb::eStopReasonDynamicLoader);
        process->Halt();
      }
    }
  }
  return response;
}

GPUActions LLDBServerPluginAMDGPU::GetInitializeActions() {
  GPUActions init_actions;
  init_actions.plugin_name = GetPluginName();

  if (kSetDbgApiBreakpointByName) {
    GPUBreakpointByName bp_name;
    bp_name.function_name = kSetDbgApiBreakpointByName;

    GPUBreakpointInfo bp;
    bp.identifier = kGpuLoaderBreakpointIdentifier;
    bp.name_info = std::move(bp_name);
    bp.symbol_names = {kSetDbgApiBreakpointByName};
    init_actions.breakpoints.emplace_back(std::move(bp));
  }

  return init_actions;
}
