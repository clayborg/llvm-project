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
#include "lldb/Utility/State.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

#include <cerrno>
#include <chrono>
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

namespace {
/// Whether a GPU stop, at a breakpoint, an exception or an internal error,
/// stops the native process too. NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1 leaves it
/// running.
bool CPUStopsWithGPU() {
  return sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1";
}

StringRef EventKindName(CUDBGEventKind kind) {
#define CUDBG_EVENT_CASE(x)                                                    \
  case x:                                                                      \
    return #x;
  switch (kind) {
    CUDBG_EVENT_CASE(CUDBG_EVENT_INVALID)
    CUDBG_EVENT_CASE(CUDBG_EVENT_ELF_IMAGE_LOADED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_KERNEL_READY)
    CUDBG_EVENT_CASE(CUDBG_EVENT_KERNEL_FINISHED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_INTERNAL_ERROR)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CTX_PUSH)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CTX_POP)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CTX_CREATE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CTX_DESTROY)
    CUDBG_EVENT_CASE(CUDBG_EVENT_TIMEOUT)
    CUDBG_EVENT_CASE(CUDBG_EVENT_ATTACH_COMPLETE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_DETACH_COMPLETE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_ELF_IMAGE_UNLOADED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_FUNCTIONS_LOADED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_ALL_DEVICES_SUSPENDED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CUDA_LOGS_AVAILABLE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED)
    CUDBG_EVENT_CASE(CUDBG_EVENT_SINGLE_STEP_COMPLETE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CUDA_LOGS_RULESET_CHANGED)
  }
#undef CUDBG_EVENT_CASE
  return "unknown CUDBGEventKind";
}
} // namespace

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

  m_gdb_server->SetPlugin(this);

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

  // The GPU process is fake and shouldn't fail to launch. Let's abort if we see
  // an error.
  if (error.Fail())
    logAndReportFatalError("Failed to launch the GPU process. {}", error);
}

LLDBServerPluginNVGPU::~LLDBServerPluginNVGPU() {
  if (m_gpu_thread.joinable()) {
    m_shutting_down = true;
    // The loop outlives this object, and only a callback on the loop's own
    // thread may terminate it.
    if (!m_main_loop.AddPendingCallback(
            [](MainLoopBase &loop) { loop.RequestTermination(); })) {
      // The callback is queued, but without the wakeup the loop might never
      // run it, so a thread that runs the loop might never end. It must not
      // outlive this object, which it uses. A thread that has not started the
      // loop does not start it, and stops waiting for the client, since
      // m_shutting_down is set.
      std::lock_guard<std::mutex> guard(m_connect_mutex);
      if (m_is_connected)
        logAndReportFatalError("~LLDBServerPluginNVGPU: could not wake the GPU "
                               "main loop to stop its thread");
    }
    m_gpu_thread.join();
  }
  // Only the driver's event callback can still reach this object.
  ReleaseDebuggerAPI();
}

llvm::StringRef LLDBServerPluginNVGPU::GetPluginName() { return "nvgpu"; }

ProcessNVGPU *LLDBServerPluginNVGPU::GetGPUProcess() const {
  return static_cast<ProcessNVGPU *>(m_gdb_server->GetCurrentProcess());
}

std::optional<GPUActions> LLDBServerPluginNVGPU::NativeProcessIsStopping() {
  return {};
}

void LLDBServerPluginNVGPU::AcceptAndMainLoopThread(
    std::unique_ptr<TCPSocket> listen_socket_up) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread spawned");
  // Wait for lldb for as long as this object lives, in slices so that the
  // destructor does not wait long for a client that never connects.
  Socket *socket = nullptr;
  Status error;
  do {
    error = listen_socket_up->Accept(std::chrono::milliseconds(100), socket);
  } while (!socket && error.GetError() == ETIMEDOUT && !m_shutting_down);
  std::unique_ptr<Socket> socket_up(socket);
  bool connected = false;
  // Scope for lock guard.
  {
    // Protect access to m_is_listening and m_is_connected.
    std::lock_guard<std::mutex> guard(m_connect_mutex);
    m_is_listening = false;
    connected = socket_up && !m_shutting_down;
    m_is_connected = connected;
  }
  if (!connected) {
    if (m_shutting_down)
      LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread the "
                    "plugin is shutting down before lldb connected");
    else
      LLDB_LOG(log,
               "LLDBServerPluginNVGPU::AcceptAndMainLoopThread cannot accept "
               "lldb's connection, so the GPU cannot be debugged in this "
               "session: {0}",
               error);
    ReleaseUnservedDebuggerAPI();
    return;
  }

  LLDB_LOG(log, "LLDBServerPluginNVGPU::AcceptAndMainLoopThread initializing "
                "connection");
  std::unique_ptr<Connection> connection_up(
      new ConnectionFileDescriptor(std::move(socket_up)));
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

Expected<GPUPluginConnectionInfo>
LLDBServerPluginNVGPU::CreateConnection(function_ref<Error()> before_accept) {
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
    connection_info.triple =
        ProcessNVGPU::GetNVPTXArchitecture().GetTriple().str();
    const uint16_t listen_port = (*sock)->GetLocalPortNumber();
    connection_info.connect_url =
        llvm::formatv("connect://{}:{}", remote_host, listen_port);
    LLDB_LOG(log, "LLDBServerPluginNVGPU::CreateConnection listening to {}",
             listen_port);
    // An earlier connection's thread clears m_is_connected as its last step,
    // so it has finished or is about to.
    if (m_gpu_thread.joinable())
      m_gpu_thread.join();
    if (Error err = before_accept()) {
      // Closing the socket is enough: no thread is waiting on it yet.
      m_is_listening = false;
      return err;
    }
    m_gpu_thread = std::thread(&LLDBServerPluginNVGPU::AcceptAndMainLoopThread,
                               this, std::move(*sock));
    return connection_info;
  }
  m_is_listening = false;
  return createStringErrorFmt("LLDBServerPluginNVGPU::CreateConnection error: "
                              "failed to listen to localhost:0: {}",
                              llvm::toString(sock.takeError()));
}

Expected<GPUActions> LLDBServerPluginNVGPU::InitializeAPIAndConnect(
    const GPUPluginBreakpointHitArgs &args, StringRef libcuda_library_name,
    NativeProcessProtocol &native_process) {
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu)
    return createStringError(
        "No GPU process to initialize the CUDA Debugger API for");

  // We are assuming that no kernels will run until we resume the CPU.
  Expected<CUDADebuggerAPI> api_or =
      CUDADebuggerAPI::Initialize(args, libcuda_library_name, native_process);
  if (!api_or)
    return api_or.takeError();

  m_cuda_api = std::move(*api_or);
  // Undo the partial setup if a step below fails, so that nothing is left
  // pointing into an API the next initialization attempt replaces.
  auto release_api = llvm::make_scope_exit([this] { ReleaseDebuggerAPI(); });
  if (Error err = gpu->SetDebuggerAPI(*m_cuda_api))
    return err;

  // We are registering the event notifier in the GPU main loop. We might want
  // to use the CPU main loop at some point if needed.
  Expected<MainLoopEventNotifierUP> main_loop_event_notifier =
      MainLoopEventNotifier::CreateForEventCallback(
          "CUDA Debugger API event notifier", m_main_loop,
          [this]() { OnDebuggerAPIEvent(); });
  if (!main_loop_event_notifier)
    return main_loop_event_notifier.takeError();
  m_main_loop_event_notifier_up = std::move(*main_loop_event_notifier);

  // The driver only notifies for newly enqueued events, so the events that
  // waited while the GPU process was stopped would never be drained on their
  // own. Let the GPU process re-run the drain when it resumes.
  gpu->SetSyncEventDrainNotifier(
      [this]() { m_main_loop_event_notifier_up->FireEvent(); });
  gpu->SetDebuggerAPIFaultHandler([this](std::string description) {
    ReportDebuggerAPIFault(std::move(description));
  });

  CUDBGResult res =
      (*m_cuda_api)
          ->setNotifyNewEventCallback31(
              [](void *data) {
                Log *log = GetLog(GDBRLog::Plugin);
                LLDB_LOGV(log, "CUDA Debugger API event notifier callback");
                static_cast<LLDBServerPluginNVGPU *>(data)
                    ->m_main_loop_event_notifier_up->FireEvent();
              },
              this);
  if (res != CUDBG_SUCCESS)
    return createStringError(
        "Failed to set the event callback for the CUDA Debugger API. {}",
        cudbgGetErrorString(res));

  // Only tell the driver that a debugger is ready once every other step has
  // succeeded, the event callback included, so that it cannot emit a
  // notification before anything can observe it. If one fails, the client
  // resumes the process, which runs on without GPU debugging instead of
  // waiting for a debugger that does not answer.
  Expected<GPUPluginConnectionInfo> connection_info = CreateConnection(
      [&] { return CUDADebuggerAPI::SetIpcFlag(args, native_process); });
  if (!connection_info)
    return connection_info.takeError();

  GPUActions actions = GetNewGPUAction();
  actions.connect_info = std::move(*connection_info);
  actions.session_name = "NVIDIA GPU Session";
  release_api.release();
  return actions;
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  // The plug-in's breakpoints are each needed once.
  auto disabling_response = [](GPUActions actions) {
    GPUPluginBreakpointHitResponse response(std::move(actions));
    response.disable_bp = true;
    return response;
  };
  // Both libcuda sonames can resolve, so this fires more than once.
  // Initializing twice would build a second API table and reverse connection
  // and leak the live one.
  if (m_api_initialized) {
    Log *log = GetLog(GDBRLog::Plugin);
    LLDB_LOG(log, "debugger API already initialized; ignoring breakpoint and "
                  "disabling it.");
    return disabling_response(GetNewGPUAction());
  }

  const std::optional<GPUBreakpointByName> &name_info =
      args.breakpoint.name_info;
  if (!name_info || !name_info->shlib)
    return createStringError(
        "The breakpoint names no library to initialize the CUDA Debugger API "
        "from");
  StringRef library_name = *name_info->shlib;

  // Breakpoint hits arrive on the native server's thread, the only one that
  // may use its process, and the process stays stopped until this returns.
  NativeProcessProtocol *native_process = m_native_process.GetCurrentProcess();
  if (!native_process)
    return createStringError(
        "No native process to initialize the CUDA Debugger API for");

  Expected<GPUActions> actions =
      InitializeAPIAndConnect(args, library_name, *native_process);
  if (!actions) {
    // Leave m_api_initialized false so a later breakpoint can retry the init.
    return actions.takeError();
  }

  m_api_initialized = true;
  return disabling_response(std::move(*actions));
}

GPUActions LLDBServerPluginNVGPU::GetInitializeActions() {
  GPUActions init_actions = GetNewGPUAction();

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
  LLDB_LOGV(log, "LLDBServerPluginNVGPU::OnDebuggerAPIEvent");
  // The application has exited, and acknowledging its events can crash the
  // debugger library.
  if (m_native_process_exited)
    return;
  if (Expected<int> handled = DrainSyncEvents(); !handled)
    ReportDebuggerAPIFault(llvm::toString(handled.takeError()));
}

Expected<int> LLDBServerPluginNVGPU::DrainSyncEvents() {
  // Without a GPU process, which the GPU server forgets when lldb detaches
  // from it, nothing can handle the events. A poisoned API must not be touched
  // again.
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu || gpu->HasDebuggerAPIFaulted())
    return 0;

  Log *log = GetLog(GDBRLog::Plugin);
  CUDADebuggerAPI &cuda_api = *m_cuda_api;

  int events_handled = 0;
  // Handle events only while the GPU process runs, that is, while the client
  // waits for a stop. The class comment of ProcessNVGPU says why.
  while (StateIsRunningState(gpu->GetState())) {
    CUDBGEvent event;
    CUDBGResult res = cuda_api->getNextEvent(
        CUDBGEventQueueType::CUDBG_EVENT_QUEUE_TYPE_SYNC, &event);
    if (res == CUDBGResult::CUDBG_ERROR_NO_EVENT_AVAILABLE)
      break; // queue drained
    if (res != CUDBG_SUCCESS)
      return createStringErrorFmt(
          "Failed to get the next CUDA Debugger API event. {0}",
          cudbgGetErrorString(res));
    // The API also signals "no more events" with this sentinel kind.
    if (event.kind == CUDBG_EVENT_INVALID)
      break;

    ++events_handled;
    LLDB_LOG(log, "{0}", EventKindName(event.kind));

    switch (event.kind) {
    case CUDBG_EVENT_ELF_IMAGE_LOADED: {
      // Report a dyld stop to the client and stop draining: the events stay
      // unacknowledged until the client resumes, and Resume acknowledges them
      // and drains the rest. This will need to change once we support multiple
      // contexts.
      if (Error err = gpu->OnElfImageLoaded(event.cases.elfImageLoaded))
        return std::move(err);
      // A cubin that does not parse is skipped, which leaves nothing to load.
      if (!gpu->HasUnreportedLibraries())
        break;
      gpu->ReportDyldStop();
      return events_handled;
    }
    case CUDBG_EVENT_INTERNAL_ERROR: {
      // Acknowledging would go through the poisoned API.
      ReportDebuggerAPIFault(
          formatv("CUDA debugger API internal error: {0}",
                  cudbgGetErrorString(event.cases.internalError.errorType))
              .str());
      return events_handled;
    }
    case CUDBG_EVENT_ALL_DEVICES_SUSPENDED: {
      LLDB_LOG(log, "broken devices {0:x}, faulted devices {1:x}",
               event.cases.allDevicesSuspended.brokenDevicesMask,
               event.cases.allDevicesSuspended.faultedDevicesMask);
      Expected<ProcessNVGPU::SuspendStop> stop =
          gpu->OnAllDevicesSuspended(event.cases.allDevicesSuspended);
      if (!stop)
        return stop.takeError();
      // An interrupt can be lldb sending a packet, after which it resumes the
      // GPU at once, so the CPU keeps running.
      if (*stop == ProcessNVGPU::SuspendStop::Stopped && CPUStopsWithGPU())
        HaltNativeProcessIfNeeded();
      break;
    }
    default:
      // Logged above; nothing else to do.
      break;
    }
  }

  LLDB_LOGV(log, "Done servicing CUDA API events ({0} handled)",
            events_handled);

  // Nothing to acknowledge, because the GPU process is stopped or a prior
  // drain consumed the events. Not fatal.
  if (events_handled == 0) {
    LLDB_LOG(log, "no events handled while the GPU process is {0}",
             StateToString(gpu->GetState()));
    return events_handled;
  }

  CUDBGResult res = cuda_api->acknowledgeSyncEvents();
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to acknowledge CUDA Debugger API events. {0}",
        cudbgGetErrorString(res));
  return events_handled;
}

void LLDBServerPluginNVGPU::ReportDebuggerAPIFault(std::string description) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "{0}", description);

  // The GPU process has exited, or lldb has detached from it.
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu)
    return;

  // Only valid while the GPU is running: the client is then waiting for a stop
  // reply and will not mistake this for a query response.
  if (gpu->GetState() == lldb::eStateRunning) {
    m_gdb_server->SendStructuredDataPacket(
        llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                             {"subtype", "error"},
                                             {"message", description}}));
  }

  // Stop with an exception-class reason so the client keeps the GPU stopped and
  // shows the error. Refreshing device state would go through the poisoned API.
  gpu->OnDebuggerAPIFault(std::move(description));
  if (CPUStopsWithGPU())
    HaltNativeProcessIfNeeded();
}

void LLDBServerPluginNVGPU::ReleaseDebuggerAPI() {
  Log *log = GetLog(GDBRLog::Plugin);

  // The GPU process's copy of the table is cleared so it cannot dangle. The
  // driver's event callback uses the notifier, so the notifier outlives the
  // session that finalizing shuts down. It still goes before this returns to
  // the GPU MainLoop, which would otherwise dispatch into the finalized API.
  UnhookGPUProcess();
  if (m_cuda_api) {
    LLDB_LOG(log, "finalizing the debugger API");
    m_cuda_api.reset();
  }
  m_main_loop_event_notifier_up.reset();
}

void LLDBServerPluginNVGPU::ReleaseUnservedDebuggerAPI() {
  if (!m_cuda_api)
    return;
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "letting the debugger API go without finalizing it");
  // The driver stays up, and must not reach this object with later events.
  // It refuses a null data pointer. A callback already under way still uses
  // the notifier, which is kept until this object is destroyed.
  static char s_unused;
  CUDBGResult res =
      (*m_cuda_api)->setNotifyNewEventCallback31([](void *) {}, &s_unused);
  if (res != CUDBG_SUCCESS)
    LLDB_LOG(log, "setNotifyNewEventCallback31 failed: {0}",
             cudbgGetErrorString(res));
  UnhookGPUProcess();
  m_cuda_api->Abandon();
  m_cuda_api.reset();
}

void LLDBServerPluginNVGPU::UnhookGPUProcess() {
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu)
    return;
  gpu->SetSyncEventDrainNotifier(nullptr);
  gpu->SetDebuggerAPIFaultHandler(nullptr);
  gpu->ClearDebuggerAPI();
}

void LLDBServerPluginNVGPU::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  m_native_process_exited = true;
  // The GPU process, and the GPU server that reports its exit, belong to the
  // GPU MainLoop thread. The loop outlives this object, and runs everything
  // queued before it stops.
  m_main_loop.AddPendingCallback([this, exit_status](MainLoopBase &) {
    if (ProcessNVGPU *gpu = GetGPUProcess())
      gpu->OnNativeProcessExit(exit_status);
  });
}
