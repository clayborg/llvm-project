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
  m_gpu = static_cast<ProcessNVGPU *>(m_gdb_server->GetCurrentProcess());

  // The GPU process is fake and shouldn't fail to launch. Let's abort if we see
  // an error.
  if (error.Fail())
    logAndReportFatalError("Failed to launch the GPU process. {}", error);
}

llvm::StringRef LLDBServerPluginNVGPU::GetPluginName() { return "nvgpu"; }

std::optional<GPUActions> LLDBServerPluginNVGPU::NativeProcessIsStopping() {
  // While attaching, every native stop is an opportunity to initiate the safe
  // attach. We do the whole handshake server-side here (resolve the driver
  // symbols from the inferior and write the magic byte) rather than asking the
  // client to resolve symbols: issuing a gdb-remote round-trip during attach
  // stop processing corrupts the in-flight continue cycle on the client. Doing
  // it server-side touches only the inferior (via ptrace) and a procfs FD, so
  // it never perturbs the gdb-remote packet stream. We therefore have no
  // GPUActions to return to the client here.
  TryInitiateAttachServerSide();
  return std::nullopt;
}

void LLDBServerPluginNVGPU::TryInitiateAttachServerSide() {
  Log *log = GetLog(GDBRLog::Plugin);
  using std::chrono::steady_clock;

  // Snapshot the state under the lock, then perform all blocking work
  // (procfs/ptrace reads and the FD write) WITHOUT holding it. Holding
  // m_attach_mutex across blocking syscalls could stall the GPU event thread,
  // which also takes the lock (see OnDebuggerAPIEvent / OnAttachComplete).
  //
  // No "in progress" claim state is needed. All native callbacks
  // (NativeProcessIsStopping / BreakpointWasHit / GetInitializeActions) run
  // serially on the single LLGS MainLoop thread, so a second native stop cannot
  // run this concurrently, and the unlocked work below cannot be re-entered
  // while this thread is blocked in it. The only other writers of
  // m_attach_state run on the GPU MainLoop thread, but that thread does not yet
  // exist while we are eProbing -- it is created later, in
  // InitializeAPIAndConnect at the report-finished breakpoint, which also moves
  // the state out of eProbing. So while eProbing nothing else can touch the
  // state, and we simply read it, do the unlocked work, then commit eInjected.
  NativeProcessProtocol *cpu_process = nullptr;
  steady_clock::time_point deadline;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    deadline = m_attach_deadline;
    cpu_process = m_native_process.GetCurrentProcess();
  }

  if (!cpu_process) {
    LLDB_LOG(log, "TryInitiateAttachServerSide: no current native process yet");
    // Retryable: stay in eProbing so a later stop tries again.
    return;
  }

  // Bounded retry: if we cannot initiate the safe attach within the timeout,
  // stop probing and fall back to the launch-style initialization breakpoints
  // rather than retrying on every stop forever.
  if (steady_clock::now() > deadline) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: timed out after {0}s trying to "
             "initiate the safe attach; falling back to launch-style "
             "initialization.",
             kAttachProbeTimeoutSeconds);
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state == AttachState::eProbing)
      m_attach_state = AttachState::eNone;
    return;
  }

  // Resolve the handshake symbols directly from the inferior's libcuda image.
  // If libcuda is not resolvable yet (not mapped, or required symbols missing),
  // stay in the probing state so a later native stop retries until the deadline
  // above.
  Expected<llvm::StringMap<uint64_t>> symbols =
      CUDADebuggerAPI::ResolveInferiorAttachSymbols(*cpu_process);
  if (!symbols) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: could not resolve libcuda symbols "
             "yet: {0}",
             llvm::toString(symbols.takeError()));
    // Retryable: stay in eProbing.
    return;
  }

  auto get_addr = [&symbols](StringRef name) -> std::optional<uint64_t> {
    llvm::StringMap<uint64_t>::const_iterator it = symbols->find(name);
    if (it == symbols->end())
      return std::nullopt;
    return it->second;
  };

  // Determine whether the running process exposes a usable safe-attach handler.
  // Because ResolveInferiorAttachSymbols above guarantees every required symbol
  // resolved, an error here is a genuine read failure (not an unresolved
  // symbol), and a false result means the handler flag is present but not set.
  Expected<bool> supported =
      CUDADebuggerAPI::IsLateAttachSupported(get_addr, *cpu_process);
  if (!supported) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: attach handler not available yet: "
             "{0}",
             llvm::toString(supported.takeError()));
    // Retryable: stay in eProbing.
    return;
  }
  if (!*supported) {
    // CUDA is present but the driver does not advertise the safe-attach
    // handler. There is nothing more we can do via late attach; fall back to
    // the launch-style breakpoints (which still cover the not-yet-initialized
    // case) and stop probing.
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: safe attach handler unavailable; "
             "falling back to launch-style initialization");
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state == AttachState::eProbing)
      m_attach_state = AttachState::eNone;
    return;
  }

  // Whether the driver wants the application to keep running to complete the
  // attach. In our model the CPU is resumed by the user's "continue" after
  // attach, after which the driver services the request and calls
  // CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED; the rest of the handshake is then
  // driven asynchronously off the GPU MainLoop by the debugger-API event
  // callback (see OnDebuggerAPIEvent / CUDBG_EVENT_ATTACH_COMPLETE). We read the
  // flag for diagnostics only.
  bool resume_for_attach_detach = false;
  if (Expected<bool> resume =
          CUDADebuggerAPI::ShouldResumeForAttachDetach(get_addr, *cpu_process))
    resume_for_attach_detach = *resume;
  else
    llvm::consumeError(resume.takeError());

  if (Error err = CUDADebuggerAPI::InitiateSafeAttach(get_addr, *cpu_process)) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: failed to initiate safe attach: {0}",
             llvm::toString(std::move(err)));
    // Retryable: stay in eProbing.
    return;
  }

  // Commit the transition under the lock. The state is still eProbing -- only
  // this MainLoop thread can move it out of eProbing, and the GPU thread that
  // could otherwise change it does not exist yet (see comment above).
  bool committed = false;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    m_attach_state = AttachState::eInjected;
    m_attach_deadline =
        steady_clock::now() + std::chrono::seconds(kAttachInjectTimeoutSeconds);
    committed = true;
  }
  if (committed)
    ScheduleInjectedPhaseWatchdog();
  LLDB_LOG(log,
           "TryInitiateAttachServerSide: safe attach initiated "
           "(resume_for_attach={0}). Waiting for "
           "CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED.",
           resume_for_attach_detach);
}

void LLDBServerPluginNVGPU::CheckInjectedPhaseTimeoutLocked() {
  Log *log = GetLog(GDBRLog::Plugin);
  if (m_attach_state != AttachState::eInjected)
    return;
  if (std::chrono::steady_clock::now() > m_attach_deadline) {
    LLDB_LOG(log,
             "NVGPU late attach: safe attach did not complete within {0}s "
             "after injection; the driver may not have serviced the attach "
             "request. The CPU process must keep running for the driver to "
             "finish injecting the debug engine.",
             kAttachInjectTimeoutSeconds);
  }
}

void LLDBServerPluginNVGPU::ScheduleInjectedPhaseWatchdog() {
  // Drive the injected-phase deadline from a MainLoop timer -- the single
  // surface for this diagnostics-only deadline. The GPU main loop runs once the
  // reverse connection is established (at
  // CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED), so this fires precisely while
  // waiting for CUDBG_EVENT_ATTACH_COMPLETE. AddCallback is one-shot, so the
  // wedge warning is logged at most once without needing a dedup latch.
  m_main_loop.AddCallback(
      [this](MainLoopBase &) {
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        CheckInjectedPhaseTimeoutLocked();
      },
      std::chrono::seconds(kAttachInjectTimeoutSeconds));
}

void LLDBServerPluginNVGPU::OnAttachComplete() {
  Log *log = GetLog(GDBRLog::Plugin);
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state == AttachState::eComplete)
      return;
    m_attach_state = AttachState::eComplete;
  }
  LLDB_LOG(log, "LLDBServerPluginNVGPU::OnAttachComplete(). Refreshing device "
                "state so CUDA threads are enumerated.");
  auto log_to_client_callback = [](llvm::StringRef message) {};
  m_gpu->SuspendAllDevicesAndRefresh(log_to_client_callback,
                                     "attached to running CUDA application");
  if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1") {
    bool was_halted = false;
    HaltNativeProcessIfNeeded(was_halted);
  }
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
    connection_info.triple =
        ProcessNVGPU::GetNVPTXArchitecture().GetTriple().str();
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

Expected<GPUActions> LLDBServerPluginNVGPU::InitializeAPIAndConnect(
    SymbolAddressProvider get_symbol_address, StringRef libcuda_library_name) {
  Expected<CUDADebuggerAPI> api_or = CUDADebuggerAPI::Initialize(
      get_symbol_address, libcuda_library_name,
      *m_native_process.GetCurrentProcess());
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

  // deferred sync events. The driver only fires the new-event notification for
  // newly enqueued events, so any events queued behind the deferred
  // CUDBG_EVENT_ELF_IMAGE_LOADED (e.g. CUDBG_EVENT_ATTACH_COMPLETE) would never
  // be drained on their own. Give the GPU process a way to re-run the
  // event-processing loop so those already-queued events are handled.
  m_gpu->SetSyncEventDrainNotifier(
      [this]() { m_main_loop_event_notifier_up->FireEvent(); });

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

  Expected<GPUPluginConnectionInfo> connection_info = CreateConnection();
  if (!connection_info)
    return connection_info.takeError();

  GPUActions actions = GetNewGPUAction();
  actions.connect_info = std::move(*connection_info);
  return actions;
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  std::string library_name = *args.breakpoint.name_info->shlib;
  // This method is invoked when a CPU breakpoint set by this plugin is hit.
  //
  //  - For launch, it is the cuInit-style breakpoint signaling that the driver
  //    is initializing. We assume no kernels run until we resume the CPU.
  //  - For late attach, it is CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED, signaling
  //    that the driver has safely injected the debug engine.
  //
  // In both cases the debug engine is now present, so this is the right time to
  // initialize the debugger API. The symbol addresses needed for the handshake
  // are delivered in the breakpoint hit args.
  // Guard against initializing a second debugger API. The initialization
  // breakpoint can be hit more than once (e.g. both libcuda sonames resolve, or
  // it fires again after late attach already brought the API up). Re-running
  // InitializeAPIAndConnect would create a second API table and reverse
  // connection, finalizing/leaking the live one, so just disable the breakpoint
  // and return.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    // No "initializing in progress" claim is needed to close the
    // check-then-act window: BreakpointWasHit runs only on the single LLGS
    // MainLoop thread (via Handle_jGPUPluginBreakpointHit), so two breakpoint
    // callbacks cannot run concurrently, and while this thread is blocked in
    // the unlocked InitializeAPIAndConnect below it cannot be re-entered. A
    // simple committed flag is therefore sufficient.
    if (m_api_initialized) {
      Log *log = GetLog(GDBRLog::Plugin);
      LLDB_LOG(log,
               "LLDBServerPluginNVGPU::BreakpointWasHit: debugger API already "
               "initialized; ignoring breakpoint and disabling it.");
      GPUPluginBreakpointHitResponse response(GetNewGPUAction());
      response.disable_bp = true;
      return response;
    }
  }

  auto get_addr = [&args](StringRef name) -> std::optional<uint64_t> {
    return args.GetSymbolValue(name);
  };
  Expected<GPUActions> actions =
      InitializeAPIAndConnect(get_addr, library_name);
  if (!actions) {
    // Leave m_api_initialized false so a later breakpoint can retry the init.
    return actions.takeError();
  }

  // If the API was initialized while still probing (CUDA came up via the
  // launch-style breakpoint during attach), leave the probing state so we stop
  // retrying the attach handshake on subsequent stops. The normal late attach
  // path is already in eInjected here.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_api_initialized = true;
    if (m_attach_state == AttachState::eProbing)
      m_attach_state = AttachState::eInjected;
  }

  GPUPluginBreakpointHitResponse response(std::move(*actions));
  response.disable_bp = true;
  return response;
}

GPUActions LLDBServerPluginNVGPU::GetInitializeActions(
    const GPUPluginInitializeArgs &args) {
  GPUActions init_actions = GetNewGPUAction();

  // The cuInit-style initialization breakpoints handle the launch case, and
  // also the case where we attach before CUDA has been initialized.
  init_actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetInitializationBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME));
  init_actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetInitializationBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME_ALT));

  if (args.is_attach) {
    Log *log = GetLog(GDBRLog::Plugin);
    LLDB_LOG(log, "LLDBServerPluginNVGPU: preparing for late attach");
    {
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      m_attach_state = AttachState::eProbing;
      m_attach_deadline =
          std::chrono::steady_clock::now() +
          std::chrono::seconds(kAttachProbeTimeoutSeconds);
    }
    // When attaching to an already-running CUDA process, cuInit has already
    // been called, so the launch breakpoints above will not fire. Instead we
    // initiate the safe attach procedure and wait for the driver to call
    // CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED. Set that breakpoint here (it
    // resolves lazily once libcuda is loaded after the attach completes).
    init_actions.breakpoints.emplace_back(
        CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(
            CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME));
    init_actions.breakpoints.emplace_back(
        CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(
            CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME_ALT));
  }
  return init_actions;
}

void LLDBServerPluginNVGPU::OnDebuggerAPIEvent() {
  Log *log = GetLog(GDBRLog::Plugin);
  CUDADebuggerAPI &cuda_api = *m_cuda_api;
  LLDB_LOGV(log, "LLDBServerPluginNVGPU::OnDebuggerAPIEvent");

  // Drain the entire sync event queue before acknowledging. A single
  // notification can correspond to several queued events, so dequeuing only one
  // could drop later events (e.g. CUDBG_EVENT_ATTACH_COMPLETE) or, on the next
  // notification, find the queue already drained and hit the fatal NO_EVENT
  // path. We loop until the queue reports empty (CUDBG_ERROR_NO_EVENT_AVAILABLE)
  // or yields the CUDBG_EVENT_INVALID sentinel, then acknowledge once.
  size_t events_handled = 0;
  while (true) {
    CUDBGEvent event;
    CUDBGResult res = cuda_api->getNextEvent(
        CUDBGEventQueueType::CUDBG_EVENT_QUEUE_TYPE_SYNC, &event);
    if (res == CUDBGResult::CUDBG_ERROR_NO_EVENT_AVAILABLE)
      break; // queue drained
    if (res != CUDBG_SUCCESS)
      logAndReportFatalError(
          "Failed to get the next CUDA Debugger API event. {}",
          cudbgGetErrorString(res));
    if (event.kind == CUDBG_EVENT_INVALID) {
      // The API also signals "no more events" with this sentinel kind.
      LLDB_LOG(log, "CUDBG_EVENT_INVALID");
      break;
    }

    ++events_handled;

    switch (event.kind) {
    case CUDBG_EVENT_ELF_IMAGE_LOADED: {
      LLDB_LOG(log, "CUDBG_EVENT_ELF_IMAGE_LOADED");
      // When we get an elf file, we report a dyld stop to the client. We hold
      // ack'ing the events until we have gotten the autoresume from the client
      // (Resume acknowledges them), so return without draining the rest; the
      // next notification after resume will pick up where we left off. This
      // will need to be changed once we support multiple contexts.
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
      OnAttachComplete();
      break;
    }
    case CUDBG_EVENT_DETACH_COMPLETE: {
      LLDB_LOG(log, "CUDBG_EVENT_DETACH_COMPLETE");
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      m_attach_state = AttachState::eNone;
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
        // The structured data packet can only be sent when the client is
        // waiting for the stop reply packet. Otherwise, it might think that
        // this is the response to a pending query packet. Creating the callback
        // at this point is safe because we are about to report that the state
        // is stopped, which means that we are running.
        if (m_gpu->GetState() != lldb::eStateRunning) {
          logAndReportFatalError(
              "Logging to client is only supported when the GPU is running.");
        }

        m_gdb_server->SendStructuredDataPacket(
            llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                                 {"subtype", "log"},
                                                 {"message", message}}));
      };
      m_gpu->OnAllDevicesSuspended(event.cases.allDevicesSuspended,
                                   log_to_client_callback);
      // Here we can force the two processes to be in sync. Not synchronizing
      // them allows for a non-stop mode for the native process.
      if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1") {
        bool was_halted = false;
        HaltNativeProcessIfNeeded(was_halted);
      }
      break;
    }
    case CUDBG_EVENT_INVALID: {
      // Handled above as the loop terminator; unreachable here.
      break;
    }
    default:
      LLDB_LOG(log, "Unknown event kind: {}", event.kind);
      break;
    }
  }

  LLDB_LOGV(log, "Done servicing CUDA API events ({0} handled)",
            events_handled);

  // A notification with no events to service is not fatal (it can happen e.g.
  // if a prior drain already consumed them); just skip the acknowledgement.
  if (events_handled == 0) {
    LLDB_LOG(log, "OnDebuggerAPIEvent: notification with no pending events");
    return;
  }

  // Handled all pending events. Acknowledge them.
  CUDBGResult res = cuda_api->acknowledgeSyncEvents();
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
