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
#include "lldb/Host/Debug.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <csignal>
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
/// RAII helper that sets the native inferior to pass (without stopping) a broad
/// set of signals for the duration of a debugger-driven resume window, then
/// restores the prior disposition on scope exit.
///
/// During the detach resume window the application is briefly continued so the
/// driver can finish its cleanup. We do not want that resume to be derailed by
/// an ordinary signal stopping the inferior, so -- mirroring cuda-gdb's
/// cuda_gdb_bypass_signals (cuda-utils.c) -- we let most signals pass straight
/// through. A few must NOT be bypassed: SIGTRAP (breakpoint/step control),
/// SIGKILL and SIGSTOP (cannot be caught/ignored), SIGCHLD (process bookkeeping)
/// and SIGURG (the CUDA debugger's notification/stop signal, GDB_SIGNAL_URG in
/// cuda-gdb).
class ScopedInferiorSignalBypass {
public:
  explicit ScopedInferiorSignalBypass(NativeProcessProtocol *process)
      : m_process(process) {
    if (!m_process)
      return;
    m_saved = m_process->GetIgnoredSignals();

    llvm::SmallVector<int, 64> bypass;
#ifdef SIGRTMAX
    const int max_signal = SIGRTMAX;
#else
    const int max_signal = 64;
#endif
    for (int signo = 1; signo <= max_signal; ++signo) {
      if (signo == SIGTRAP || signo == SIGKILL || signo == SIGSTOP ||
          signo == SIGCHLD || signo == SIGURG)
        continue;
      bypass.push_back(signo);
    }
    m_process->IgnoreSignals(bypass);
  }

  ~ScopedInferiorSignalBypass() {
    if (!m_process)
      return;
    llvm::SmallVector<int, 64> restore(m_saved.begin(), m_saved.end());
    m_process->IgnoreSignals(restore);
  }

  ScopedInferiorSignalBypass(const ScopedInferiorSignalBypass &) = delete;
  ScopedInferiorSignalBypass &
  operator=(const ScopedInferiorSignalBypass &) = delete;

private:
  NativeProcessProtocol *m_process;
  llvm::DenseSet<int> m_saved;
};
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
  m_gpu = static_cast<ProcessNVGPU *>(m_gdb_server->GetCurrentProcess());

  // The GPU process is fake and shouldn't fail to launch. Let's abort if we see
  // an error.
  if (error.Fail())
    logAndReportFatalError("Failed to launch the GPU process. {}", error);

  // Wire the back-pointer so ProcessNVGPU::Detach can delegate the detach
  // cleanup sequence to this plugin.
  if (m_gpu)
    m_gpu->SetPlugin(this);
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
             GetAttachProbeTimeoutSeconds());
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

unsigned LLDBServerPluginNVGPU::GetAttachProbeTimeoutSeconds() const {
  // Allow the probing-phase deadline to be overridden at runtime (e.g. for slow
  // drivers or tests) without recompiling. A malformed or zero value falls back
  // to the compiled default.
  if (std::optional<std::string> override_value =
          sys::Process::GetEnv("NVGPU_ATTACH_PROBE_TIMEOUT_SECONDS")) {
    unsigned parsed = 0;
    if (!llvm::StringRef(*override_value).getAsInteger(10, parsed) && parsed > 0)
      return parsed;
  }
  return kAttachProbeTimeoutSeconds;
}

void LLDBServerPluginNVGPU::ScheduleAttachProbe() {
  // Re-probe on a cadence (gap 9), independent of native stops, so late attach
  // still makes progress if the inferior produces no further stops. This runs
  // on the CPU MainLoop -- the loop that is actually running while eProbing (the
  // GPU MainLoop is not started until the reverse connection is made at the
  // report-finished breakpoint). It mirrors the one-shot timer pattern of
  // ScheduleInjectedPhaseWatchdog and re-arms itself until the state leaves
  // eProbing.
  m_native_process.GetMainLoop().AddCallback(
      [this](MainLoopBase &) {
        {
          std::lock_guard<std::mutex> guard(m_attach_mutex);
          if (m_attach_state != AttachState::eProbing)
            return; // stop re-arming once probing is over
        }
        // TryInitiateAttachServerSide re-reads the state under the lock and is a
        // no-op once it has moved on; it runs serially with native stops on this
        // same CPU MainLoop thread.
        TryInitiateAttachServerSide();
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        if (m_attach_state == AttachState::eProbing)
          ScheduleAttachProbe();
      },
      std::chrono::seconds(kAttachProbeIntervalSeconds));
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
          std::chrono::seconds(GetAttachProbeTimeoutSeconds());
    }
    // Drive re-probing on a cadence as well as on native stops, so attach makes
    // progress even if the inferior produces no further stops (gap 9).
    ScheduleAttachProbe();
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
  LLDB_LOGV(log, "LLDBServerPluginNVGPU::OnDebuggerAPIEvent");
  // The async notifier path simply drives one drain pass. The synchronous
  // detach loop calls DrainSyncEventsOnce directly.
  DrainSyncEventsOnce();
}

int LLDBServerPluginNVGPU::DrainSyncEventsOnce() {
  Log *log = GetLog(GDBRLog::Plugin);
  CUDADebuggerAPI &cuda_api = *m_cuda_api;

  // Drain the entire sync event queue before acknowledging. A single
  // notification can correspond to several queued events, so dequeuing only one
  // could drop later events (e.g. CUDBG_EVENT_ATTACH_COMPLETE) or, on the next
  // notification, find the queue already drained and hit the fatal NO_EVENT
  // path. We loop until the queue reports empty (CUDBG_ERROR_NO_EVENT_AVAILABLE)
  // or yields the CUDBG_EVENT_INVALID sentinel, then acknowledge once.
  int events_handled = 0;
  while (true) {
    // If a prior event poisoned the API (CUDBG_EVENT_INTERNAL_ERROR), stop
    // touching it: neither getNextEvent nor acknowledgeSyncEvents below run on
    // a faulted API. We have already forced a clean stop and reported the
    // error to the client.
    {
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      if (m_api_faulted)
        return events_handled;
    }
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
      return events_handled;
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
      // Poison the API, force a clean stop and surface a structured error to
      // the client. The next loop iteration sees m_api_faulted and returns
      // without acking the poisoned API.
      HandleInternalError(event.cases.internalError.errorType);
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
      // Latch completion so the synchronous detach drain loop can stop, and
      // return the state to eNone (a deliberate detach finished).
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      m_detach_complete = true;
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
    LLDB_LOG(log, "DrainSyncEventsOnce: notification with no pending events");
    return events_handled;
  }

  // Do not acknowledge on a poisoned API (an internal error consumed during
  // this pass); the forced stop and structured error have already been sent.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_api_faulted)
      return events_handled;
  }

  // Handled all pending events. Acknowledge them.
  CUDBGResult res = cuda_api->acknowledgeSyncEvents();
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("Failed to acknowledge CUDA Debugger API events. {}",
                           cudbgGetErrorString(res));
  }
  return events_handled;
}

void LLDBServerPluginNVGPU::HandleInternalError(CUDBGResult error_type) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::string description =
      llvm::formatv("CUDA debugger API internal error: {0}",
                    cudbgGetErrorString(error_type))
          .str();
  LLDB_LOG(log, "LLDBServerPluginNVGPU::HandleInternalError(). {0}",
           description);

  // Poison the API so the drain loop and any API callers (e.g. the detach
  // path) stop calling into it. Read/commit under the lock only.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_api_faulted = true;
  }

  // Surface a structured error to the client. This must be sent while the GPU
  // is still running (the client is then waiting for the stop reply and will
  // not mistake it for a query response); otherwise just log it.
  if (m_gpu->GetState() == lldb::eStateRunning) {
    m_gdb_server->SendStructuredDataPacket(
        llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                             {"subtype", "error"},
                                             {"message", description}}));
  }

  // Force a clean GPU stop with an exception-class stop reason so the client
  // keeps the GPU stopped and shows the error, preserving the session rather
  // than aborting lldb-server.
  auto log_to_client_callback = [](llvm::StringRef message) {};
  m_gpu->SuspendAllDevicesAndRefresh(log_to_client_callback, description);
  if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1") {
    bool was_halted = false;
    HaltNativeProcessIfNeeded(was_halted);
  }
}

llvm::Error LLDBServerPluginNVGPU::DetachCleanup() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU::DetachCleanup()");

  // 1. Tear down device breakpoints first, while the debugger API and devices
  // are still valid. The debug API otherwise leaves them set on the device.
  if (m_gpu)
    m_gpu->TeardownDeviceBreakpoints();

  CUDBGAPI api = m_gpu ? m_gpu->GetDebuggerAPI() : nullptr;

  // If the API was poisoned by an internal error, do not call into it again
  // (acking/resuming on a faulted API can wedge or crash the session). Just
  // mark the state and let the caller finish the teardown.
  bool faulted = false;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    faulted = m_api_faulted;
    m_attach_state = AttachState::eDetaching;
    m_detach_complete = false;
  }

  NativeProcessProtocol *cpu = m_native_process.GetCurrentProcess();

  // Resolve the detach handshake symbols server-side (no gdb-remote round-trip),
  // exactly like the attach path. Used to read the resume flag and to reset the
  // driver globals below.
  std::optional<llvm::StringMap<uint64_t>> symbols;
  if (cpu) {
    Expected<llvm::StringMap<uint64_t>> syms =
        CUDADebuggerAPI::ResolveInferiorDetachSymbols(*cpu);
    if (syms)
      symbols = std::move(*syms);
    else
      LLDB_LOG(log, "DetachCleanup: could not resolve detach symbols: {0}",
               llvm::toString(syms.takeError()));
  }
  auto get_addr = [&symbols](StringRef name) -> std::optional<uint64_t> {
    if (!symbols)
      return std::nullopt;
    llvm::StringMap<uint64_t>::const_iterator it = symbols->find(name);
    if (it == symbols->end())
      return std::nullopt;
    return it->second;
  };

  // 2. Read CUDBG_RESUME_FOR_ATTACH_DETACH: whether the driver needs the
  // application resumed so it can complete its cleanup.
  bool resume_for_detach = false;
  if (cpu && symbols) {
    if (Expected<bool> resume =
            CUDADebuggerAPI::ShouldResumeForAttachDetach(get_addr, *cpu))
      resume_for_detach = *resume;
    else
      llvm::consumeError(resume.takeError());
  }
  LLDB_LOG(log, "DetachCleanup: resume_for_detach={0}, api_faulted={1}",
           resume_for_detach, faulted);

  // 3. Reset the driver handshake flags (gap 7) so a later re-attach is clean.
  // Best-effort: log but do not abort detach on a write failure.
  if (cpu && symbols) {
    if (Error err = CUDADebuggerAPI::ResetDetachSymbols(get_addr, *cpu)) {
      LLDB_LOG(log, "DetachCleanup: failed to reset detach symbols: {0}",
               llvm::toString(std::move(err)));
    }
  }

  // 4. If the driver wants the app resumed and the API is healthy, request
  // cleanup and resume both the devices and the CPU under a signal-bypass
  // window, then drain detach events inline until the driver reports completion.
  // Otherwise mark completion directly.
  if (resume_for_detach && api && !faulted) {
    // The signal bypass keeps an ordinary signal from derailing the resume
    // while the driver finishes its cleanup (gap 5).
    ScopedInferiorSignalBypass bypass(cpu);

    CUDBGResult res = api->requestCleanupOnDetach(/*appResumeFlag=*/1);
    if (res != CUDBG_SUCCESS)
      LLDB_LOG(log, "DetachCleanup: requestCleanupOnDetach failed: {0}",
               cudbgGetErrorString(res));

    if (m_gpu) {
      for (DeviceState &device : m_gpu->GetAllDevices().GetDevices()) {
        CUDBGResult dres = api->resumeDevice(device.GetDeviceId());
        if (dres != CUDBG_SUCCESS)
          LLDB_LOG(log, "DetachCleanup: resumeDevice {0} failed: {1}",
                   device.GetDeviceId(), cudbgGetErrorString(dres));
      }
    }

    if (cpu) {
      ResumeActionList resume_actions(lldb::eStateRunning,
                                      LLDB_INVALID_SIGNAL_NUMBER);
      Status status = cpu->Resume(resume_actions);
      if (status.Fail())
        LLDB_LOG(log, "DetachCleanup: failed to resume the CPU process: {0}",
                 status.AsCString());
    }

    // Inline drain loop. We are on the GPU MainLoop thread (the same thread the
    // async notifier would dispatch on), so we must drain events ourselves
    // rather than waiting for the notifier to deliver DETACH_COMPLETE.
    using std::chrono::milliseconds;
    int iterations = 0;
    for (; iterations < kDetachMaxIterations; ++iterations) {
      {
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        if (m_detach_complete || m_api_faulted)
          break;
      }
      if (cpu && cpu->GetState() == lldb::eStateExited) {
        LLDB_LOG(log, "DetachCleanup: CPU process exited during detach");
        break;
      }
      int handled = DrainSyncEventsOnce();
      {
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        if (m_detach_complete || m_api_faulted)
          break;
      }
      if (handled == 0)
        std::this_thread::sleep_for(milliseconds(1));
    }

    bool completed = false;
    {
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      completed = m_detach_complete;
    }
    if (!completed)
      LLDB_LOG(log,
               "DetachCleanup: detach cleanup did not complete after {0} "
               "iterations; proceeding with teardown anyway.",
               kDetachMaxIterations);
  } else {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_detach_complete = true;
    m_attach_state = AttachState::eNone;
  }

  // 5. Clear the attach-specific driver state (unless the API is poisoned).
  if (api && !faulted) {
    CUDBGResult res = api->clearAttachState();
    if (res != CUDBG_SUCCESS)
      LLDB_LOG(log, "DetachCleanup: clearAttachState failed: {0}",
               cudbgGetErrorString(res));
  }

  // Return to a clean idle state. The caller (ProcessNVGPU::Detach) performs the
  // final SetState(eStateDetached).
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_attach_state = AttachState::eNone;
  }

  return Error::success();
}

void LLDBServerPluginNVGPU::CancelInProgressAttach() {
  Log *log = GetLog(GDBRLog::Plugin);

  // Decide what to do based on how far the attach has progressed. Reads and
  // commits happen under the lock; the heavier DetachCleanup path runs without
  // the lock held. Idempotent against OnAttachComplete: once the state has
  // advanced to eComplete (or back to eNone), there is nothing to cancel.
  enum class Action { eNothing, eResetProbing, eRouteThroughDetach } action =
      Action::eNothing;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    switch (m_attach_state) {
    case AttachState::eProbing:
      // Not yet injected: just stop probing.
      m_attach_state = AttachState::eNone;
      action = Action::eResetProbing;
      break;
    case AttachState::eInjected:
      // Injected. If the API is up we must route through the full detach
      // cleanup; otherwise just reset (nothing to clean up in the driver yet).
      if (m_api_initialized)
        action = Action::eRouteThroughDetach;
      else {
        m_attach_state = AttachState::eNone;
        action = Action::eResetProbing;
      }
      break;
    case AttachState::eNone:
    case AttachState::eComplete:
    case AttachState::eDetaching:
      action = Action::eNothing;
      break;
    }
  }

  switch (action) {
  case Action::eNothing:
    LLDB_LOG(log, "CancelInProgressAttach: nothing to cancel");
    break;
  case Action::eResetProbing:
    LLDB_LOG(log, "CancelInProgressAttach: cancelled before API came up");
    break;
  case Action::eRouteThroughDetach:
    LLDB_LOG(log, "CancelInProgressAttach: routing through detach cleanup");
    if (Error err = DetachCleanup())
      LLDB_LOG(log, "CancelInProgressAttach: detach cleanup reported: {0}",
               llvm::toString(std::move(err)));
    break;
  }
}

void LLDBServerPluginNVGPU::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  if (m_gpu)
    m_gpu->OnNativeProcessExit(exit_status);
}
