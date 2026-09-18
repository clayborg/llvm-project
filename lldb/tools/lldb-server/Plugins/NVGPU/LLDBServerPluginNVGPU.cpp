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
#include <future>
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
/// Lets ordinary signals pass through the inferior without stopping it, for
/// the duration of a debugger-driven resume, restoring the prior disposition
/// on scope exit. Mirrors cuda-gdb's cuda_gdb_bypass_signals.
///
/// SIGTRAP (breakpoint and step control), SIGKILL and SIGSTOP (cannot be
/// ignored), SIGCHLD (process bookkeeping) and SIGURG (the debug API's
/// notification signal) are excluded.
class ScopedInferiorSignalBypass {
public:
  explicit ScopedInferiorSignalBypass(NativeProcessProtocol *process)
      : m_process(process) {
    if (!m_process)
      return;
    m_saved = m_process->GetIgnoredSignals();

#ifdef SIGRTMAX
    const int max_signal = SIGRTMAX;
#else
    const int max_signal = 64;
#endif
    llvm::SmallVector<int, 64> bypass;
    for (int signo = 1; signo <= max_signal; ++signo) {
      if (signo != SIGTRAP && signo != SIGKILL && signo != SIGSTOP &&
          signo != SIGCHLD && signo != SIGURG)
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

const char *EventKindName(CUDBGEventKind kind) {
  switch (kind) {
  case CUDBG_EVENT_INVALID:
    return "CUDBG_EVENT_INVALID";
  case CUDBG_EVENT_ELF_IMAGE_LOADED:
    return "CUDBG_EVENT_ELF_IMAGE_LOADED";
  case CUDBG_EVENT_KERNEL_READY:
    return "CUDBG_EVENT_KERNEL_READY";
  case CUDBG_EVENT_KERNEL_FINISHED:
    return "CUDBG_EVENT_KERNEL_FINISHED";
  case CUDBG_EVENT_INTERNAL_ERROR:
    return "CUDBG_EVENT_INTERNAL_ERROR";
  case CUDBG_EVENT_CTX_PUSH:
    return "CUDBG_EVENT_CTX_PUSH";
  case CUDBG_EVENT_CTX_POP:
    return "CUDBG_EVENT_CTX_POP";
  case CUDBG_EVENT_CTX_CREATE:
    return "CUDBG_EVENT_CTX_CREATE";
  case CUDBG_EVENT_CTX_DESTROY:
    return "CUDBG_EVENT_CTX_DESTROY";
  case CUDBG_EVENT_TIMEOUT:
    return "CUDBG_EVENT_TIMEOUT";
  case CUDBG_EVENT_ATTACH_COMPLETE:
    return "CUDBG_EVENT_ATTACH_COMPLETE";
  case CUDBG_EVENT_DETACH_COMPLETE:
    return "CUDBG_EVENT_DETACH_COMPLETE";
  case CUDBG_EVENT_ELF_IMAGE_UNLOADED:
    return "CUDBG_EVENT_ELF_IMAGE_UNLOADED";
  case CUDBG_EVENT_FUNCTIONS_LOADED:
    return "CUDBG_EVENT_FUNCTIONS_LOADED";
  case CUDBG_EVENT_ALL_DEVICES_SUSPENDED:
    return "CUDBG_EVENT_ALL_DEVICES_SUSPENDED";
  case CUDBG_EVENT_CUDA_LOGS_AVAILABLE:
    return "CUDBG_EVENT_CUDA_LOGS_AVAILABLE";
  case CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED:
    return "CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED";
  case CUDBG_EVENT_SINGLE_STEP_COMPLETE:
    return "CUDBG_EVENT_SINGLE_STEP_COMPLETE";
  case CUDBG_EVENT_CUDA_LOGS_RULESET_CHANGED:
    return "CUDBG_EVENT_CUDA_LOGS_RULESET_CHANGED";
  }
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
  TryInitiateAttachServerSide();
  return std::nullopt;
}

void LLDBServerPluginNVGPU::TryInitiateAttachServerSide() {
  Log *log = GetLog(GDBRLog::Plugin);
  using std::chrono::steady_clock;

  // Take the lock only to read and to commit the transition; the procfs/ptrace
  // reads and the FD write in between must not block the GPU event thread,
  // which takes the same lock.
  //
  // Nothing can race the unlocked work: the native callbacks all run serially
  // on the single LLGS MainLoop thread, and the only other writer of
  // m_attach_state lives on the GPU MainLoop thread, which does not exist until
  // InitializeAPIAndConnect creates it.
  NativeProcessProtocol *cpu_process = nullptr;
  steady_clock::time_point deadline;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    deadline = m_attach_deadline;
    cpu_process = m_native_process.GetCurrentProcess();
  }

  // Everything below either commits eInjected or leaves the state at eProbing
  // so a later stop retries.
  if (!cpu_process) {
    LLDB_LOG(log, "TryInitiateAttachServerSide: no current native process yet");
    return;
  }

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

  Expected<llvm::StringMap<uint64_t>> symbols =
      CUDADebuggerAPI::ResolveInferiorAttachSymbols(*cpu_process);
  if (!symbols) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: could not resolve libcuda symbols "
             "yet: {0}",
             llvm::toString(symbols.takeError()));
    return;
  }

  auto get_addr = [&symbols](StringRef name) -> std::optional<uint64_t> {
    llvm::StringMap<uint64_t>::const_iterator it = symbols->find(name);
    if (it == symbols->end())
      return std::nullopt;
    return it->second;
  };

  // Every required symbol resolved above, so an error here is a read failure
  // rather than a missing symbol, and false means the flag is simply not set.
  Expected<bool> supported =
      CUDADebuggerAPI::IsLateAttachSupported(get_addr, *cpu_process);
  if (!supported) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: attach handler not available yet: "
             "{0}",
             llvm::toString(supported.takeError()));
    return;
  }
  if (!*supported) {
    // CUDA is present but offers no safe-attach handler. The launch-style
    // breakpoints still cover the not-yet-initialized case, so stop probing.
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: safe attach handler unavailable; "
             "falling back to launch-style initialization");
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state == AttachState::eProbing)
      m_attach_state = AttachState::eNone;
    return;
  }

  if (Error err = CUDADebuggerAPI::InitiateSafeAttach(get_addr, *cpu_process)) {
    LLDB_LOG(log,
             "TryInitiateAttachServerSide: failed to initiate safe attach: {0}",
             llvm::toString(std::move(err)));
    return;
  }

  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    m_attach_state = AttachState::eInjected;
    m_attach_deadline =
        steady_clock::now() + std::chrono::seconds(kAttachInjectTimeoutSeconds);
  }
  ScheduleInjectedPhaseWatchdog();
  LLDB_LOG(log, "TryInitiateAttachServerSide: safe attach initiated; waiting "
                "for CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED.");
}

void LLDBServerPluginNVGPU::ScheduleAttachProbe() {
  // Runs on the native MainLoop, the only one running while eProbing: the GPU
  // MainLoop does not start until the reverse connection is made.
  m_native_process.GetMainLoop().AddCallback(
      [this](MainLoopBase &) {
        {
          std::lock_guard<std::mutex> guard(m_attach_mutex);
          if (m_attach_state != AttachState::eProbing)
            return; // stop re-arming once probing is over
        }
        TryInitiateAttachServerSide();
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        if (m_attach_state == AttachState::eProbing)
          ScheduleAttachProbe();
      },
      std::chrono::seconds(kAttachProbeIntervalSeconds));
}

void LLDBServerPluginNVGPU::ScheduleInjectedPhaseWatchdog() {
  // One-shot, so the warning cannot repeat. It runs on the GPU MainLoop, which
  // by then is exactly the loop waiting for CUDBG_EVENT_ATTACH_COMPLETE.
  m_main_loop.AddCallback(
      [this](MainLoopBase &) {
        std::lock_guard<std::mutex> guard(m_attach_mutex);
        if (m_attach_state != AttachState::eInjected ||
            std::chrono::steady_clock::now() <= m_attach_deadline)
          return;
        LLDB_LOG(GetLog(GDBRLog::Plugin),
                 "NVGPU late attach: the driver did not finish injecting the "
                 "debug engine within {0}s. The CPU process must keep running "
                 "for it to do so.",
                 kAttachInjectTimeoutSeconds);
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
    SymbolAddressProvider get_symbol_address, StringRef libcuda_library_name,
    bool is_late_attach) {
  Expected<CUDADebuggerAPI> api_or = CUDADebuggerAPI::Initialize(
      get_symbol_address, libcuda_library_name,
      *m_native_process.GetCurrentProcess(),
      is_late_attach ? CUDADebuggerAPI::InitContext::eLateAttach
                     : CUDADebuggerAPI::InitContext::eLaunch);
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

  // The driver only notifies for newly enqueued events, so anything sitting
  // behind a deferred CUDBG_EVENT_ELF_IMAGE_LOADED would never be drained on
  // its own. Let the GPU process re-run the drain to pick those up.
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

  // Only now that the event callback is registered may the driver be told a
  // client is ready, so no notification can be emitted before we can observe
  // it. The launch path published the IPC flag during Initialize.
  if (is_late_attach) {
    if (Error err = FinishLateAttachIpcHandshake(get_symbol_address))
      return err;
  }

  Expected<GPUPluginConnectionInfo> connection_info = CreateConnection();
  if (!connection_info)
    return connection_info.takeError();

  GPUActions actions = GetNewGPUAction();
  actions.connect_info = std::move(*connection_info);
  return actions;
}

llvm::Error LLDBServerPluginNVGPU::FinishLateAttachIpcHandshake(
    SymbolAddressProvider get_symbol_address) {
  Log *log = GetLog(GDBRLog::Plugin);
  NativeProcessProtocol *cpu_process = m_native_process.GetCurrentProcess();
  if (!cpu_process)
    return createStringError(
        "No native process available to complete the late attach handshake");

  // Required on both branches below: the driver emits no callbacks at all,
  // CUDBG_EVENT_ATTACH_COMPLETE included, until a client declares itself ready.
  if (Error err = CUDADebuggerAPI::SetIpcFlag(get_symbol_address, *cpu_process,
                                              /*enabled=*/true))
    return err;

  Expected<uint32_t> resume_for_attach =
      CUDADebuggerAPI::ReadResumeForAttachDetach(get_symbol_address,
                                                 *cpu_process);
  if (!resume_for_attach)
    return resume_for_attach.takeError();

  if (*resume_for_attach != 0) {
    LLDB_LOG(log,
             "FinishLateAttachIpcHandshake: driver will replay pre-existing "
             "state; waiting for CUDBG_EVENT_ATTACH_COMPLETE.");
    return Error::success();
  }

  LLDB_LOG(log,
           "FinishLateAttachIpcHandshake: no pre-existing state to replay; "
           "completing the attach directly.");

  // OnAttachComplete belongs on the GPU MainLoop, like the event-driven path.
  // That loop is started by the connection this call precedes, so queue it.
  m_main_loop.AddPendingCallback([this](MainLoopBase &) { OnAttachComplete(); });
  return Error::success();
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  std::string library_name = *args.breakpoint.name_info->shlib;
  // Either the cuInit-style breakpoint (launch) or
  // CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED (late attach). The debug engine is
  // present in both cases, and the hit args carry the handshake symbols.
  //
  // The breakpoint can fire more than once -- both libcuda sonames may resolve.
  // Re-running InitializeAPIAndConnect would build a second API table and
  // reverse connection and leak the live one, so just disable it. A plain flag
  // suffices because BreakpointWasHit only ever runs on the LLGS MainLoop
  // thread, which is also blocked inside the unlocked initialization below.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
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
  const bool is_late_attach = CUDADebuggerAPI::IsAttachFinishedBreakpoint(
      args.breakpoint.name_info->function_name);
  Expected<GPUActions> actions =
      InitializeAPIAndConnect(get_addr, library_name, is_late_attach);
  if (!actions) {
    // Leave m_api_initialized false so a later breakpoint can retry the init.
    return actions.takeError();
  }

  // Leaving eProbing stops the attach handshake retries. We are still probing
  // when CUDA came up via the launch-style breakpoint during an attach; the
  // normal late attach path is already at eInjected.
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
      m_attach_deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(kAttachProbeTimeoutSeconds);
    }
    ScheduleAttachProbe();
    // cuInit has already run in an already-running CUDA process, so the
    // breakpoints above will not fire. The safe attach procedure ends at
    // CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED instead.
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

  int events_handled = 0;
  while (true) {
    // A poisoned API must not be touched again: the clean stop and structured
    // error have already been reported.
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
    // The API also signals "no more events" with this sentinel kind.
    if (event.kind == CUDBG_EVENT_INVALID)
      break;

    ++events_handled;
    LLDB_LOG(log, "{0}", EventKindName(event.kind));

    switch (event.kind) {
    case CUDBG_EVENT_ELF_IMAGE_LOADED: {
      // Report a dyld stop to the client and stop draining: the events stay
      // unacknowledged until the client resumes (Resume acknowledges them), and
      // the next notification picks up where we left off. This will need to
      // change once we support multiple contexts.
      m_gpu->OnElfImageLoaded(event.cases.elfImageLoaded);
      m_gpu->ReportDyldStop();
      return events_handled;
    }
    case CUDBG_EVENT_INTERNAL_ERROR: {
      // The next iteration sees m_api_faulted and returns without acking.
      HandleInternalError(event.cases.internalError.errorType);
      break;
    }
    case CUDBG_EVENT_ATTACH_COMPLETE: {
      OnAttachComplete();
      break;
    }
    case CUDBG_EVENT_DETACH_COMPLETE: {
      // eNone is how the drain loop tells a finished detach from an attach.
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      m_detach_complete = true;
      m_attach_state = AttachState::eNone;
      break;
    }
    case CUDBG_EVENT_ALL_DEVICES_SUSPENDED: {
      LLDB_LOG(log, "broken devices {0:x}, faulted devices {1:x}",
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
    default:
      // Logged above; nothing else to do.
      break;
    }
  }

  LLDB_LOGV(log, "Done servicing CUDA API events ({0} handled)",
            events_handled);

  // Nothing to acknowledge. Not fatal: a prior drain may have consumed them.
  if (events_handled == 0) {
    LLDB_LOG(log, "DrainSyncEventsOnce: notification with no pending events");
    return events_handled;
  }

  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_api_faulted)
      return events_handled;
  }

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

  // Poison the API so the drain loop and the detach path stop calling into it.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_api_faulted = true;
  }

  // Only valid while the GPU is running: the client is then waiting for a stop
  // reply and will not mistake this for a query response.
  if (m_gpu->GetState() == lldb::eStateRunning) {
    m_gdb_server->SendStructuredDataPacket(
        llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                             {"subtype", "error"},
                                             {"message", description}}));
  }

  // Stop with an exception-class reason so the client keeps the GPU stopped and
  // shows the error, preserving the session rather than aborting lldb-server.
  auto log_to_client_callback = [](llvm::StringRef message) {};
  m_gpu->SuspendAllDevicesAndRefresh(log_to_client_callback, description);
  if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1") {
    bool was_halted = false;
    HaltNativeProcessIfNeeded(was_halted);
  }
}

llvm::Error
LLDBServerPluginNVGPU::RunOnNativeMainLoop(std::function<llvm::Error()> work,
                                           std::chrono::milliseconds timeout) {
  // Held by both this frame and the callback, so a timeout here cannot leave
  // the native thread writing into a destroyed promise.
  struct SharedWork {
    std::function<llvm::Error()> work;
    /// An empty string means success; otherwise the rendered error.
    std::promise<std::string> result;
  };
  std::shared_ptr<SharedWork> shared = std::make_shared<SharedWork>();
  shared->work = std::move(work);
  std::future<std::string> future = shared->result.get_future();

  if (!m_native_process.GetMainLoop().AddPendingCallback(
          [shared](MainLoopBase &) {
            std::string message;
            if (Error err = shared->work())
              message = llvm::toString(std::move(err));
            shared->result.set_value(std::move(message));
          }))
    return createStringError("the native MainLoop is no longer accepting work");

  if (future.wait_for(timeout) != std::future_status::ready)
    return createStringErrorFmt(
        "timed out after {0}ms waiting for the native MainLoop to run the "
        "request",
        timeout.count());

  std::string message = future.get();
  if (message.empty())
    return Error::success();
  return createStringError(message);
}

llvm::Error LLDBServerPluginNVGPU::DetachCleanup() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU::DetachCleanup()");

  // 1. Tear the device breakpoints down first, while the API and devices are
  // still valid. The debug API otherwise leaves them set on the device.
  if (m_gpu)
    m_gpu->TeardownDeviceBreakpoints();

  CUDBGAPI api = m_gpu ? m_gpu->GetDebuggerAPI() : nullptr;

  // A faulted API must not be called again; the steps below skip it and the
  // caller still finishes the teardown.
  bool faulted = false;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    faulted = m_api_faulted;
    m_attach_state = AttachState::eDetaching;
    m_detach_complete = false;
  }

  NativeProcessProtocol *cpu = m_native_process.GetCurrentProcess();

  // Resolved server-side like the attach path: used to read the resume flag and
  // to reset the driver globals below.
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

  // 2. Whether the driver needs the application resumed to complete its
  // cleanup. Kept raw because requestCleanupOnDetach takes it verbatim below.
  uint32_t resume_for_detach = 0;
  if (cpu && symbols) {
    if (Expected<uint32_t> resume =
            CUDADebuggerAPI::ReadResumeForAttachDetach(get_addr, *cpu))
      resume_for_detach = *resume;
    else
      llvm::consumeError(resume.takeError());
  }
  LLDB_LOG(log, "DetachCleanup: resume_for_detach={0}, api_faulted={1}",
           resume_for_detach, faulted);

  // 3. Let the driver clean up: resume the devices and the app, then drain
  // until it reports completion.
  if (resume_for_detach && api && !faulted) {
    // Keeps an ordinary signal from derailing the resume window below.
    ScopedInferiorSignalBypass bypass(cpu);

    // A flag word, not a boolean: cuda-gdb passes it through too, and is seen
    // passing 3 when it requests more capabilities than we do.
    CUDBGResult res = api->requestCleanupOnDetach(resume_for_detach);
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
      // PTRACE_CONT, so it has the same thread affinity requirement as the flag
      // writes below. The signal bypass is plain bookkeeping and stays here,
      // where it also covers the drain loop.
      if (Error err = RunOnNativeMainLoop(
              [cpu]() -> Error {
                ResumeActionList resume_actions(lldb::eStateRunning,
                                                LLDB_INVALID_SIGNAL_NUMBER);
                return cpu->Resume(resume_actions).ToError();
              },
              std::chrono::milliseconds(kNativeWorkTimeoutMs)))
        LLDB_LOG(log, "DetachCleanup: failed to resume the CPU process: {0}",
                 llvm::toString(std::move(err)));
    }

    // We occupy the GPU MainLoop thread the notifier would dispatch on, so
    // DETACH_COMPLETE has to be drained here rather than awaited.
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

  // 4. Clear the attach-specific driver state (unless the API is poisoned).
  if (api && !faulted) {
    CUDBGResult res = api->clearAttachState();
    if (res != CUDBG_SUCCESS)
      LLDB_LOG(log, "DetachCleanup: clearAttachState failed: {0}",
               cudbgGetErrorString(res));
  }

  // 5. Only now reset the driver handshake flags, so a later debugger
  // re-negotiates. Order matters: cudbgIpcFlag is what authorizes the driver to
  // emit callbacks, so clearing it before step 3 would cut off the very
  // mechanism that cleanup runs on, and the driver would go on believing a
  // debugger is attached -- which stops it re-running its attach procedure for
  // a later re-attach. Best-effort; a failed write does not abort the detach.
  if (cpu && symbols) {
    // Step 3 left the application running, and ptrace refuses to write to a
    // running tracee. cuda-gdb interrupts the target at this same point.
    bool was_halted = false;
    lldb::StateType state = HaltNativeProcessIfNeeded(was_halted);
    LLDB_LOG(log, "DetachCleanup: halted the app for the flag reset (state {0})",
             StateToString(state));

    llvm::StringMap<uint64_t> detach_symbols = *symbols;
    if (Error err = RunOnNativeMainLoop(
            [cpu, detach_symbols = std::move(detach_symbols)]() -> Error {
              auto addr_of =
                  [&detach_symbols](StringRef name) -> std::optional<uint64_t> {
                llvm::StringMap<uint64_t>::const_iterator it =
                    detach_symbols.find(name);
                if (it == detach_symbols.end())
                  return std::nullopt;
                return it->second;
              };
              return CUDADebuggerAPI::ResetDetachSymbols(addr_of, *cpu);
            },
            std::chrono::milliseconds(kNativeWorkTimeoutMs))) {
      LLDB_LOG(log, "DetachCleanup: failed to reset detach symbols: {0}",
               llvm::toString(std::move(err)));
    } else {
      LLDB_LOG(log, "DetachCleanup: reset the driver handshake flags");
    }
  }

  // 6. Finalize and drop the API, mirroring initialization. The notifier goes
  // first so nothing can dispatch into the API once it is gone, and the GPU
  // process's copy of the table is cleared so it cannot dangle.
  m_main_loop_event_notifier_up.reset();
  if (m_gpu) {
    m_gpu->SetSyncEventDrainNotifier(nullptr);
    m_gpu->ClearDebuggerAPI();
  }
  if (m_cuda_api) {
    LLDB_LOG(log, "DetachCleanup: finalizing the debugger API");
    m_cuda_api.reset();
  }

  // Return to a clean idle state. The caller (ProcessNVGPU::Detach) performs the
  // final SetState(eStateDetached).
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_attach_state = AttachState::eNone;
    m_api_initialized = false;
  }

  LLDB_LOG(log, "DetachCleanup: complete");
  return Error::success();
}

void LLDBServerPluginNVGPU::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  if (m_gpu)
    m_gpu->OnNativeProcessExit(exit_status);
}
