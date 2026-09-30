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
#include "lldb/Utility/State.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
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
#include <utility>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

namespace {
/// The signals to pass through the inferior without stopping it for the
/// duration of a debugger-driven resume: all but SIGTRAP, SIGKILL, SIGSTOP,
/// SIGCHLD and the debug API's SIGURG. Mirrors cuda_gdb_bypass_signals.
llvm::SmallVector<int, 64> GetBypassSignals() {
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
  return bypass;
}

/// Adapt a resolved symbol map to the SymbolAddressProvider callback shape.
std::optional<uint64_t> LookupSymbol(const llvm::StringMap<uint64_t> &symbols,
                                     StringRef name) {
  llvm::StringMap<uint64_t>::const_iterator it = symbols.find(name);
  if (it == symbols.end())
    return std::nullopt;
  return it->second;
}

/// How long the client may keep the process running for the late attach to
/// finish. NVGPU_ATTACH_WAIT_TIMEOUT_MS overrides \a default_timeout so tests
/// can force the timeout, which the driver otherwise never comes close to.
std::chrono::milliseconds
GetAttachWaitTimeout(std::chrono::milliseconds default_timeout) {
  std::optional<std::string> value =
      sys::Process::GetEnv("NVGPU_ATTACH_WAIT_TIMEOUT_MS");
  if (!value)
    return default_timeout;
  unsigned ms = 0;
  if (StringRef(*value).getAsInteger(10, ms)) {
    LLDB_LOG(GetLog(GDBRLog::Plugin),
             "ignoring NVGPU_ATTACH_WAIT_TIMEOUT_MS={0}, which is not a number "
             "of milliseconds",
             *value);
    return default_timeout;
  }
  return std::chrono::milliseconds(ms);
}

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

  if (m_gpu)
    m_gpu->SetPlugin(this);
}

llvm::StringRef LLDBServerPluginNVGPU::GetPluginName() { return "nvgpu"; }

std::optional<GPUActions> LLDBServerPluginNVGPU::NativeProcessIsStopping() {
  TryInitiateSafeAttach();
  return std::nullopt;
}

void LLDBServerPluginNVGPU::TryInitiateSafeAttach() {
  Log *log = GetLog(GDBRLog::Plugin);
  using std::chrono::steady_clock;

  // Called on the native MainLoop thread with the process stopped: for every
  // stop reply, and once qSymbol has delivered the last handshake symbol. The
  // ptrace writes below need both, and must not run under m_attach_mutex,
  // which the GPU event thread also takes. Nothing can race them: the only
  // other writer of m_attach_state runs on the GPU MainLoop thread, which
  // InitializeAPIAndConnect has not created yet.
  NativeProcessProtocol *cpu_process = nullptr;
  steady_clock::time_point deadline;
  llvm::StringMap<uint64_t> symbols;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    deadline = m_attach_deadline;
    cpu_process = m_native_process.GetCurrentProcess();
    symbols = m_libcuda_symbols;
  }

  // Everything below commits eInjected, falls back to eNone, or stays at
  // eProbing to retry at the next stop.
  if (!cpu_process) {
    LLDB_LOG(log, "TryInitiateSafeAttach: no current native process yet");
    return;
  }

  if (steady_clock::now() > deadline) {
    LLDB_LOG(log,
             "TryInitiateSafeAttach: timed out after {0}s trying to "
             "initiate the safe attach; falling back to launch-style "
             "initialization.",
             kAttachProbeTimeoutSeconds);
    SetAttachStateIfProbing(AttachState::eNone);
    return;
  }

  // Without libcuda, CUDA cannot have been initialized, and the launch-style
  // breakpoints from GetInitializeActions will catch it when it is.
  Expected<bool> libcuda_loaded =
      CUDADebuggerAPI::IsLibcudaLoaded(*cpu_process);
  if (!libcuda_loaded) {
    LLDB_LOG(log,
             "TryInitiateSafeAttach: could not read the loaded library "
             "list: {0}",
             llvm::toString(libcuda_loaded.takeError()));
  } else if (!*libcuda_loaded) {
    LLDB_LOG(log, "TryInitiateSafeAttach: libcuda is not loaded; CUDA "
                  "will be initialized as on launch.");
    SetAttachStateIfProbing(AttachState::eNone);
    return;
  }

  // The client looks these up through qSymbol while it loads modules.
  for (const std::string &name : CUDADebuggerAPI::GetSafeAttachSymbolNames())
    if (!symbols.contains(name))
      return;

  auto get_addr = [&symbols](StringRef name) {
    return LookupSymbol(symbols, name);
  };
  Expected<bool> initiated =
      CUDADebuggerAPI::InitiateSafeAttach(get_addr, *cpu_process);
  if (!initiated) {
    std::string message = llvm::toString(initiated.takeError());
    LLDB_LOG(log, "TryInitiateSafeAttach: failed to initiate safe attach: {0}",
             message);
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_attach_start_error = std::move(message);
    return;
  }
  if (!*initiated) {
    LLDB_LOG(log, "TryInitiateSafeAttach: the driver has not finished "
                  "initializing; retrying at the next stop.");
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_attach_start_error.clear();
    return;
  }

  SetAttachStateIfProbing(AttachState::eInjected);
  LLDB_LOG(log, "TryInitiateSafeAttach: safe attach initiated; waiting for "
                "CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED.");
}

void LLDBServerPluginNVGPU::SetAttachStateIfProbing(AttachState state) {
  std::lock_guard<std::mutex> guard(m_attach_mutex);
  if (m_attach_state == AttachState::eProbing)
    m_attach_state = state;
}

std::vector<std::string> LLDBServerPluginNVGPU::GetSymbolsToLookUp() {
  std::lock_guard<std::mutex> guard(m_attach_mutex);
  if (m_attach_state != AttachState::eProbing)
    return {};
  std::vector<std::string> missing;
  for (std::string &name : CUDADebuggerAPI::GetSafeAttachSymbolNames())
    if (!m_libcuda_symbols.contains(name))
      missing.push_back(std::move(name));
  return missing;
}

void LLDBServerPluginNVGPU::SymbolLookedUp(StringRef name,
                                           std::optional<uint64_t> value) {
  std::vector<std::string> wanted = CUDADebuggerAPI::GetSafeAttachSymbolNames();
  if (!value || !llvm::is_contained(wanted, name))
    return;
  bool have_all = true;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state != AttachState::eProbing)
      return;
    m_libcuda_symbols[name] = *value;
    for (const std::string &wanted_name : wanted)
      have_all &= m_libcuda_symbols.contains(wanted_name);
  }
  // The client looks symbols up while it completes the attach, with the
  // process still stopped at the attach stop, so the handshake can run now.
  if (have_all)
    TryInitiateSafeAttach();
}

GPUPluginFinishAttachResponse
LLDBServerPluginNVGPU::FinishAttach(const GPUPluginFinishAttachArgs &args) {
  GPUPluginFinishAttachResponse finish;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    // Still probing means libcuda is loaded, so CUDA may already be running
    // and its GPU would never appear without a word to the user.
    if (m_attach_state == AttachState::eProbing) {
      for (const std::string &name :
           CUDADebuggerAPI::GetSafeAttachSymbolNames()) {
        if (!m_libcuda_symbols.contains(name)) {
          finish.warnings.push_back(
              llvm::formatv("cannot attach to the GPU: {0} was not found in "
                            "libcuda; the CUDA driver may be too old to "
                            "support attaching to a running application",
                            name)
                  .str());
          break;
        }
      }
      if (finish.warnings.empty() && !m_attach_start_error.empty())
        finish.warnings.push_back("cannot attach to the GPU: " +
                                  m_attach_start_error);
    }
    // Until the driver has the request, running the process will not finish
    // the attach. cuda-gdb does not continue in that case either.
    if (!args.may_resume || m_attach_state != AttachState::eInjected)
      return finish;
    m_client_waiting_for_attach = true;
  }
  LLDB_LOG(GetLog(GDBRLog::Plugin),
           "FinishAttach: the client will run the process until the attach "
           "completes");

  // Stop the process if the attach takes too long, so the client's attach
  // cannot hang. If the client is briefly holding the process at a breakpoint,
  // the SIGSTOP stays pending and stops it as soon as the client resumes it.
  const std::chrono::milliseconds timeout =
      GetAttachWaitTimeout(std::chrono::seconds(kAttachWaitTimeoutSeconds));
  m_native_process.GetMainLoop().AddCallback(
      [this, timeout](MainLoopBase &) {
        {
          std::lock_guard<std::mutex> guard(m_attach_mutex);
          if (!std::exchange(m_client_waiting_for_attach, false))
            return;
        }
        LLDB_LOG(GetLog(GDBRLog::Plugin),
                 "NVGPU late attach: not finished after {0}ms; stopping the "
                 "process so the attach completes without the GPU for now.",
                 timeout.count());
        if (NativeProcessProtocol *cpu = m_native_process.GetCurrentProcess())
          cpu->Halt();
      },
      timeout);
  finish.resume = true;
  return finish;
}

void LLDBServerPluginNVGPU::OnAttachComplete() {
  Log *log = GetLog(GDBRLog::Plugin);
  bool client_waiting = false;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    if (m_attach_state == AttachState::eComplete ||
        m_attach_state == AttachState::eDetaching)
      return;
    m_attach_state = AttachState::eComplete;
    client_waiting = std::exchange(m_client_waiting_for_attach, false);
  }
  LLDB_LOG(log, "LLDBServerPluginNVGPU::OnAttachComplete(). Refreshing device "
                "state so CUDA threads are enumerated.");
  m_gpu->SuspendAllDevicesAndRefresh("attached to running CUDA application");
  // A waiting client needs this stop to end its attach, even when a GPU stop
  // otherwise leaves the CPU running.
  if (client_waiting ||
      sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1")
    HaltNativeProcess();
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
  // Undo the partial setup if a step below fails, so that nothing is left
  // pointing into an API the next initialization attempt replaces.
  auto release_api = llvm::make_scope_exit([this] { ReleaseDebuggerAPI(); });

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

  // Deferred until the event callback exists, so the driver cannot emit a
  // notification before anything can observe it. Initialize already published
  // the IPC flag on the launch path.
  if (is_late_attach) {
    if (Error err = FinishLateAttachIpcHandshake(get_symbol_address))
      return err;
  }

  Expected<GPUPluginConnectionInfo> connection_info = CreateConnection();
  if (!connection_info)
    return connection_info.takeError();

  GPUActions actions = GetNewGPUAction();
  actions.connect_info = std::move(*connection_info);
  release_api.release();
  return actions;
}

llvm::Error LLDBServerPluginNVGPU::FinishLateAttachIpcHandshake(
    SymbolAddressProvider get_symbol_address) {
  Log *log = GetLog(GDBRLog::Plugin);
  NativeProcessProtocol *cpu_process = m_native_process.GetCurrentProcess();
  if (!cpu_process)
    return createStringError(
        "No native process available to complete the late attach handshake");

  if (Error err = CUDADebuggerAPI::SetIpcFlag(get_symbol_address, *cpu_process))
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
  m_main_loop.AddPendingCallback(
      [this](MainLoopBase &) { OnAttachComplete(); });
  return Error::success();
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  std::string library_name = *args.breakpoint.name_info->shlib;
  // Both libcuda sonames can resolve, so this fires more than once.
  // Initializing twice would build a second API table and reverse connection
  // and leak the live one.
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

  // Still probing means CUDA came up via the launch-style breakpoint during an
  // attach, which makes this a launch; leaving eProbing stops the handshake
  // retries.
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_api_initialized = true;
    if (m_attach_state == AttachState::eProbing)
      m_attach_state = AttachState::eNone;
    // Kept for detach, which has no breakpoint of its own to carry them.
    for (const SymbolValue &symbol : args.symbol_values)
      if (symbol.value)
        m_libcuda_symbols[symbol.name] = *symbol.value;
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
  if (Expected<int> handled = DrainSyncEventsOnce(); !handled)
    logAndReportFatalError(llvm::toString(handled.takeError()));
}

Expected<int> LLDBServerPluginNVGPU::DrainSyncEventsOnce() {
  Log *log = GetLog(GDBRLog::Plugin);
  CUDADebuggerAPI &cuda_api = *m_cuda_api;

  int events_handled = 0;
  while (true) {
    // A poisoned API must not be touched again.
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
      // Leaving eDetaching is what ends DrainUntilDetachComplete.
      std::lock_guard<std::mutex> guard(m_attach_mutex);
      if (m_attach_state == AttachState::eDetaching)
        m_attach_state = AttachState::eNone;
      break;
    }
    case CUDBG_EVENT_ALL_DEVICES_SUSPENDED: {
      LLDB_LOG(log, "broken devices {0:x}, faulted devices {1:x}",
               event.cases.allDevicesSuspended.brokenDevicesMask,
               event.cases.allDevicesSuspended.faultedDevicesMask);
      m_gpu->OnAllDevicesSuspended(
          event.cases.allDevicesSuspended,
          [this](llvm::StringRef message) { SendMonitorLogToClient(message); });
      // Here we can force the two processes to be in sync. Not synchronizing
      // them allows for a non-stop mode for the native process.
      if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1")
        HaltNativeProcess();
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
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to acknowledge CUDA Debugger API events. {0}",
        cudbgGetErrorString(res));
  return events_handled;
}

void LLDBServerPluginNVGPU::SendMonitorLogToClient(llvm::StringRef message) {
  // We are about to report a stop, so the client is waiting for the stop reply
  // and will not mistake this for the response to a pending query.
  if (m_gpu->GetState() != lldb::eStateRunning)
    logAndReportFatalError(
        "Logging to client is only supported when the GPU is running.");

  m_gdb_server->SendStructuredDataPacket(
      llvm::json::Value(llvm::json::Object{{"type", "nvgpu-monitor"},
                                           {"subtype", "log"},
                                           {"message", message}}));
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
  // shows the error. Refreshing device state would go through the poisoned API,
  // and the decoding treats a failed read as fatal.
  m_gpu->ReportFallbackStop(description);
  if (sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1")
    HaltNativeProcess();
}

llvm::Error LLDBServerPluginNVGPU::RunOnNativeMainLoop(
    std::function<llvm::Error(NativeProcessProtocol &)> work) {
  // Held by both this frame and the callback, so a timeout here cannot leave
  // the native thread writing into a destroyed promise.
  struct SharedWork {
    std::function<llvm::Error(NativeProcessProtocol &)> work;
    /// An empty string means success; otherwise the rendered error.
    std::promise<std::string> result;
  };
  std::shared_ptr<SharedWork> shared = std::make_shared<SharedWork>();
  shared->work = std::move(work);
  std::future<std::string> future = shared->result.get_future();

  auto run = [this, shared](MainLoopBase &) {
    std::string message;
    NativeProcessProtocol *process = m_native_process.GetCurrentProcess();
    if (!process)
      message = "the native process has exited";
    else if (Error err = shared->work(*process))
      message = llvm::toString(std::move(err));
    shared->result.set_value(std::move(message));
  };
  if (!m_native_process.GetMainLoop().AddPendingCallback(run))
    return createStringError("the native MainLoop is no longer accepting work");

  const std::chrono::milliseconds timeout(kNativeWorkTimeoutMs);
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

lldb::StateType LLDBServerPluginNVGPU::HaltNativeProcess() {
  Log *log = GetLog(GDBRLog::Plugin);
  // Written on the native thread. Shared so that a check still queued after
  // RunOnNativeMainLoop gives up has somewhere to write.
  auto state = std::make_shared<lldb::StateType>(lldb::eStateInvalid);
  auto halt = [state](NativeProcessProtocol &cpu) -> Error {
    if (cpu.IsRunning()) {
      if (Error err = cpu.Halt().ToError())
        return err;
    }
    *state = cpu.GetState();
    return Error::success();
  };
  auto check = [state](NativeProcessProtocol &cpu) -> Error {
    *state = cpu.GetState();
    return Error::success();
  };

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(kNativeHaltTimeoutSeconds);
  if (Error err = RunOnNativeMainLoop(halt)) {
    LLDB_LOG(log, "HaltNativeProcess: {0}", llvm::toString(std::move(err)));
    return lldb::eStateInvalid;
  }
  while (StateIsRunningState(*state) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    if (Error err = RunOnNativeMainLoop(check)) {
      LLDB_LOG(log, "HaltNativeProcess: {0}", llvm::toString(std::move(err)));
      return lldb::eStateInvalid;
    }
  }
  return *state;
}

void LLDBServerPluginNVGPU::DetachCleanup() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "LLDBServerPluginNVGPU::DetachCleanup()");

  // A faulted API must not be called again; the steps below skip it.
  bool faulted = false;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    faulted = m_api_faulted;
    m_attach_state = AttachState::eDetaching;
  }

  // 1. Tear the device breakpoints down first, while the API and devices are
  // still valid. The debug API otherwise leaves them set on the device. lldb
  // removes its own before detaching, so this catches any a client left.
  if (m_gpu && !faulted)
    m_gpu->TeardownDeviceBreakpoints();

  CUDBGAPI api = m_gpu ? m_gpu->GetDebuggerAPI() : nullptr;

  // Only read before step 3 resumes the application: once it runs it can exit,
  // and the native thread then destroys this process.
  NativeProcessProtocol *cpu = m_native_process.GetCurrentProcess();

  llvm::StringMap<uint64_t> symbols;
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    symbols = m_libcuda_symbols;
  }
  if (symbols.empty())
    LLDB_LOG(log, "DetachCleanup: no libcuda symbols were ever delivered; "
                  "skipping the driver handshake cleanup");
  auto get_addr = [&symbols](StringRef name) {
    return LookupSymbol(symbols, name);
  };

  // 2. Whether the driver needs the application resumed to complete its
  // cleanup. Kept raw because requestCleanupOnDetach takes it verbatim below.
  uint32_t resume_for_detach = 0;
  if (cpu && !symbols.empty()) {
    if (Expected<uint32_t> resume =
            CUDADebuggerAPI::ReadResumeForAttachDetach(get_addr, *cpu))
      resume_for_detach = *resume;
    else
      LLDB_LOG(log, "DetachCleanup: could not read the resume flag: {0}",
               llvm::toString(resume.takeError()));
  }
  LLDB_LOG(log, "DetachCleanup: resume_for_detach={0}, api_faulted={1}",
           resume_for_detach, faulted);

  // 3. Let the driver clean up, which it can only do while the app runs.
  if (resume_for_detach && api && !faulted)
    ResumeForDriverCleanup(api, resume_for_detach);

  // 4. Clear the attach-specific driver state.
  if (api && !faulted) {
    CUDBGResult res = api->clearAttachState();
    if (res != CUDBG_SUCCESS)
      LLDB_LOG(log, "DetachCleanup: clearAttachState failed: {0}",
               cudbgGetErrorString(res));
  }

  // 5. Only now reset the handshake flags, so a later debugger re-negotiates.
  // cudbgIpcFlag is what authorizes the driver to emit callbacks, so clearing
  // it before step 3 cuts off the mechanism that cleanup runs on and leaves the
  // driver believing a debugger is still attached, which in turn stops it
  // re-running its attach procedure for a later re-attach.
  if (!symbols.empty())
    ResetDriverHandshakeFlags(symbols);

  // 6. Drop the API and everything pointing into it.
  ReleaseDebuggerAPI();

  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_attach_state = AttachState::eNone;
    m_api_initialized = false;
  }

  LLDB_LOG(log, "DetachCleanup: complete");
}

void LLDBServerPluginNVGPU::ResumeForDriverCleanup(CUDBGAPI api,
                                                   uint32_t resume_flags) {
  Log *log = GetLog(GDBRLog::Plugin);

  // A flag word, not a boolean: cuda-gdb passes it through too, and is seen
  // passing 3 when it requests more capabilities than we do.
  CUDBGResult res = api->requestCleanupOnDetach(resume_flags);
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

  // Resuming is PTRACE_CONT, so it has to happen on the native thread. So does
  // changing the ignored signals, which that thread reads whenever the inferior
  // gets one; passing them through keeps an ordinary signal from stopping the
  // application before the driver has finished.
  std::shared_ptr<llvm::DenseSet<int>> saved_signals =
      std::make_shared<llvm::DenseSet<int>>();
  auto resume = [saved_signals](NativeProcessProtocol &cpu) -> Error {
    *saved_signals = cpu.GetIgnoredSignals();
    cpu.IgnoreSignals(GetBypassSignals());
    // Still running if the GPU stop did not stop the CPU; see
    // NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP.
    if (cpu.IsRunning())
      return Error::success();
    ResumeActionList resume_actions(lldb::eStateRunning,
                                    LLDB_INVALID_SIGNAL_NUMBER);
    return cpu.Resume(resume_actions).ToError();
  };
  if (Error err = RunOnNativeMainLoop(resume))
    LLDB_LOG(log, "DetachCleanup: failed to resume the CPU process: {0}",
             llvm::toString(std::move(err)));

  DrainUntilDetachComplete();

  auto restore_signals = [saved_signals](NativeProcessProtocol &cpu) -> Error {
    cpu.IgnoreSignals(llvm::to_vector(*saved_signals));
    return Error::success();
  };
  if (Error err = RunOnNativeMainLoop(restore_signals))
    LLDB_LOG(log, "DetachCleanup: failed to restore the ignored signals: {0}",
             llvm::toString(std::move(err)));
}

void LLDBServerPluginNVGPU::DrainUntilDetachComplete() {
  Log *log = GetLog(GDBRLog::Plugin);

  auto finished = [this]() {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    return m_attach_state != AttachState::eDetaching || m_api_faulted ||
           m_native_process_exited;
  };

  for (int i = 0; i < kDetachMaxIterations && !finished(); ++i) {
    Expected<int> handled = DrainSyncEventsOnce();
    if (!handled) {
      LLDB_LOG(log, "DetachCleanup: stopped waiting for the driver: {0}",
               llvm::toString(handled.takeError()));
      return;
    }
    if (*handled == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (!finished())
    LLDB_LOG(log,
             "DetachCleanup: detach cleanup did not complete after {0} "
             "iterations; proceeding with teardown anyway.",
             kDetachMaxIterations);
}

void LLDBServerPluginNVGPU::ResetDriverHandshakeFlags(
    const llvm::StringMap<uint64_t> &symbols) {
  Log *log = GetLog(GDBRLog::Plugin);

  // Step 3 left the application running, and ptrace refuses to write to a
  // running tracee. cuda-gdb interrupts the target at this same point.
  lldb::StateType state = HaltNativeProcess();
  LLDB_LOG(log, "DetachCleanup: halted the app for the flag reset (state {0})",
           StateToString(state));

  auto reset = [symbols](NativeProcessProtocol &cpu) -> Error {
    return CUDADebuggerAPI::ResetDetachSymbols(
        [&symbols](StringRef name) { return LookupSymbol(symbols, name); },
        cpu);
  };
  if (Error err = RunOnNativeMainLoop(reset))
    LLDB_LOG(log, "DetachCleanup: failed to reset detach symbols: {0}",
             llvm::toString(std::move(err)));
  else
    LLDB_LOG(log, "DetachCleanup: reset the driver handshake flags");
}

void LLDBServerPluginNVGPU::ReleaseDebuggerAPI() {
  Log *log = GetLog(GDBRLog::Plugin);

  // The notifier goes first so nothing can dispatch into the API once it is
  // gone, and the GPU process's copy of the table is cleared so it cannot
  // dangle.
  m_main_loop_event_notifier_up.reset();
  if (m_gpu) {
    m_gpu->SetSyncEventDrainNotifier(nullptr);
    m_gpu->ClearDebuggerAPI();
  }
  if (m_cuda_api) {
    LLDB_LOG(log, "ReleaseDebuggerAPI: finalizing the debugger API");
    m_cuda_api.reset();
  }
}

void LLDBServerPluginNVGPU::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  {
    std::lock_guard<std::mutex> guard(m_attach_mutex);
    m_native_process_exited = true;
    m_client_waiting_for_attach = false;
  }
  if (m_gpu)
    m_gpu->OnNativeProcessExit(exit_status);
}
