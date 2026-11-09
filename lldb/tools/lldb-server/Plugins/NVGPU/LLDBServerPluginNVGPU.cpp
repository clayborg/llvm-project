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
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/CUDADebuggerAPIVersion.h"
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
#include <utility>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

/// Whether a GPU stop, at a breakpoint, an exception, an internal error or the
/// end of a late attach, stops the native process too.
/// NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1 leaves it running.
static bool CPUStopsWithGPU() {
  return sys::Process::GetEnv("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1";
}

static StringRef EventKindName(CUDBGEventKind kind) {
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
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(193)
    CUDBG_EVENT_CASE(CUDBG_EVENT_SINGLE_STEP_COMPLETE)
    CUDBG_EVENT_CASE(CUDBG_EVENT_CUDA_LOGS_RULESET_CHANGED)
#endif
  }
#undef CUDBG_EVENT_CASE
  return "unknown CUDBGEventKind";
}

/// A response that disables the breakpoint that was hit. Each of the plug-in's
/// breakpoints is needed until it has done its part.
static GPUPluginBreakpointHitResponse DisablingResponse(GPUActions actions) {
  GPUPluginBreakpointHitResponse response(std::move(actions));
  response.disable_bp = true;
  return response;
}

/// The thread of \a process at \a pc, preferring the current thread. lldb
/// selects other threads to read their registers, so the current thread need
/// not be the one that hit a breakpoint.
static NativeThreadProtocol *FindThreadAt(NativeProcessProtocol &process,
                                          addr_t pc) {
  NativeThreadProtocol *current = process.GetCurrentThread();
  if (current && current->GetRegisterContext().GetPC() == pc)
    return current;
  for (NativeThreadProtocol &thread : process.Threads())
    if (thread.GetRegisterContext().GetPC() == pc)
      return &thread;
  return nullptr;
}

/// The address that \a thread returns to from the function at whose first
/// instruction it is stopped: in the link register on AArch64, and on top of
/// the stack, where the call has just pushed it, on x86-64.
static Expected<addr_t> ReadReturnAddress(NativeProcessProtocol &process,
                                          NativeThreadProtocol &thread) {
  NativeRegisterContext &registers = thread.GetRegisterContext();
  if (addr_t return_address = registers.GetReturnAddress();
      return_address != LLDB_INVALID_ADDRESS)
    return return_address;

  addr_t sp = registers.GetSP();
  if (sp == LLDB_INVALID_ADDRESS)
    return createStringErrorFmt(
        "Failed to read the stack pointer of thread {0}", thread.GetID());
  uint8_t bytes[8];
  const uint32_t address_size = process.GetAddressByteSize();
  if (address_size > sizeof(bytes))
    return createStringErrorFmt("Unexpected address size {0}", address_size);
  size_t bytes_read = 0;
  Status status = process.ReadMemory(sp, bytes, address_size, bytes_read);
  if (status.Fail() || bytes_read != address_size)
    return createStringErrorFmt(
        "Failed to read the return address of thread {0} at {1:x}",
        thread.GetID(), sp);
  lldb_private::DataExtractor data(bytes, address_size, process.GetByteOrder(),
                                   address_size);
  lldb::offset_t offset = 0;
  return data.GetAddress(&offset);
}

/// The value of the symbol \a name in \a symbols, if lldb resolved it.
static std::optional<uint64_t> FindSymbolValue(ArrayRef<SymbolValue> symbols,
                                               StringRef name) {
  for (const SymbolValue &symbol : symbols)
    if (symbol.name == name)
      return symbol.value;
  return std::nullopt;
}

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
  return TryLateAttach();
}

std::optional<GPUActions> LLDBServerPluginNVGPU::SetRequestedSymbols(
    const std::vector<SymbolValue> &symbols) {
  // lldb delivers libcuda's symbols together, once libcuda has loaded. Only a
  // libcuda that predates the driver's attach procedure lacks some, which
  // TryLateAttach reports.
  if (m_libcuda_symbols_received || m_api_initialized)
    return std::nullopt;
  m_libcuda_symbols_received = true;
  m_late_attach_symbols = symbols;
  std::optional<GPUActions> actions = TryLateAttach();
  // When lldb launched the application, or attached before it initialized
  // CUDA, the initialization breakpoints bring the debugger API up instead.
  if (!m_late_attach_symbols.empty())
    LLDB_LOGV(GetLog(GDBRLog::Plugin),
              "The CUDA driver cannot attach before the application has a "
              "CUDA context. lldb-server asks it to attach again each time "
              "the process stops");
  return actions;
}

std::optional<GPUActions> LLDBServerPluginNVGPU::TryLateAttach() {
  // A call to cuInit that the initialization breakpoints follow can bring the
  // debugger API up, which its return shows first.
  if (m_late_attach_symbols.empty() || m_pending_cuinit)
    return std::nullopt;
  // jGPUPluginSymbols and stops are handled on the native server's thread, the
  // only one that may use its process, while the process is stopped.
  NativeProcessProtocol *native_process = m_native_process.GetCurrentProcess();
  if (!native_process)
    return std::nullopt;

  auto get_symbol_address = [this](StringRef name) {
    return FindSymbolValue(m_late_attach_symbols, name);
  };
  Expected<bool> requested =
      CUDADebuggerAPI::RequestLateAttach(get_symbol_address, *native_process);
  if (!requested) {
    m_late_attach_symbols.clear();
    LLDB_LOG_ERROR(GetLog(GDBRLog::Plugin), requested.takeError(),
                   "Cannot attach to the GPU of the running application: {0}");
    return std::nullopt;
  }
  if (!*requested)
    return std::nullopt;
  m_late_attach_symbols.clear();
  m_late_attach_requested = true;

  // The driver calls the function these breakpoints are on once the
  // application runs, and lldb sets them before it resumes the application.
  GPUActions actions = GetNewGPUAction();
  actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME));
  actions.breakpoints.emplace_back(
      CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(
          CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME_ALT));
  return actions;
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

Error LLDBServerPluginNVGPU::SetUpDebuggerAPI(
    SymbolAddressProvider get_symbol_address, StringRef libcuda_library_name,
    NativeProcessProtocol &native_process) {
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu)
    return createStringError(
        "No GPU process to initialize the CUDA Debugger API for");

  Expected<CUDADebuggerAPI> api_or = CUDADebuggerAPI::Initialize(
      get_symbol_address, libcuda_library_name, native_process);
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
    return createStringErrorFmt(
        "Failed to set the event callback for the CUDA Debugger API. {0}",
        cudbgGetErrorString(res));

  release_api.release();
  return Error::success();
}

Expected<GPUActions>
LLDBServerPluginNVGPU::Connect(function_ref<Error()> before_accept) {
  Expected<GPUPluginConnectionInfo> connection_info =
      CreateConnection(before_accept);
  if (!connection_info)
    return connection_info.takeError();

  GPUActions actions = GetNewGPUAction();
  actions.connect_info = std::move(*connection_info);
  actions.session_name = "NVIDIA GPU Session";
  return actions;
}

Expected<GPUActions> LLDBServerPluginNVGPU::InitializeAPIForLateAttach(
    SymbolAddressProvider get_symbol_address, StringRef libcuda_library_name,
    NativeProcessProtocol &native_process) {
  if (Error err = SetUpDebuggerAPI(get_symbol_address, libcuda_library_name,
                                   native_process))
    return err;

  // Only tell the driver that a debugger is ready once every other step has
  // succeeded, the event callback included, so that it cannot emit a
  // notification before anything can observe it. If one fails, the client
  // resumes the process, which runs on without GPU debugging instead of
  // waiting for a debugger that does not answer.
  Expected<GPUActions> actions = Connect(
      [&] { return FinishLateAttach(get_symbol_address, native_process); });
  if (!actions)
    ReleaseDebuggerAPI();
  return actions;
}

Error LLDBServerPluginNVGPU::FinishLateAttach(
    SymbolAddressProvider get_symbol_address,
    NativeProcessProtocol &native_process) {
  Expected<uint32_t> resume_for_attach =
      CUDADebuggerAPI::ReadResumeForAttachDetach(get_symbol_address,
                                                 native_process);
  if (!resume_for_attach)
    return resume_for_attach.takeError();
  if (Error err =
          CUDADebuggerAPI::SetIpcFlag(get_symbol_address, native_process))
    return err;

  // The GPU MainLoop thread, which uses m_attach_state from now on, starts
  // after this returns.
  Log *log = GetLog(GDBRLog::Plugin);
  if (*resume_for_attach) {
    LLDB_LOG(log, "The CUDA driver replays the application's contexts and "
                  "modules before the attach finishes");
    m_attach_state = AttachState::Replaying;
  } else {
    LLDB_LOG(log, "The CUDA driver attached without replaying any of the "
                  "application's contexts and modules");
    m_attach_state = AttachState::Unreported;
  }
  return Error::success();
}

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {
  // Breakpoint hits arrive on the native server's thread, the only one that
  // may use its process, and the process stays stopped until this returns.
  NativeProcessProtocol *native_process = m_native_process.GetCurrentProcess();
  if (!native_process)
    return createStringError(
        "No native process to initialize the CUDA Debugger API for");

  // Only the breakpoint where cuInit returns is set by address.
  if (args.breakpoint.addr_info)
    return CuInitReturned(args, *native_process);

  // Both libcuda sonames can resolve, so each breakpoint can fire more than
  // once. Initializing twice would build a second API table and reverse
  // connection and leak the live one.
  if (m_api_initialized) {
    Log *log = GetLog(GDBRLog::Plugin);
    LLDB_LOG(log, "debugger API already initialized; ignoring breakpoint and "
                  "disabling it.");
    return DisablingResponse(GetNewGPUAction());
  }

  const std::optional<GPUBreakpointByName> &name_info =
      args.breakpoint.name_info;
  if (!name_info || !name_info->shlib)
    return createStringError(
        "The breakpoint names no library to initialize the CUDA Debugger API "
        "from");
  StringRef library_name = *name_info->shlib;
  std::optional<CUDADebuggerAPI::BreakpointFunction> function =
      CUDADebuggerAPI::GetBreakpointFunction(name_info->function_name);
  if (!function)
    return createStringErrorFmt("Unexpected breakpoint on {0}",
                                name_info->function_name);

  if (*function == CUDADebuggerAPI::BreakpointFunction::AttachFinished) {
    auto get_symbol_address = [&args](StringRef name) {
      return args.GetSymbolValue(name);
    };
    Expected<GPUActions> actions = InitializeAPIForLateAttach(
        get_symbol_address, library_name, *native_process);
    // Leave m_api_initialized false so a later breakpoint can retry the init.
    if (!actions)
      return actions.takeError();
    m_api_initialized = true;
    return DisablingResponse(std::move(*actions));
  }

  // The initialization breakpoints can bring the debugger API up only in the
  // call to cuInit that initializes the driver. Once the plug-in has asked the
  // driver to attach, or found that the driver initialized before lldb
  // attached, the driver's attach procedure brings it up instead.
  if (m_late_attach_requested || m_driver_initialized_before_attach) {
    LLDB_LOG(GetLog(GDBRLog::Plugin),
             "The CUDA driver has initialized already; disabling the "
             "breakpoint on {0}",
             name_info->function_name);
    return DisablingResponse(GetNewGPUAction());
  }
  if (*function == CUDADebuggerAPI::BreakpointFunction::CuInit)
    return CuInitWasCalled(args, library_name, *native_process);
  return PreInitWasCalled(args, *native_process);
}

Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::CuInitWasCalled(const GPUPluginBreakpointHitArgs &args,
                                       StringRef libcuda_library_name,
                                       NativeProcessProtocol &native_process) {
  // Another thread's call is under way, or this is the same call's breakpoint
  // in the other libcuda soname. The driver initializes in one call only, and
  // the pending one finds out whether it was that call.
  if (m_pending_cuinit)
    return GPUPluginBreakpointHitResponse(GetNewGPUAction());

  Log *log = GetLog(GDBRLog::Plugin);
  auto get_symbol_address = [&args](StringRef name) {
    return args.GetSymbolValue(name);
  };
  Expected<bool> engine_initialized = CUDADebuggerAPI::ReadDebuggerInitialized(
      get_symbol_address, native_process);
  if (!engine_initialized)
    return engine_initialized.takeError();
  // An earlier debugger's debug engine is up, so the flag cannot show whether
  // the driver picks up this debugger.
  if (*engine_initialized) {
    LLDB_LOG(log, "The CUDA debug engine was initialized before lldb "
                  "attached; disabling the breakpoint on cuInit");
    m_driver_initialized_before_attach = true;
    return DisablingResponse(GetNewGPUAction());
  }

  std::optional<uint64_t> cuinit_address =
      CUDADebuggerAPI::GetCuInitAddress(get_symbol_address);
  NativeThreadProtocol *thread =
      cuinit_address ? FindThreadAt(native_process, *cuinit_address) : nullptr;
  if (!thread)
    return createStringError("No thread is at the breakpoint on cuInit");
  Expected<addr_t> return_address = ReadReturnAddress(native_process, *thread);
  if (!return_address)
    return return_address.takeError();

  m_pending_cuinit =
      PendingCuInit{thread->GetID(), *return_address,
                    libcuda_library_name.str(), args.symbol_values};
  // Without the hook, set up now, which is still before the driver
  // initializes.
  if (!CUDADebuggerAPI::HasPreInit(get_symbol_address)) {
    if (Error err = SetUpForCuInit(get_symbol_address, native_process)) {
      m_pending_cuinit.reset();
      return err;
    }
  }

  LLDB_LOG(log,
           "Following the call to cuInit on thread {0}, which returns "
           "to {1:x}",
           thread->GetID(), *return_address);
  GPUActions actions = GetNewGPUAction();
  GPUBreakpointInfo return_breakpoint;
  return_breakpoint.addr_info = GPUBreakpointByAddress{*return_address};
  actions.breakpoints.push_back(std::move(return_breakpoint));
  return GPUPluginBreakpointHitResponse(std::move(actions));
}

Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::PreInitWasCalled(const GPUPluginBreakpointHitArgs &args,
                                        NativeProcessProtocol &native_process) {
  // Every call to cuInit hits the breakpoint at its entry before the hook, so
  // without a pending call, the driver started to initialize elsewhere, where
  // no return shows whether it picked up the debugger API. A call that is set
  // up already hits the hook again in the other libcuda soname.
  if (!m_pending_cuinit || m_pending_cuinit->set_up) {
    if (!m_pending_cuinit)
      LLDB_LOG(GetLog(GDBRLog::Plugin),
               "CUDBG_PRE_INIT was called outside a call to cuInit that "
               "lldb-server follows; ignoring it");
    return GPUPluginBreakpointHitResponse(GetNewGPUAction());
  }

  auto get_symbol_address = [&args](StringRef name) {
    return args.GetSymbolValue(name);
  };
  if (Error err = SetUpForCuInit(get_symbol_address, native_process))
    return err;
  return GPUPluginBreakpointHitResponse(GetNewGPUAction());
}

Error LLDBServerPluginNVGPU::SetUpForCuInit(
    SymbolAddressProvider get_symbol_address,
    NativeProcessProtocol &native_process) {
  if (Error err = SetUpDebuggerAPI(get_symbol_address,
                                   m_pending_cuinit->libcuda_library_name,
                                   native_process))
    return err;
  // Tell the driver last, as InitializeAPIForLateAttach does.
  if (Error err =
          CUDADebuggerAPI::SetIpcFlag(get_symbol_address, native_process)) {
    ReleaseDebuggerAPI();
    return err;
  }
  m_pending_cuinit->set_up = true;
  return Error::success();
}

Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginNVGPU::CuInitReturned(const GPUPluginBreakpointHitArgs &args,
                                      NativeProcessProtocol &native_process) {
  const addr_t address = args.breakpoint.addr_info->load_address;
  if (!m_pending_cuinit || m_pending_cuinit->return_address != address)
    return DisablingResponse(GetNewGPUAction());
  // Other threads can run the code that cuInit returns to.
  NativeThreadProtocol *thread =
      native_process.GetThreadByID(m_pending_cuinit->tid);
  if (thread && thread->GetRegisterContext().GetPC() != address)
    return GPUPluginBreakpointHitResponse(GetNewGPUAction());

  Log *log = GetLog(GDBRLog::Plugin);
  PendingCuInit cuinit = std::move(*m_pending_cuinit);
  m_pending_cuinit.reset();
  if (!cuinit.set_up) {
    LLDB_LOG(log, "cuInit returned before lldb-server could set up the "
                  "debugger API for it");
    return DisablingResponse(GetNewGPUAction());
  }

  auto get_symbol_address = [&cuinit](StringRef name) {
    return FindSymbolValue(cuinit.symbols, name);
  };
  Expected<bool> engine_initialized = CUDADebuggerAPI::ReadDebuggerInitialized(
      get_symbol_address, native_process);
  if (!engine_initialized) {
    // Finalizing the debugger API while the driver uses it can crash the
    // application.
    ReleaseUnservedDebuggerAPI();
    return engine_initialized.takeError();
  }
  if (*engine_initialized) {
    Expected<GPUActions> actions =
        Connect([]() -> Error { return Error::success(); });
    if (!actions) {
      ReleaseUnservedDebuggerAPI();
      return actions.takeError();
    }
    m_api_initialized = true;
    // The initialization breakpoints brought the API up, so the driver must
    // not be asked to attach too.
    m_late_attach_symbols.clear();
    return DisablingResponse(std::move(*actions));
  }

  LLDB_LOG(log, "The CUDA driver initialized before lldb attached, and did "
                "not pick up the debugger API. It can attach only once the "
                "application has a CUDA context, and lldb-server asks it to "
                "attach each time the process stops");
  m_driver_initialized_before_attach = true;
  // Clear the flag first, so that the driver cannot take it up while the
  // debugger API is finalized.
  if (Error err =
          CUDADebuggerAPI::ClearIpcFlag(get_symbol_address, native_process))
    LLDB_LOG_ERROR(log, std::move(err), "Failed to clear the IPC flag: {0}");
  ReleaseDebuggerAPI();
  return DisablingResponse(GetNewGPUAction());
}

GPUActions LLDBServerPluginNVGPU::GetInitializeActions() {
  GPUActions init_actions = GetNewGPUAction();

  // The initialization breakpoints bring the debugger API up in the call to
  // cuInit that initializes the driver. When lldb attaches to an application
  // that has initialized CUDA already, the driver's attach procedure brings it
  // up instead, which SetRequestedSymbols starts once lldb has these symbols
  // of libcuda: as soon as it loads the libraries of a process it attaches to,
  // or when the process loads libcuda. See the class comment.
  for (const char *library_name : {CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME,
                                   CUDADebuggerAPI::LIBCUDA_LIBRARY_NAME_ALT}) {
    init_actions.breakpoints.push_back(
        CUDADebuggerAPI::GetCuInitBreakpointInfo(library_name));
    init_actions.breakpoints.push_back(
        CUDADebuggerAPI::GetPreInitBreakpointInfo(library_name));
  }
  for (StringLiteral name : CUDADebuggerAPI::GetLateAttachSymbolNames())
    init_actions.symbol_names.push_back(name.str());
  return init_actions;
}

void LLDBServerPluginNVGPU::OnDebuggerAPIEvent() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGV(log, "LLDBServerPluginNVGPU::OnDebuggerAPIEvent");
  // The application has exited, and acknowledging its events can crash the
  // debugger library.
  if (m_native_process_exited)
    return;
  if (Expected<int> handled = DrainSyncEvents(); !handled) {
    ReportDebuggerAPIFault(llvm::toString(handled.takeError()));
    return;
  }
  if (Error err = StopIfAttachFinished())
    ReportDebuggerAPIFault(llvm::toString(std::move(err)));
}

Error LLDBServerPluginNVGPU::StopIfAttachFinished() {
  if (m_attach_state != AttachState::Finished &&
      m_attach_state != AttachState::Unreported)
    return Error::success();
  ProcessNVGPU *gpu = GetGPUProcess();
  if (!gpu || gpu->HasDebuggerAPIFaulted() ||
      !StateIsRunningState(gpu->GetState()))
    return Error::success();
  if (std::exchange(m_attach_state, AttachState::None) ==
      AttachState::Unreported)
    return createStringError(
        "The CUDA driver attached without reporting the application's CUDA "
        "contexts and modules, as it does when the application loads modules "
        "lazily. To debug its GPU, start the application with "
        "CUDA_MODULE_LOADING=EAGER and "
        "CUDA_DISABLE_FUNCTION_LAZY_FINALIZATION=1 before attaching");
  return gpu->StopAfterAttach();
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
      if (*stop == ProcessNVGPU::SuspendStop::Stopped) {
        // This stop shows the threads that the stop at the end of a late
        // attach would.
        m_attach_state = AttachState::None;
        // An interrupt can be lldb sending a packet, after which it resumes
        // the GPU at once, so the CPU keeps running.
        if (CPUStopsWithGPU())
          HaltNativeProcessIfNeeded();
      }
      break;
    }
    case CUDBG_EVENT_ATTACH_COMPLETE: {
      if (m_attach_state == AttachState::Replaying)
        m_attach_state = AttachState::Finished;
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
