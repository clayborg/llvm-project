//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDADebuggerAPI.h"
#include "../Utils/Utils.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

#define STRINGIFY_SYMBOL_HELPER(x) #x
#define STRINGIFY_SYMBOL(x) STRINGIFY_SYMBOL_HELPER(x)

// The driver can briefly report CUDBG_ERROR_ATTACH_NOT_POSSIBLE right after
// injection, so initialize() backs off (doubling, capped) rather than failing.
static constexpr unsigned kInitRetryDelayMs = 100;
static constexpr unsigned kInitMaxRetryDelayMs = 1000;
static constexpr unsigned kInitTimeoutMs = 5000;

// The debugger capabilities we request, which Initialize then requires the
// debug engine to support.
static constexpr uint32_t kRequestedCapabilities =
    CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS |
    CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

// Assumed size of the driver's fixed CUDBG_INJECTION_PATH buffer, including the
// trailing NUL. Longer paths are rejected rather than truncated.
static constexpr size_t kInjectionPathMaxSize = 4096;

namespace Symbols {
static std::string CUDBG_IPC_FLAG_NAME = STRINGIFY_SYMBOL(CUDBG_IPC_FLAG_NAME);
static std::string CUDBG_APICLIENT_PID = STRINGIFY_SYMBOL(CUDBG_APICLIENT_PID);
static std::string CUDBG_APICLIENT_REVISION =
    STRINGIFY_SYMBOL(CUDBG_APICLIENT_REVISION);
static std::string CUDBG_SESSION_ID = STRINGIFY_SYMBOL(CUDBG_SESSION_ID);
static std::string CUDBG_DEBUGGER_CAPABILITIES =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_CAPABILITIES);
static std::string CUDBG_INJECTION_PATH = "cudbgInjectionPath";
static std::string CUDBG_GET_API = "cudbgGetAPI";
static std::string CUDBG_GET_API_VERSION = "cudbgGetAPIVersion";
static std::string CUDA_INITIALIZATION_SYMBOL =
    CMAKE_NVGPU_INITIALIZATION_SYMBOL;
// Late-attach handshake symbols exported by the driver into the running
// process. See the "Attaching and Detaching" section of cudadebugger.h.
static std::string CUDBG_RESUME_FOR_ATTACH_DETACH =
    STRINGIFY_SYMBOL(CUDBG_RESUME_FOR_ATTACH_DETACH);
static std::string CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED =
    STRINGIFY_SYMBOL(CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED);
} // namespace Symbols

/// Read a uint32_t global from the running process.
static Expected<uint32_t> ReadUInt32FromHost(SymbolAddressProvider get_addr,
                                             HostServer &host_server,
                                             llvm::StringRef symbol_name) {
  std::optional<uint64_t> symbol_address = get_addr(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                symbol_name);
  uint32_t value = 0;
  if (Error err =
          host_server.ReadProcessMemory(*symbol_address, &value, sizeof(value)))
    return createStringErrorFmt("Failed to read symbol {0}: {1}", symbol_name,
                                llvm::toString(std::move(err)));
  return value;
}

namespace lldb_private::lldb_server {
void CUDBGAPIDeleter(CUDBGAPI api) {
  if (!api)
    return;

  CUDBGResult res = api->finalize();
  if (res != CUDBG_SUCCESS) {
    Log *log = GetLog(GDBRLog::Plugin);
    LLDB_LOG(log, "Failed to finalize the CUDA Debugger API. {}",
             cudbgGetErrorString(res));
  }
}
} // namespace lldb_private::lldb_server

static Error VerifyDebuggerCapabilities(CUDADebuggerAPI &api) {
  CUDBGCapabilityFlags supported_capabilities;
  CUDBGResult res =
      api->getSupportedDebuggerCapabilities(&supported_capabilities);
  if (res != CUDBG_SUCCESS)
    return createStringError(
        "Failed to get the GPU debugger supported capabilities. {}",
        cudbgGetErrorString(res));
  if (!(supported_capabilities & CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS))
    return createStringError(
        "The GPU debugger does not support suspend events");
  if (!(supported_capabilities &
        CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS))
    return createStringError(
        "The GPU debugger does not support skipping context "
        "push/pop events");
  return Error::success();
}

template <typename T>
static Error WriteToHostSymbol(SymbolAddressProvider get_addr,
                               HostServer &host_server,
                               llvm::StringRef symbol_name, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "WriteToHostSymbol only writes fixed-size trivially-copyable "
                "values; use WriteInjectionPathToInferior for strings");
  std::optional<uint64_t> symbol_address = get_addr(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {}",
                                symbol_name);

  if (Error err = host_server.WriteProcessMemory(*symbol_address, &value,
                                                 sizeof(value)))
    return createStringErrorFmt("Failed to write symbol {}: {}", symbol_name,
                                llvm::toString(std::move(err)));
  return Error::success();
}

/// Write the driver's CUDBG_INJECTION_PATH global into the inferior, rejecting
/// anything that would not fit its fixed-size buffer.
static Error WriteInjectionPathToInferior(SymbolAddressProvider get_addr,
                                          HostServer &host_server,
                                          llvm::StringRef path) {
  if (path.size() + 1 > kInjectionPathMaxSize)
    return createStringErrorFmt(
        "CUDBG_INJECTION_PATH is too long ({0} bytes); the driver's injection "
        "path buffer holds at most {1} bytes including the NUL terminator",
        path.size(), kInjectionPathMaxSize);

  std::optional<uint64_t> symbol_address =
      get_addr(Symbols::CUDBG_INJECTION_PATH);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                Symbols::CUDBG_INJECTION_PATH);

  // A StringRef is not guaranteed to be NUL-terminated, so terminate a copy.
  llvm::SmallVector<char, 256> buffer(path.begin(), path.end());
  buffer.push_back('\0');
  if (Error err = host_server.WriteProcessMemory(*symbol_address, buffer.data(),
                                                 buffer.size()))
    return createStringErrorFmt("Failed to write symbol {0}: {1}",
                                Symbols::CUDBG_INJECTION_PATH,
                                llvm::toString(std::move(err)));
  return Error::success();
}

/// Publish the client identity and requested capabilities into the inferior,
/// and optionally the IPC flag -- which must come last, and which the attach
/// path defers entirely (see CUDADebuggerAPI::SetIpcFlag).
static Error WriteInitializationSymbolsToHost(SymbolAddressProvider get_addr,
                                              HostServer &host_server,
                                              uint32_t pid, uint32_t session_id,
                                              uint32_t revision,
                                              bool set_ipc_flag) {
  auto write_uint32_t = [&](const std::string &symbol_name,
                            const uint32_t &value) -> Error {
    return WriteToHostSymbol(get_addr, host_server, symbol_name, value);
  };

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_PID, pid))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_REVISION, revision))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_SESSION_ID, session_id))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_DEBUGGER_CAPABILITIES,
                                 kRequestedCapabilities))
    return err;

  if (!set_ipc_flag)
    return Error::success();

  return CUDADebuggerAPI::SetIpcFlag(get_addr, host_server);
}

static Error WriteConfigurationToLibcuda(llvm::sys::DynamicLibrary &libcuda,
                                         uint32_t pid, uint32_t revision,
                                         uint32_t session_id,
                                         StringRef libcuda_library_name) {
  auto *api_client_pid = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_APICLIENT_PID.c_str()));
  if (!api_client_pid)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_APICLIENT_PID,
                                libcuda_library_name);

  auto *api_client_revision = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_APICLIENT_REVISION.c_str()));
  if (!api_client_revision)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_APICLIENT_REVISION,
                                libcuda_library_name);

  auto *session_id_ptr = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_SESSION_ID.c_str()));
  if (!session_id_ptr)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_SESSION_ID,
                                libcuda_library_name);

  *api_client_pid = pid;
  *api_client_revision = revision;
  *session_id_ptr = session_id;

  return Error::success();
}

static Error WriteInjectionPathToLibcuda(SymbolAddressProvider get_addr,
                                         HostServer &host_server,
                                         llvm::sys::DynamicLibrary &libcuda,
                                         StringRef libcuda_library_name) {
  const char *path = getenv("CUDBG_INJECTION_PATH");
  if (!path)
    return Error::success();

  // This also rejects a path too long for the driver's buffer, which the local
  // copy below relies on.
  llvm::StringRef path_ref(path);
  if (Error err = WriteInjectionPathToInferior(get_addr, host_server, path_ref))
    return err;

  // Also update this server's locally loaded libcuda image so the API table it
  // returns is consistent.
  char *injection_path = reinterpret_cast<char *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_INJECTION_PATH.c_str()));
  if (!injection_path)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_INJECTION_PATH,
                                libcuda_library_name);
  std::memcpy(injection_path, path_ref.data(), path_ref.size());
  injection_path[path_ref.size()] = '\0';
  return Error::success();
}

// Query the CUDA debugger API version that the live driver supports. This is
// the basis for choosing a mutually-supported revision so the debugger can
// attach to any driver within its compiled major release.
static Expected<nvgpu::CudbgApiVersion>
GetDriverAPIVersion(llvm::sys::DynamicLibrary &libcuda,
                    StringRef libcuda_library_name) {
  using CudbgGetAPIVersionFn =
      CUDBGResult (*)(uint32_t *, uint32_t *, uint32_t *);
  const auto cudbgGetAPIVersion = reinterpret_cast<CudbgGetAPIVersionFn>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_GET_API_VERSION.c_str()));
  if (!cudbgGetAPIVersion)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_GET_API_VERSION,
                                libcuda_library_name);

  nvgpu::CudbgApiVersion version;
  CUDBGResult res =
      cudbgGetAPIVersion(&version.major, &version.minor, &version.revision);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("The `cudbgGetAPIVersion` call failed. {}",
                                cudbgGetErrorString(res));

  return version;
}

static Expected<CUDBGAPI>
GetRawAPIInstance(llvm::sys::DynamicLibrary &libcuda,
                  StringRef libcuda_library_name,
                  const nvgpu::CudbgApiVersion &version) {
  using CudbgGetAPIFn =
      CUDBGResult (*)(uint32_t, uint32_t, uint32_t, CUDBGAPI *);
  const CudbgGetAPIFn cudbgGetAPI = reinterpret_cast<CudbgGetAPIFn>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_GET_API.c_str()));
  if (!cudbgGetAPI)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_GET_API, libcuda_library_name);

  // Request the chosen version (the lesser of the compiled and driver
  // versions). The driver returns an API table matching the requested
  // version's ABI.
  CUDBGAPI api;
  CUDBGResult res =
      cudbgGetAPI(version.major, version.minor, version.revision, &api);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("The `cudbgGetAPI` call failed. {}",
                                cudbgGetErrorString(res));

  return api;
}

Expected<CUDADebuggerAPI> CUDADebuggerAPI::InitializeImpl(
    SymbolAddressProvider get_symbol_address, StringRef libcuda_library_name,
    HostServer &host_server, InitContext init_context) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "CUDADebuggerAPI::Initialize()");

  const uint32_t pid = llvm::sys::Process::getProcessId();
  const uint32_t session_id = 0;

  std::string load_error;
  llvm::sys::DynamicLibrary libcuda =
      llvm::sys::DynamicLibrary::getPermanentLibrary(
          libcuda_library_name.str().c_str(), &load_error);
  if (!libcuda.isValid())
    return createStringErrorFmt("Failed to load {}: {}", libcuda_library_name,
                                load_error);

  // Discover the driver's supported API version and choose the version to
  // use. We support any driver within our compiled major release; using the
  // lesser of the compiled and driver versions lets an older in-major driver
  // still attach (newer-than-driver features are simply unavailable).
  Expected<nvgpu::CudbgApiVersion> driver_version_or =
      GetDriverAPIVersion(libcuda, libcuda_library_name);
  if (!driver_version_or)
    return driver_version_or.takeError();

  const nvgpu::CudbgApiVersion driver_version = *driver_version_or;
  const nvgpu::CudbgApiVersion compiled_version =
      nvgpu::CudbgApiVersion::Compiled();

  if (driver_version.major != compiled_version.major)
    return createStringErrorFmt(
        "The CUDA driver debugger API major version ({0}) does not match the "
        "version this lldb-server was built against ({1}). Cross-major-release "
        "GPU debugging is not supported; use an lldb-server built against CUDA "
        "{0}.x.",
        driver_version.major, compiled_version.major);

  const nvgpu::CudbgApiVersion api_version =
      driver_version < compiled_version ? driver_version : compiled_version;
  const uint32_t revision = api_version.revision;

  LLDB_LOG(log,
           "CUDADebuggerAPI: compiled {0}.{1}.{2}, driver {3}.{4}.{5}, "
           "using {6}.{7}.{8}",
           compiled_version.major, compiled_version.minor,
           compiled_version.revision, driver_version.major,
           driver_version.minor, driver_version.revision, api_version.major,
           api_version.minor, api_version.revision);

  if (Error err = WriteInitializationSymbolsToHost(
          get_symbol_address, host_server, pid, session_id, revision,
          /*set_ipc_flag=*/init_context == InitContext::eLaunch))
    return err;

  if (Error err = WriteConfigurationToLibcuda(libcuda, pid, revision,
                                              session_id, libcuda_library_name))
    return err;

  if (Error err = WriteInjectionPathToLibcuda(get_symbol_address, host_server,
                                              libcuda, libcuda_library_name))
    return err;

  Expected<CUDBGAPI> api_or =
      GetRawAPIInstance(libcuda, libcuda_library_name, api_version);
  if (!api_or)
    return api_or.takeError();

  CUDADebuggerAPI api(*api_or, api_version);

  // Only CUDBG_ERROR_ATTACH_NOT_POSSIBLE is retried; anything else is terminal.
  using std::chrono::milliseconds;
  using std::chrono::steady_clock;
  const steady_clock::time_point deadline =
      steady_clock::now() + milliseconds(kInitTimeoutMs);
  unsigned backoff_ms = kInitRetryDelayMs;
  while (true) {
    CUDBGResult res = api->initialize();
    if (res == CUDBG_SUCCESS)
      break;
    if (res != CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
      return createStringErrorFmt("The `CUDBGAPI.initialize` call failed. {}",
                                  cudbgGetErrorString(res));
    steady_clock::time_point now = steady_clock::now();
    if (now >= deadline)
      return createStringErrorFmt(
          "The `CUDBGAPI.initialize` call did not become possible within "
          "{0}ms (CUDBG_ERROR_ATTACH_NOT_POSSIBLE). {1}",
          kInitTimeoutMs, cudbgGetErrorString(res));
    // Sleep the backoff, but never past the overall deadline, then grow it
    // (capped) for the next attempt.
    milliseconds remaining =
        std::chrono::duration_cast<milliseconds>(deadline - now);
    std::this_thread::sleep_for(std::min(milliseconds(backoff_ms), remaining));
    backoff_ms = std::min(backoff_ms * 2, kInitMaxRetryDelayMs);
  }

  if (Error err = VerifyDebuggerCapabilities(api))
    return err;

  return api;
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::Initialize(SymbolAddressProvider get_symbol_address,
                            StringRef libcuda_library_name,
                            HostServer &host_server, InitContext init_context) {
  Expected<CUDADebuggerAPI> api = InitializeImpl(
      get_symbol_address, libcuda_library_name, host_server, init_context);
  if (!api)
    return createStringErrorFmt(
        "Failed to initialize the CUDA Debugger API. {}",
        llvm::toString(api.takeError()));
  return api;
}

nvgpu::AttachHandshake CUDADebuggerAPI::GetAttachHandshake() {
  nvgpu::AttachHandshake handshake;
  handshake.api_client_pid = llvm::sys::Process::getProcessId();
  // The version cannot be negotiated before the debug engine is up, so announce
  // the compiled one. Initialize writes the negotiated one at the breakpoint
  // that ends the attach.
  handshake.api_client_revision = nvgpu::CudbgApiVersion::Compiled().revision;
  handshake.session_id = 0;
  handshake.capabilities = kRequestedCapabilities;
  handshake.injection_path = llvm::sys::Process::GetEnv("CUDBG_INJECTION_PATH");
  return handshake;
}

Error CUDADebuggerAPI::SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                  HostServer &host_server) {
  const uint32_t enabled = 1;
  return WriteToHostSymbol(get_symbol_address, host_server,
                           Symbols::CUDBG_IPC_FLAG_NAME, enabled);
}

Expected<uint32_t> CUDADebuggerAPI::ReadResumeForAttachDetach(
    SymbolAddressProvider get_symbol_address, HostServer &host_server) {
  return ReadUInt32FromHost(get_symbol_address, host_server,
                            Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH);
}

GPUBreakpointInfo
CUDADebuggerAPI::GetInitializationBreakpointInfo(StringRef library_name) {
  GPUBreakpointInfo bp;
  bp.name_info = {library_name.str(), Symbols::CUDA_INITIALIZATION_SYMBOL};
  bp.symbol_names.push_back(Symbols::CUDBG_IPC_FLAG_NAME);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_PID);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_REVISION);
  bp.symbol_names.push_back(Symbols::CUDBG_SESSION_ID);
  bp.symbol_names.push_back(Symbols::CUDBG_DEBUGGER_CAPABILITIES);
  bp.symbol_names.push_back(Symbols::CUDBG_INJECTION_PATH);
  // Only needed at detach, which has no breakpoint of its own to carry it.
  bp.symbol_names.push_back(Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH);
  return bp;
}

bool CUDADebuggerAPI::IsAttachFinishedBreakpoint(StringRef function_name) {
  return function_name == Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED;
}

GPUBreakpointInfo
CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(StringRef library_name) {
  GPUBreakpointInfo bp = GetInitializationBreakpointInfo(library_name);
  bp.name_info->function_name = Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED;
  return bp;
}
