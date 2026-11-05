//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDADebuggerAPI.h"
#include "../Utils/Utils.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

#define STRINGIFY_SYMBOL_HELPER(x) #x
#define STRINGIFY_SYMBOL(x) STRINGIFY_SYMBOL_HELPER(x)

// The driver can report CUDBG_ERROR_ATTACH_NOT_POSSIBLE for a while after it
// injects its debug engine, so initialize() is retried, with the delay between
// attempts doubling up to a cap.
static constexpr std::chrono::milliseconds g_init_retry_delay(100);
static constexpr std::chrono::milliseconds g_init_max_retry_delay(1000);
static constexpr std::chrono::milliseconds g_init_timeout(5000);

// The debugger capabilities to request, which VerifyDebuggerCapabilities then
// requires the debug engine to support.
static constexpr uint32_t g_requested_capabilities =
    CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS |
    CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

// The size of the driver's CUDBG_INJECTION_PATH buffer, including the
// terminating NUL. A longer path is rejected rather than truncated.
static constexpr size_t g_injection_path_buffer_size = 4096;

// Written to the driver's attach procedure descriptor to ask it to attach. The
// driver accepts any byte.
static constexpr uint8_t g_attach_request_byte = 0xAB;

namespace Symbols {
static constexpr StringLiteral CUDBG_IPC_FLAG_NAME =
    STRINGIFY_SYMBOL(CUDBG_IPC_FLAG_NAME);
static constexpr StringLiteral CUDBG_APICLIENT_PID =
    STRINGIFY_SYMBOL(CUDBG_APICLIENT_PID);
static constexpr StringLiteral CUDBG_APICLIENT_REVISION =
    STRINGIFY_SYMBOL(CUDBG_APICLIENT_REVISION);
static constexpr StringLiteral CUDBG_SESSION_ID =
    STRINGIFY_SYMBOL(CUDBG_SESSION_ID);
static constexpr StringLiteral CUDBG_DEBUGGER_CAPABILITIES =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_CAPABILITIES);
static constexpr StringLiteral CUDBG_INJECTION_PATH = "cudbgInjectionPath";
static constexpr StringLiteral CUDBG_GET_API = "cudbgGetAPI";
static constexpr StringLiteral CUDBG_GET_API_VERSION = "cudbgGetAPIVersion";
// The driver's initialization. Every call to cuInit calls CUDBG_PRE_INIT, if
// libcuda has it, before the driver initializes, but only the first call
// initializes it, and the debug engine sets CUDBG_DEBUGGER_INITIALIZED if that
// initialization brought it up.
static constexpr StringLiteral CU_INIT = "cuInit";
static constexpr StringLiteral CUDBG_PRE_INIT =
    STRINGIFY_SYMBOL(CUDBG_PRE_INIT);
static constexpr StringLiteral CUDBG_DEBUGGER_INITIALIZED =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_INITIALIZED);
// The driver's attach procedure. A write to the descriptor in
// CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD asks the driver to attach, and
// it calls CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED once it has.
static constexpr StringLiteral CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD =
    STRINGIFY_SYMBOL(CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD);
static constexpr StringLiteral CUDBG_RESUME_FOR_ATTACH_DETACH =
    STRINGIFY_SYMBOL(CUDBG_RESUME_FOR_ATTACH_DETACH);
static constexpr StringLiteral CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED =
    STRINGIFY_SYMBOL(CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED);
} // namespace Symbols

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
    return createStringErrorFmt(
        "Failed to get the GPU debugger supported capabilities. {0}",
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

/// Write exactly \a size bytes at \a addr in the debugged process.
static Error WriteToHost(NativeProcessProtocol &process, lldb::addr_t addr,
                         const void *buf, size_t size) {
  size_t bytes_written = 0;
  Status status = process.WriteMemory(addr, buf, size, bytes_written);
  if (status.Fail())
    return status.ToError();
  if (bytes_written != size)
    return createStringErrorFmt("Wrote {0} of {1} bytes at {2:x}",
                                bytes_written, size, addr);
  return Error::success();
}

template <typename T>
static Error WriteToHostSymbol(SymbolAddressProvider get_symbol_address,
                               NativeProcessProtocol &process,
                               StringRef symbol_name, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "only fixed-size values can be written this way; use "
                "WriteInjectionPathToHost for the injection path");
  std::optional<uint64_t> symbol_address = get_symbol_address(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {}",
                                symbol_name);

  if (Error err = WriteToHost(process, *symbol_address, &value, sizeof(value)))
    return createStringErrorFmt("Failed to write symbol {0}: {1}", symbol_name,
                                llvm::toString(std::move(err)));
  return Error::success();
}

template <typename T>
static Expected<T> ReadFromHostSymbol(SymbolAddressProvider get_symbol_address,
                                      NativeProcessProtocol &process,
                                      StringRef symbol_name) {
  static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(uint64_t),
                "only integers can be read this way");
  std::optional<uint64_t> symbol_address = get_symbol_address(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                symbol_name);

  uint8_t bytes[sizeof(T)];
  size_t bytes_read = 0;
  Status status =
      process.ReadMemory(*symbol_address, bytes, sizeof(bytes), bytes_read);
  if (status.Fail())
    return createStringErrorFmt("Failed to read symbol {0}: {1}", symbol_name,
                                status.AsCString());
  if (bytes_read != sizeof(bytes))
    return createStringErrorFmt("Read {0} of {1} bytes of symbol {2}",
                                bytes_read, sizeof(bytes), symbol_name);
  lldb_private::DataExtractor data(bytes, sizeof(bytes), process.GetByteOrder(),
                                   process.GetAddressByteSize());
  lldb::offset_t offset = 0;
  if constexpr (std::is_signed_v<T>)
    return static_cast<T>(data.GetMaxS64(&offset, sizeof(bytes)));
  return static_cast<T>(data.GetMaxU64(&offset, sizeof(bytes)));
}

/// The path in the CUDBG_INJECTION_PATH environment variable, if it is set,
/// from which the driver loads its debug engine.
static std::optional<std::string> GetInjectionPath() {
  return sys::Process::GetEnv("CUDBG_INJECTION_PATH");
}

/// Reject \a path, from GetInjectionPath, if the driver's buffer cannot hold it
/// with its terminating NUL.
static Error CheckInjectionPathFits(StringRef path) {
  if (path.size() < g_injection_path_buffer_size)
    return Error::success();
  return createStringErrorFmt(
      "CUDBG_INJECTION_PATH is {0} bytes long, but the CUDA driver holds at "
      "most {1}",
      path.size(), g_injection_path_buffer_size - 1);
}

/// Write \a path into the driver's CUDBG_INJECTION_PATH buffer in the debugged
/// process, rejecting a path that does not fit.
static Error WriteInjectionPathToHost(SymbolAddressProvider get_symbol_address,
                                      NativeProcessProtocol &process,
                                      StringRef path) {
  if (Error err = CheckInjectionPathFits(path))
    return err;

  std::optional<uint64_t> symbol_address =
      get_symbol_address(Symbols::CUDBG_INJECTION_PATH);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                Symbols::CUDBG_INJECTION_PATH);

  // A StringRef need not be NUL-terminated.
  SmallVector<char, 256> buffer(path.begin(), path.end());
  buffer.push_back('\0');
  if (Error err =
          WriteToHost(process, *symbol_address, buffer.data(), buffer.size()))
    return createStringErrorFmt("Failed to write symbol {0}: {1}",
                                Symbols::CUDBG_INJECTION_PATH,
                                llvm::toString(std::move(err)));
  return Error::success();
}

/// Write the API client's identity and requested capabilities into the
/// debugged process. The IPC flag is not written here: see
/// CUDADebuggerAPI::SetIpcFlag.
static Error
WriteInitializationSymbolsToHost(SymbolAddressProvider get_symbol_address,
                                 NativeProcessProtocol &process, uint32_t pid,
                                 uint32_t session_id, uint32_t revision) {
  auto write_uint32_t = [&](StringRef symbol_name,
                            const uint32_t &value) -> Error {
    return WriteToHostSymbol(get_symbol_address, process, symbol_name, value);
  };

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_PID, pid))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_REVISION, revision))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_SESSION_ID, session_id))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_DEBUGGER_CAPABILITIES,
                                 g_requested_capabilities))
    return err;

  return Error::success();
}

static Error WriteConfigurationToLibcuda(llvm::sys::DynamicLibrary &libcuda,
                                         uint32_t pid, uint32_t revision,
                                         uint32_t session_id,
                                         StringRef libcuda_library_name) {
  auto *api_client_pid = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_APICLIENT_PID.data()));
  if (!api_client_pid)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_APICLIENT_PID,
                                libcuda_library_name);

  auto *api_client_revision = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_APICLIENT_REVISION.data()));
  if (!api_client_revision)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_APICLIENT_REVISION,
                                libcuda_library_name);

  auto *session_id_ptr = reinterpret_cast<uint32_t *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_SESSION_ID.data()));
  if (!session_id_ptr)
    return createStringErrorFmt("Failed to find symbol {} in {}",
                                Symbols::CUDBG_SESSION_ID,
                                libcuda_library_name);

  *api_client_pid = pid;
  *api_client_revision = revision;
  *session_id_ptr = session_id;

  return Error::success();
}

static Error WriteInjectionPathToLibcuda(
    SymbolAddressProvider get_symbol_address, NativeProcessProtocol &process,
    llvm::sys::DynamicLibrary &libcuda, StringRef libcuda_library_name) {
  std::optional<std::string> path = GetInjectionPath();
  if (!path)
    return Error::success();

  if (Error err = WriteInjectionPathToHost(get_symbol_address, process, *path))
    return err;

  char *injection_path = reinterpret_cast<char *>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_INJECTION_PATH.data()));
  if (!injection_path)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_INJECTION_PATH,
                                libcuda_library_name);
  if (Error err = CheckInjectionPathFits(*path))
    return err;
  std::memcpy(injection_path, path->c_str(), path->size() + 1);
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
      libcuda.getAddressOfSymbol(Symbols::CUDBG_GET_API_VERSION.data()));
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

/// Load \a libcuda_library_name into lldb-server, which reaches the debugger
/// API through its own copy of libcuda.
static Expected<llvm::sys::DynamicLibrary>
LoadLibcuda(StringRef libcuda_library_name) {
  std::string load_error;
  llvm::sys::DynamicLibrary libcuda =
      llvm::sys::DynamicLibrary::getPermanentLibrary(
          libcuda_library_name.str().c_str(), &load_error);
  if (!libcuda.isValid())
    return createStringErrorFmt("Failed to load {0}: {1}", libcuda_library_name,
                                load_error);
  return libcuda;
}

/// Choose the debugger API version to use with the driver. We support any
/// driver within our compiled major release; using the lesser of the compiled
/// and driver versions lets an older in-major driver still attach
/// (newer-than-driver features are simply unavailable).
static Expected<nvgpu::CudbgApiVersion>
NegotiateAPIVersion(llvm::sys::DynamicLibrary &libcuda,
                    StringRef libcuda_library_name) {
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
  LLDB_LOG(GetLog(GDBRLog::Plugin),
           "CUDADebuggerAPI: compiled {0}.{1}.{2}, driver {3}.{4}.{5}, "
           "using {6}.{7}.{8}",
           compiled_version.major, compiled_version.minor,
           compiled_version.revision, driver_version.major,
           driver_version.minor, driver_version.revision, api_version.major,
           api_version.minor, api_version.revision);
  return api_version;
}

static Expected<CUDBGAPI>
GetRawAPIInstance(llvm::sys::DynamicLibrary &libcuda,
                  StringRef libcuda_library_name,
                  const nvgpu::CudbgApiVersion &version) {
  using CudbgGetAPIFn =
      CUDBGResult (*)(uint32_t, uint32_t, uint32_t, CUDBGAPI *);
  const CudbgGetAPIFn cudbgGetAPI = reinterpret_cast<CudbgGetAPIFn>(
      libcuda.getAddressOfSymbol(Symbols::CUDBG_GET_API.data()));
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

/// Call initialize(), retrying while the driver reports
/// CUDBG_ERROR_ATTACH_NOT_POSSIBLE, until g_init_timeout has passed.
static Error InitializeWithRetries(CUDADebuggerAPI &api) {
  using std::chrono::steady_clock;
  const steady_clock::time_point deadline =
      steady_clock::now() + g_init_timeout;
  std::chrono::milliseconds delay = g_init_retry_delay;
  while (true) {
    CUDBGResult res = api->initialize();
    if (res == CUDBG_SUCCESS)
      return Error::success();
    if (res != CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
      return createStringErrorFmt("The `CUDBGAPI.initialize` call failed. {0}",
                                  cudbgGetErrorString(res));
    steady_clock::time_point now = steady_clock::now();
    if (now >= deadline)
      return createStringErrorFmt(
          "The `CUDBGAPI.initialize` call still failed after {0} ms. {1}",
          g_init_timeout.count(), cudbgGetErrorString(res));
    std::this_thread::sleep_for(
        std::min<steady_clock::duration>(delay, deadline - now));
    delay = std::min(delay * 2, g_init_max_retry_delay);
  }
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::InitializeImpl(SymbolAddressProvider get_symbol_address,
                                StringRef libcuda_library_name,
                                NativeProcessProtocol &process) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "CUDADebuggerAPI::Initialize()");

  const uint32_t pid = getpid();
  const uint32_t session_id = 0;

  Expected<llvm::sys::DynamicLibrary> libcuda =
      LoadLibcuda(libcuda_library_name);
  if (!libcuda)
    return libcuda.takeError();

  Expected<nvgpu::CudbgApiVersion> api_version =
      NegotiateAPIVersion(*libcuda, libcuda_library_name);
  if (!api_version)
    return api_version.takeError();
  const uint32_t revision = api_version->revision;

  if (Error err = WriteInitializationSymbolsToHost(get_symbol_address, process,
                                                   pid, session_id, revision))
    return err;

  if (Error err = WriteConfigurationToLibcuda(*libcuda, pid, revision,
                                              session_id, libcuda_library_name))
    return err;

  if (Error err = WriteInjectionPathToLibcuda(get_symbol_address, process,
                                              *libcuda, libcuda_library_name))
    return err;

  Expected<CUDBGAPI> api_or =
      GetRawAPIInstance(*libcuda, libcuda_library_name, *api_version);
  if (!api_or)
    return api_or.takeError();

  CUDADebuggerAPI api(*api_or, *api_version);

  if (Error err = InitializeWithRetries(api))
    return err;

  if (Error err = VerifyDebuggerCapabilities(api))
    return err;

  return api;
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::Initialize(SymbolAddressProvider get_symbol_address,
                            StringRef libcuda_library_name,
                            NativeProcessProtocol &process) {
  Expected<CUDADebuggerAPI> api =
      InitializeImpl(get_symbol_address, libcuda_library_name, process);
  if (!api)
    return createStringErrorFmt(
        "Failed to initialize the CUDA Debugger API. {}",
        llvm::toString(api.takeError()));
  return api;
}

Error CUDADebuggerAPI::SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                  NativeProcessProtocol &process) {
  const uint32_t ipc_flag = 1;
  return WriteToHostSymbol(get_symbol_address, process,
                           Symbols::CUDBG_IPC_FLAG_NAME, ipc_flag);
}

Error CUDADebuggerAPI::ClearIpcFlag(SymbolAddressProvider get_symbol_address,
                                    NativeProcessProtocol &process) {
  const uint32_t ipc_flag = 0;
  return WriteToHostSymbol(get_symbol_address, process,
                           Symbols::CUDBG_IPC_FLAG_NAME, ipc_flag);
}

Expected<bool> CUDADebuggerAPI::ReadDebuggerInitialized(
    SymbolAddressProvider get_symbol_address, NativeProcessProtocol &process) {
  Expected<uint32_t> initialized = ReadFromHostSymbol<uint32_t>(
      get_symbol_address, process, Symbols::CUDBG_DEBUGGER_INITIALIZED);
  if (!initialized)
    return initialized.takeError();
  return *initialized != 0;
}

ArrayRef<StringLiteral> CUDADebuggerAPI::GetLateAttachSymbolNames() {
  static constexpr StringLiteral names[] = {
      Symbols::CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD,
      Symbols::CUDBG_APICLIENT_PID,
      Symbols::CUDBG_APICLIENT_REVISION,
      Symbols::CUDBG_SESSION_ID,
      Symbols::CUDBG_DEBUGGER_CAPABILITIES,
      Symbols::CUDBG_INJECTION_PATH,
  };
  return names;
}

Expected<bool>
CUDADebuggerAPI::RequestLateAttach(SymbolAddressProvider get_symbol_address,
                                   NativeProcessProtocol &process) {
  std::optional<std::string> injection_path = GetInjectionPath();
  // Check every symbol first, so that a missing one writes nothing.
  for (StringLiteral name : GetLateAttachSymbolNames())
    if (!get_symbol_address(name) &&
        (name != Symbols::CUDBG_INJECTION_PATH || injection_path))
      return createStringErrorFmt(
          "{0} was not found in libcuda; the CUDA driver may be too old to "
          "support attaching to a running application",
          name);

  // The driver publishes the descriptor once the application has a CUDA
  // context. Until then it is -1.
  Expected<int32_t> attach_fd = ReadFromHostSymbol<int32_t>(
      get_symbol_address, process,
      Symbols::CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD);
  if (!attach_fd)
    return attach_fd.takeError();
  if (*attach_fd < 0)
    return false;

  // Tell the driver who is attaching before asking it to, with the API
  // revision that Initialize negotiates again at the breakpoint that ends the
  // attach. The IPC flag stays clear until then.
  Expected<llvm::sys::DynamicLibrary> libcuda =
      LoadLibcuda(LIBCUDA_LIBRARY_NAME_ALT);
  if (!libcuda)
    return libcuda.takeError();
  Expected<nvgpu::CudbgApiVersion> api_version =
      NegotiateAPIVersion(*libcuda, LIBCUDA_LIBRARY_NAME_ALT);
  if (!api_version)
    return api_version.takeError();
  if (Error err = WriteInitializationSymbolsToHost(get_symbol_address, process,
                                                   getpid(), /*session_id=*/0,
                                                   api_version->revision))
    return std::move(err);
  // The driver reads the path when it injects the debug engine.
  if (injection_path)
    if (Error err = WriteInjectionPathToHost(get_symbol_address, process,
                                             *injection_path))
      return std::move(err);

  // The descriptor is the write end of a pipe that the driver reads, in the
  // debugged process's descriptor table, which procfs exposes. Opened without
  // blocking, a pipe that has lost its reader or is full fails the request
  // instead of hanging lldb-server. The driver attaches once the process runs.
  std::string fd_path =
      formatv("/proc/{0}/fd/{1}", process.GetID(), *attach_fd).str();
  int fd = sys::RetryAfterSignal(-1, ::open, fd_path.c_str(),
                                 O_WRONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return createStringErrorFmt(
        "Failed to open {0}: {1}", fd_path,
        llvm::toString(errorCodeToError(errnoAsErrorCode())));
  auto close_fd = make_scope_exit([fd] { ::close(fd); });
  ssize_t written = sys::RetryAfterSignal(
      -1, ::write, fd, &g_attach_request_byte, sizeof(g_attach_request_byte));
  if (written != static_cast<ssize_t>(sizeof(g_attach_request_byte)))
    return createStringErrorFmt(
        "Failed to write to {0}: {1}", fd_path,
        llvm::toString(errorCodeToError(errnoAsErrorCode())));

  LLDB_LOG(GetLog(GDBRLog::Plugin),
           "Asked the CUDA driver to attach to process {0}", process.GetID());
  return true;
}

Expected<uint32_t> CUDADebuggerAPI::ReadResumeForAttachDetach(
    SymbolAddressProvider get_symbol_address, NativeProcessProtocol &process) {
  return ReadFromHostSymbol<uint32_t>(get_symbol_address, process,
                                      Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH);
}

/// A breakpoint on \a function_name in \a library_name, with the symbols that
/// the debugger API's initialization needs.
static GPUBreakpointInfo GetBreakpointInfo(StringRef library_name,
                                           StringRef function_name) {
  GPUBreakpointInfo bp;
  bp.name_info = {library_name.str(), function_name.str()};
  bp.symbol_names.push_back(Symbols::CUDBG_IPC_FLAG_NAME.str());
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_PID.str());
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_REVISION.str());
  bp.symbol_names.push_back(Symbols::CUDBG_SESSION_ID.str());
  bp.symbol_names.push_back(Symbols::CUDBG_DEBUGGER_CAPABILITIES.str());
  bp.symbol_names.push_back(Symbols::CUDBG_INJECTION_PATH.str());
  return bp;
}

GPUBreakpointInfo
CUDADebuggerAPI::GetCuInitBreakpointInfo(StringRef library_name) {
  GPUBreakpointInfo bp = GetBreakpointInfo(library_name, Symbols::CU_INIT);
  bp.symbol_names.push_back(Symbols::CU_INIT.str());
  bp.symbol_names.push_back(Symbols::CUDBG_PRE_INIT.str());
  bp.symbol_names.push_back(Symbols::CUDBG_DEBUGGER_INITIALIZED.str());
  return bp;
}

GPUBreakpointInfo
CUDADebuggerAPI::GetPreInitBreakpointInfo(StringRef library_name) {
  return GetBreakpointInfo(library_name, Symbols::CUDBG_PRE_INIT);
}

GPUBreakpointInfo
CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(StringRef library_name) {
  GPUBreakpointInfo bp = GetBreakpointInfo(
      library_name, Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED);
  bp.symbol_names.push_back(Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH.str());
  return bp;
}

std::optional<CUDADebuggerAPI::BreakpointFunction>
CUDADebuggerAPI::GetBreakpointFunction(StringRef function_name) {
  if (function_name == Symbols::CU_INIT)
    return BreakpointFunction::CuInit;
  if (function_name == Symbols::CUDBG_PRE_INIT)
    return BreakpointFunction::PreInit;
  if (function_name == Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED)
    return BreakpointFunction::AttachFinished;
  return std::nullopt;
}

std::optional<uint64_t>
CUDADebuggerAPI::GetCuInitAddress(SymbolAddressProvider get_symbol_address) {
  return get_symbol_address(Symbols::CU_INIT);
}

bool CUDADebuggerAPI::HasPreInit(SymbolAddressProvider get_symbol_address) {
  return get_symbol_address(Symbols::CUDBG_PRE_INIT).has_value();
}
