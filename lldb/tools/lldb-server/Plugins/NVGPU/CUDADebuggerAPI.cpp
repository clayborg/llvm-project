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
#include "lldb/Utility/Log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/sysmacros.h>
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

// Bounds for the CUDBGAPI.initialize() retry/backoff (gap 8). The driver may
// briefly report CUDBG_ERROR_ATTACH_NOT_POSSIBLE right after injection; we back
// off starting at kInitRetryDelayMs (doubling, capped at kInitMaxRetryDelayMs)
// and give up after kInitTimeoutMs total.
static constexpr unsigned kInitRetryDelayMs = 100;
static constexpr unsigned kInitMaxRetryDelayMs = 1000;
static constexpr unsigned kInitTimeoutMs = 5000;

namespace Symbols {
static std::string CUDBG_IPC_FLAG_NAME = STRINGIFY_SYMBOL(CUDBG_IPC_FLAG_NAME);
static std::string CUDBG_APICLIENT_PID = STRINGIFY_SYMBOL(CUDBG_APICLIENT_PID);
static std::string CUDBG_APICLIENT_REVISION =
    STRINGIFY_SYMBOL(CUDBG_APICLIENT_REVISION);
static std::string CUDBG_SESSION_ID = STRINGIFY_SYMBOL(CUDBG_SESSION_ID);
static std::string CUDBG_DEBUGGER_CAPABILITIES =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_CAPABILITIES);
// Set by the debug engine to indicate it has initialized. Cleared on detach so
// the engine reinitializes from scratch on a later re-attach. Resolved
// OPTIONALLY (see ResolveInferiorDetachSymbols): a libcuda that does not export
// it must not regress attach or detach.
static std::string CUDBG_DEBUGGER_INITIALIZED =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_INITIALIZED);
static std::string CUDBG_INJECTION_PATH = "cudbgInjectionPath";
static std::string CUDBG_GET_API = "cudbgGetAPI";
static std::string CUDBG_GET_API_VERSION = "cudbgGetAPIVersion";
static std::string CUDA_INITIALIZATION_SYMBOL =
    CMAKE_NVGPU_INITIALIZATION_SYMBOL;
// Late-attach handshake symbols exported by the driver into the running
// process. See the "Attaching and Detaching" section of cudadebugger.h.
static std::string CUDBG_ATTACH_HANDLER_AVAILABLE =
    STRINGIFY_SYMBOL(CUDBG_ATTACH_HANDLER_AVAILABLE);
static std::string CUDBG_RESUME_FOR_ATTACH_DETACH =
    STRINGIFY_SYMBOL(CUDBG_RESUME_FOR_ATTACH_DETACH);
static std::string CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD =
    STRINGIFY_SYMBOL(CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD);
static std::string CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED =
    STRINGIFY_SYMBOL(CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED);
} // namespace Symbols

/// Read a uint32_t global from the running process at the address resolved for
/// \a symbol_name. Returns an error if the symbol could not be resolved or the
/// memory could not be read.
static llvm::Expected<uint32_t>
ReadUInt32FromHost(lldb_private::lldb_server::SymbolAddressProvider get_addr,
                   NativeProcessProtocol &linux_process,
                   llvm::StringRef symbol_name) {
  std::optional<uint64_t> symbol_address = get_addr(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                symbol_name);
  uint32_t value = 0;
  size_t bytes_read = 0;
  Status status = linux_process.ReadMemory(*symbol_address, &value,
                                           sizeof(value), bytes_read);
  if (status.Fail() || bytes_read != sizeof(value))
    return createStringErrorFmt("Failed to read symbol {0}: {1}", symbol_name,
                                status.AsCString());
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
                               NativeProcessProtocol &linux_process,
                               llvm::StringRef symbol_name, const T &value) {
  // Only fixed-size, trivially-copyable values are written through this helper.
  // Variable-length data (e.g. the injection path) is written with explicit
  // bounds checking by WriteInjectionPathToInferior so a destination buffer can
  // never be overrun.
  static_assert(std::is_trivially_copyable_v<T>,
                "WriteToHostSymbol only writes fixed-size trivially-copyable "
                "values; use WriteInjectionPathToInferior for strings");
  std::optional<uint64_t> symbol_address = get_addr(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {}",
                                symbol_name);

  const size_t value_size = sizeof(value);
  size_t bytes_written = 0;
  Status status =
      linux_process.WriteMemory(*symbol_address, &value, value_size,
                                bytes_written);
  if (status.Fail())
    return createStringErrorFmt("Failed to write symbol {}: {}", symbol_name,
                                status.AsCString());
  if (bytes_written != value_size)
    return createStringErrorFmt("Failed to write symbol {}", symbol_name);
  return Error::success();
}

/// Write the driver's CUDBG_INJECTION_PATH global into the inferior using a
/// bounded copy. Rejects any path that would not fit (including its trailing
/// NUL) in the driver's fixed-size path buffer, so an over-long, possibly
/// attacker-controlled path can never overrun the inferior's buffer.
static Error WriteInjectionPathToInferior(SymbolAddressProvider get_addr,
                                          NativeProcessProtocol &linux_process,
                                          llvm::StringRef path) {
  // PATH_MAX (CUDBG_INJECTION_PATH_MAX_SIZE) is the assumed size of the
  // driver's fixed path buffer; reject anything that would not fit including
  // its NUL so the bounded write below can never overrun the inferior buffer.
  if (path.size() + 1 > CUDADebuggerAPI::CUDBG_INJECTION_PATH_MAX_SIZE)
    return createStringErrorFmt(
        "CUDBG_INJECTION_PATH is too long ({0} bytes); the driver's injection "
        "path buffer holds at most {1} bytes including the NUL terminator",
        path.size(), CUDADebuggerAPI::CUDBG_INJECTION_PATH_MAX_SIZE);

  std::optional<uint64_t> symbol_address =
      get_addr(Symbols::CUDBG_INJECTION_PATH);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                Symbols::CUDBG_INJECTION_PATH);

  // Assemble an explicitly NUL-terminated buffer and write exactly its length;
  // the source StringRef is not guaranteed to be NUL-terminated.
  llvm::SmallVector<char, 256> buffer(path.begin(), path.end());
  buffer.push_back('\0');
  size_t bytes_written = 0;
  Status status = linux_process.WriteMemory(*symbol_address, buffer.data(),
                                            buffer.size(), bytes_written);
  if (status.Fail() || bytes_written != buffer.size())
    return createStringErrorFmt("Failed to write symbol {0}: {1}",
                                Symbols::CUDBG_INJECTION_PATH,
                                status.AsCString());
  return Error::success();
}

/// Publish the client identity and requested capabilities into the inferior.
///
/// \a set_ipc_flag controls the master "client ready" flag, which is written
/// last so the driver never sees a ready client whose identity and
/// capabilities have not landed yet. The attach path passes false and
/// publishes the flag once the attach procedure has finished, via SetIpcFlag.
static Error WriteInitializationSymbolsToHost(SymbolAddressProvider get_addr,
                                              NativeProcessProtocol &linux_process,
                                              uint32_t pid, uint32_t session_id,
                                              uint32_t revision,
                                              bool set_ipc_flag) {
  auto write_uint32_t = [&](const std::string &symbol_name,
                            const uint32_t &value) -> Error {
    return WriteToHostSymbol(get_addr, linux_process, symbol_name, value);
  };

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_PID, pid))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_REVISION, revision))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_SESSION_ID, session_id))
    return err;

  const uint32_t capabilities =
      CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS |
      CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;
  if (Error err =
          write_uint32_t(Symbols::CUDBG_DEBUGGER_CAPABILITIES, capabilities))
    return err;

  if (!set_ipc_flag)
    return Error::success();

  return CUDADebuggerAPI::SetIpcFlag(get_addr, linux_process,
                                     /*enabled=*/true);
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

static Error
WriteInjectionPathToLibcuda(SymbolAddressProvider get_addr,
                            NativeProcessProtocol &linux_process,
                            llvm::sys::DynamicLibrary &libcuda,
                            StringRef libcuda_library_name) {
  const char *path = getenv("CUDBG_INJECTION_PATH");
  if (!path)
    return Error::success();

  llvm::StringRef path_ref(path);
  if (path_ref.size() + 1 > CUDADebuggerAPI::CUDBG_INJECTION_PATH_MAX_SIZE)
    return createStringErrorFmt(
        "CUDBG_INJECTION_PATH is too long ({0} bytes); the driver's injection "
        "path buffer holds at most {1} bytes including the NUL terminator",
        path_ref.size(), CUDADebuggerAPI::CUDBG_INJECTION_PATH_MAX_SIZE);

  // Publish the path into the inferior's libcuda copy (bounds-checked).
  if (Error err =
          WriteInjectionPathToInferior(get_addr, linux_process, path_ref))
    return err;

  // Also update this server's locally loaded libcuda image so the API table it
  // returns is consistent. Use a bounded copy into the driver's fixed-size
  // (assumed PATH_MAX) buffer: the length was validated above against the same
  // maximum, so the memcpy plus NUL stays within the buffer.
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

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::InitializeImpl(SymbolAddressProvider get_symbol_address,
                                StringRef libcuda_library_name,
                                NativeProcessProtocol &linux_process,
                                InitContext init_context) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "CUDADebuggerAPI::Initialize()");

  const uint32_t pid = getpid();
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
          get_symbol_address, linux_process, pid, session_id, revision,
          /*set_ipc_flag=*/init_context == InitContext::eLaunch))
    return err;

  if (Error err = WriteConfigurationToLibcuda(libcuda, pid, revision,
                                              session_id, libcuda_library_name))
    return err;

  if (Error err = WriteInjectionPathToLibcuda(get_symbol_address, linux_process,
                                              libcuda, libcuda_library_name))
    return err;

  Expected<CUDBGAPI> api_or =
      GetRawAPIInstance(libcuda, libcuda_library_name, api_version);
  if (!api_or)
    return api_or.takeError();

  CUDADebuggerAPI api(*api_or, api_version);

  // Bounded retry around initialize(). Right after the driver injects the debug
  // engine it can briefly report CUDBG_ERROR_ATTACH_NOT_POSSIBLE while it is not
  // yet ready to be attached; back off and retry rather than failing the attach.
  // Any other failure is terminal and returned immediately. Initialization runs
  // unlocked (no caller holds m_attach_mutex here), so the sleeps below do not
  // stall the attach state machine.
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
    std::this_thread::sleep_for(
        std::min(milliseconds(backoff_ms), remaining));
    backoff_ms = std::min(backoff_ms * 2, kInitMaxRetryDelayMs);
  }

  if (Error err = VerifyDebuggerCapabilities(api))
    return err;

  return api;
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::Initialize(SymbolAddressProvider get_symbol_address,
                            StringRef libcuda_library_name,
                            NativeProcessProtocol &linux_process,
                            InitContext init_context) {
  Expected<CUDADebuggerAPI> api = InitializeImpl(
      get_symbol_address, libcuda_library_name, linux_process, init_context);
  if (!api)
    return createStringErrorFmt(
        "Failed to initialize the CUDA Debugger API. {}",
        llvm::toString(api.takeError()));
  return api;
}

std::vector<std::string> CUDADebuggerAPI::GetAttachSymbolNames() {
  return {
      Symbols::CUDBG_ATTACH_HANDLER_AVAILABLE,
      Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH,
      Symbols::CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD,
      Symbols::CUDBG_IPC_FLAG_NAME,
      Symbols::CUDBG_APICLIENT_PID,
      Symbols::CUDBG_APICLIENT_REVISION,
      Symbols::CUDBG_SESSION_ID,
      Symbols::CUDBG_DEBUGGER_CAPABILITIES,
      Symbols::CUDBG_INJECTION_PATH,
  };
}

namespace {
/// The location of libcuda in the inferior: how it is mapped and enough
/// identity to open the *exact* file backing the mapping (not just a pathname
/// that might resolve to a different file in our mount namespace).
struct InferiorLibrary {
  /// Pathname as it appears in /proc/<pid>/maps.
  std::string path;
  /// Start address of the mapping with file offset 0 (the ELF header), i.e. the
  /// library's load base.
  uint64_t load_base = 0;
  /// The "start-end" token from the maps line, used to address the mapping via
  /// /proc/<pid>/map_files/<start>-<end>.
  std::string map_range;
  /// Device and inode of the backing file, as reported by maps, used to verify
  /// that any file we open really is the mapped image.
  dev_t dev = 0;
  ino_t inode = 0;
};
} // namespace

/// \return true if \a filename is libcuda's soname: the unversioned "libcuda.so"
/// or a versioned "libcuda.so.<digits-and-dots>" (e.g. libcuda.so.1,
/// libcuda.so.550.54.15). This avoids matching unrelated paths that merely
/// contain "libcuda.so" as a substring.
static bool IsLibcudaSoname(llvm::StringRef filename) {
  if (filename == "libcuda.so")
    return true;
  if (!filename.consume_front("libcuda.so."))
    return false;
  // The version suffix must be one or more dot-separated runs of digits (e.g.
  // "1" or "550.54.15"). Require at least one digit and reject empty components
  // -- a leading, trailing, or doubled dot -- so malformed sonames such as
  // "libcuda.so.", "libcuda.so..", and "libcuda.so.1." do not match.
  bool has_digit = false;
  bool prev_was_dot = true; // the boundary just before the suffix acts as a dot
  for (char c : filename) {
    if (c == '.') {
      if (prev_was_dot)
        return false; // empty version component
      prev_was_dot = true;
      continue;
    }
    if (c < '0' || c > '9')
      return false;
    has_digit = true;
    prev_was_dot = false;
  }
  // prev_was_dot still set here means the suffix ended on a dot.
  return has_digit && !prev_was_dot;
}

/// Parse /proc/<pid>/maps to locate libcuda in the inferior. We cannot rely on
/// llvm::MemoryBuffer for procfs (the reported size is 0), so read it directly.
static Expected<InferiorLibrary> FindInferiorLibcuda(lldb::pid_t pid) {
  std::string maps_path = llvm::formatv("/proc/{0}/maps", pid).str();
  int fd = ::open(maps_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return createStringErrorFmt("Failed to open {0}: {1}", maps_path,
                                ::strerror(errno));
  std::string contents;
  char buffer[4096];
  while (true) {
    ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n > 0) {
      contents.append(buffer, n);
      continue;
    }
    if (n == 0)
      break; // EOF
    if (errno == EINTR)
      continue; // interrupted before reading anything; retry
    // Any other error leaves us with a possibly truncated view of the maps;
    // surface it (with errno) rather than silently parsing an incomplete file.
    int read_errno = errno;
    ::close(fd);
    return createStringErrorFmt("Failed to read {0}: {1}", maps_path,
                                ::strerror(read_errno));
  }
  ::close(fd);

  // Each line is "start-end perms offset dev inode pathname". The five leading
  // fields are fixed-width tokens; the pathname is the remainder of the line
  // and may legitimately contain spaces, so it must not be split. We want the
  // mapping of a libcuda image with file offset 0 (the ELF header), whose start
  // address is the library's load base.
  llvm::StringRef remaining(contents);
  while (!remaining.empty()) {
    std::pair<llvm::StringRef, llvm::StringRef> split = remaining.split('\n');
    llvm::StringRef line = split.first;
    remaining = split.second;

    llvm::StringRef cursor = line;
    auto next_field = [&cursor]() -> llvm::StringRef {
      cursor = cursor.ltrim(' ');
      size_t space = cursor.find(' ');
      llvm::StringRef field = cursor.substr(0, space);
      cursor = (space == llvm::StringRef::npos) ? llvm::StringRef()
                                                : cursor.substr(space);
      return field;
    };

    llvm::StringRef address_range = next_field();
    llvm::StringRef perms = next_field();
    (void)perms;
    llvm::StringRef offset_str = next_field();
    llvm::StringRef dev_str = next_field();
    llvm::StringRef inode_str = next_field();
    llvm::StringRef pathname = cursor.ltrim(' ');
    if (address_range.empty() || pathname.empty())
      continue;

    // Linux appends " (deleted)" to the pathname of a mapping whose backing
    // file has been unlinked (or replaced). Strip it before matching the soname
    // and before deriving on-disk path candidates, otherwise "libcuda.so.1
    // (deleted)" never matches and the robust /proc/<pid>/map_files path is
    // skipped. The map_files symlink is keyed by the address range (preserved
    // below), so it still resolves to the exact backing file object.
    llvm::StringRef map_pathname = pathname;
    map_pathname.consume_back(" (deleted)");

    if (!IsLibcudaSoname(llvm::sys::path::filename(map_pathname)))
      continue;

    uint64_t file_offset = 0;
    if (offset_str.getAsInteger(16, file_offset) || file_offset != 0)
      continue; // the ELF header (and the load base) lives at file offset 0

    uint64_t start = 0;
    if (address_range.split('-').first.getAsInteger(16, start))
      continue;

    // dev is "major:minor" in hex; inode is decimal. Parse best-effort; the
    // values are used to verify the file we later open is the mapped image.
    uint64_t dev_major = 0, dev_minor = 0, inode = 0;
    std::pair<llvm::StringRef, llvm::StringRef> dev_parts = dev_str.split(':');
    dev_parts.first.getAsInteger(16, dev_major);
    dev_parts.second.getAsInteger(16, dev_minor);
    inode_str.getAsInteger(10, inode);

    InferiorLibrary result;
    result.path = map_pathname.str();
    result.load_base = start;
    result.map_range = address_range.str();
    result.dev = ::makedev(static_cast<unsigned>(dev_major),
                           static_cast<unsigned>(dev_minor));
    result.inode = static_cast<ino_t>(inode);
    return result;
  }

  return createStringError(
      "libcuda is not mapped in the inferior yet (no libcuda.so mapping "
      "with file offset 0 found in the process maps).");
}

/// Open libcuda's on-disk image for the inferior, preferring handles tied to
/// the *mapping's* identity rather than a pathname that could resolve to a
/// different file (wrong mount namespace, replaced or deleted file, ...). The
/// opened file's device and inode are verified against the values reported in
/// /proc/<pid>/maps before it is trusted.
static Expected<std::unique_ptr<llvm::MemoryBuffer>>
OpenInferiorLibcuda(lldb::pid_t pid, const InferiorLibrary &libcuda) {
  // Candidates in order of trustworthiness:
  //  1. /proc/<pid>/map_files/<start>-<end>: a kernel-maintained symlink to the
  //     exact file object backing the mapping (correct across mount namespaces
  //     and for replaced/deleted files; requires ptrace-level access, which the
  //     debugger has).
  //  2. /proc/<pid>/root/<path>: resolves the pathname in the inferior's own
  //     filesystem/mount view.
  //  3. The raw pathname, as a last resort.
  llvm::SmallVector<std::string, 3> candidates;
  candidates.push_back(
      llvm::formatv("/proc/{0}/map_files/{1}", pid, libcuda.map_range).str());
  candidates.push_back(
      llvm::formatv("/proc/{0}/root{1}", pid, libcuda.path).str());
  candidates.push_back(libcuda.path);

  std::string errors;
  for (const std::string &candidate : candidates) {
    // Open the candidate exactly once and verify the identity of *that* open
    // file description (via fstat on the same fd) before mapping it. Using
    // ::stat() and then reopening the path with MemoryBuffer::getFile() would
    // leave a TOCTOU window in which the verified file could be swapped for a
    // different one between the check and the open. Building the buffer from
    // this fd guarantees we parse the bytes we vetted.
    int fd = ::open(candidate.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      errors +=
          llvm::formatv("  {0}: {1}\n", candidate, ::strerror(errno)).str();
      continue;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      errors +=
          llvm::formatv("  {0}: fstat failed: {1}\n", candidate, ::strerror(errno))
              .str();
      ::close(fd);
      continue;
    }
    // Verify the opened file's identity against what the kernel reported for the
    // mapping. (inode 0 means maps did not report a backing file; skip the check
    // in that unusual case rather than rejecting outright.)
    if (libcuda.inode != 0 &&
        (st.st_ino != libcuda.inode || st.st_dev != libcuda.dev)) {
      errors += llvm::formatv("  {0}: device/inode mismatch with the mapped "
                              "image; skipping\n",
                              candidate)
                    .str();
      ::close(fd);
      continue;
    }
    // Map the already-open, already-verified fd. getOpenFile does not take
    // ownership of the fd (it mmaps or reads eagerly), so close it afterward on
    // every path.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or =
        llvm::MemoryBuffer::getOpenFile(
            llvm::sys::fs::convertFDToNativeFile(fd), candidate,
            static_cast<uint64_t>(st.st_size),
            /*RequiresNullTerminator=*/false);
    ::close(fd);
    if (!buffer_or) {
      errors +=
          llvm::formatv("  {0}: {1}\n", candidate, buffer_or.getError().message())
              .str();
      continue;
    }
    return std::move(*buffer_or);
  }
  return createStringErrorFmt(
      "Failed to open a verified libcuda image for the inferior; tried:\n{0}",
      errors);
}

/// Resolve the inferior load addresses of \a wanted symbols from libcuda's
/// dynamic symbol table. Symbols in \a required must all resolve or an error is
/// returned; any \a wanted symbol not in \a required is optional and simply
/// absent from the result when libcuda does not export it.
static Expected<llvm::StringMap<uint64_t>>
ResolveInferiorSymbols(NativeProcessProtocol &linux_process,
                       llvm::ArrayRef<std::string> wanted,
                       const llvm::StringSet<> &required) {
  Log *log = GetLog(GDBRLog::Plugin);

  Expected<InferiorLibrary> libcuda_or =
      FindInferiorLibcuda(linux_process.GetID());
  if (!libcuda_or)
    return libcuda_or.takeError();
  const InferiorLibrary &libcuda = *libcuda_or;
  LLDB_LOG(log,
           "CUDADebuggerAPI::ResolveInferiorSymbols(). libcuda at {0} "
           "base {1:x}",
           libcuda.path, libcuda.load_base);

  // Open the exact file backing the mapping (verified by device/inode) rather
  // than trusting the bare pathname, then parse its dynamic symbol table. We
  // only read the section headers and dynsym/dynstr lazily through the mmap.
  Expected<std::unique_ptr<llvm::MemoryBuffer>> buffer_or =
      OpenInferiorLibcuda(linux_process.GetID(), libcuda);
  if (!buffer_or)
    return buffer_or.takeError();
  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*buffer_or);

  // Validate the ELF identity before parsing as 64-bit little-endian. A
  // mismatch means we resolved the wrong file or libcuda is an unexpected
  // format; either way returning a targeted diagnostic beats misinterpreting
  // the bytes.
  llvm::StringRef raw = buffer->getBuffer();
  if (raw.size() <= llvm::ELF::EI_DATA ||
      raw.take_front(4) != llvm::StringRef(llvm::ELF::ElfMagic, 4))
    return createStringErrorFmt("{0} is not an ELF object file", libcuda.path);
  const unsigned char ei_class =
      static_cast<unsigned char>(raw[llvm::ELF::EI_CLASS]);
  const unsigned char ei_data =
      static_cast<unsigned char>(raw[llvm::ELF::EI_DATA]);
  if (ei_class != llvm::ELF::ELFCLASS64 || ei_data != llvm::ELF::ELFDATA2LSB)
    return createStringErrorFmt(
        "{0} is not a 64-bit little-endian ELF image (EI_CLASS={1}, "
        "EI_DATA={2}); the NVGPU late attach path only supports "
        "ELFCLASS64/ELFDATA2LSB libcuda images",
        libcuda.path, static_cast<unsigned>(ei_class),
        static_cast<unsigned>(ei_data));

  Expected<llvm::object::ELF64LEObjectFile> object_or =
      llvm::object::ELF64LEObjectFile::create(buffer->getMemBufferRef());
  if (!object_or)
    return object_or.takeError();
  const llvm::object::ELF64LEObjectFile &elf = *object_or;

  // Compute the load bias. A symbol's st_value is relative to the library's
  // link-time base (the p_vaddr of the PT_LOAD that maps file offset 0), while
  // load_base is where that segment (the ELF header) is actually mapped. The
  // bias is the difference. Require such a segment to exist -- without it we
  // have no anchor and cannot map st_value onto a runtime address.
  Expected<llvm::object::ELF64LE::PhdrRange> phdrs_or =
      elf.getELFFile().program_headers();
  if (!phdrs_or)
    return phdrs_or.takeError();
  std::optional<uint64_t> link_time_base;
  for (const llvm::object::ELF64LE::Phdr &phdr : *phdrs_or) {
    if (phdr.p_type == llvm::ELF::PT_LOAD && phdr.p_offset == 0) {
      link_time_base = phdr.p_vaddr;
      break;
    }
  }
  if (!link_time_base)
    return createStringErrorFmt(
        "{0} has no PT_LOAD segment mapping file offset 0; cannot compute the "
        "load bias for the late-attach symbols",
        libcuda.path);
  // load_base < link_time_base would make the bias underflow into a huge
  // positive value; reject it instead of computing bogus addresses.
  if (libcuda.load_base < *link_time_base)
    return createStringErrorFmt(
        "libcuda load base {0:x} is below its link-time base {1:x}; refusing "
        "to compute a negative load bias",
        libcuda.load_base, *link_time_base);
  const uint64_t load_bias = libcuda.load_base - *link_time_base;

  // Resolve each wanted symbol, tracking which required ones remain so we can
  // report an actionable error listing the missing ones instead of silently
  // returning a partial map (which would make the caller probe forever).
  // Optional symbols (wanted but not required) simply stay absent.
  llvm::StringSet<> remaining;
  for (const llvm::StringRef name : required.keys())
    remaining.insert(name);

  llvm::StringMap<uint64_t> result;
  for (const llvm::object::ELFSymbolRef &symbol :
       elf.getDynamicSymbolIterators()) {
    Expected<llvm::StringRef> name_or = symbol.getName();
    if (!name_or) {
      llvm::consumeError(name_or.takeError());
      continue;
    }
    llvm::StringRef name = *name_or;
    if (!llvm::is_contained(wanted, name))
      continue;

    // Skip undefined symbols: they carry no usable address (the definition
    // lives elsewhere), so accepting their st_value would be wrong.
    Expected<uint32_t> flags_or = symbol.getFlags();
    if (!flags_or) {
      llvm::consumeError(flags_or.takeError());
      continue;
    }
    if (*flags_or & llvm::object::SymbolRef::SF_Undefined)
      continue;

    Expected<uint64_t> value_or = symbol.getValue();
    if (!value_or) {
      llvm::consumeError(value_or.takeError());
      continue;
    }
    const uint64_t st_value = *value_or;
    // Checked add: load_bias + st_value must not wrap around the address space.
    if (load_bias > std::numeric_limits<uint64_t>::max() - st_value)
      return createStringErrorFmt(
          "address overflow resolving symbol {0} in {1} (load bias {2:x} + "
          "st_value {3:x})",
          name, libcuda.path, load_bias, st_value);

    result[name] = load_bias + st_value;
    remaining.erase(name);
  }

  if (!remaining.empty()) {
    llvm::SmallVector<llvm::StringRef, 8> missing;
    for (const auto &entry : remaining)
      missing.push_back(entry.getKey());
    llvm::sort(missing);
    return createStringErrorFmt(
        "libcuda image {0} is missing required late-attach symbols: {1}. The "
        "running process may not be a debuggable CUDA application, or libcuda "
        "is too old to support late attach.",
        libcuda.path, llvm::join(missing, ", "));
  }

  return result;
}

Expected<llvm::StringMap<uint64_t>>
CUDADebuggerAPI::ResolveInferiorAttachSymbols(
    NativeProcessProtocol &linux_process) {
  std::vector<std::string> wanted = GetAttachSymbolNames();
  llvm::StringSet<> required;
  for (const std::string &name : wanted)
    required.insert(name);
  return ResolveInferiorSymbols(linux_process, wanted, required);
}

Expected<llvm::StringMap<uint64_t>>
CUDADebuggerAPI::ResolveInferiorDetachSymbols(
    NativeProcessProtocol &linux_process) {
  // The detach path needs the same handshake symbols as attach, plus the
  // OPTIONAL CUDBG_DEBUGGER_INITIALIZED flag (gap 7): it is reset when present
  // but its absence must not regress detach. So it is "wanted" but not
  // "required".
  std::vector<std::string> required_names = GetAttachSymbolNames();
  llvm::StringSet<> required;
  for (const std::string &name : required_names)
    required.insert(name);

  std::vector<std::string> wanted = required_names;
  wanted.push_back(Symbols::CUDBG_DEBUGGER_INITIALIZED);
  return ResolveInferiorSymbols(linux_process, wanted, required);
}

Error CUDADebuggerAPI::ResetDetachSymbols(
    SymbolAddressProvider get_symbol_address,
    NativeProcessProtocol &linux_process) {
  // Reset the driver's handshake globals so a later debugger can re-attach
  // cleanly (mirrors cuda-gdb's cuda_do_detach):
  //   - CUDBG_DEBUGGER_CAPABILITIES -> CUDBG_DEBUGGER_CAPABILITY_NONE, so the
  //     next attach is not constrained by capabilities this session requested,
  //   - CUDBG_DEBUGGER_INITIALIZED -> 0 (OPTIONAL; skipped if libcuda does not
  //     export it), so the debug engine reinitializes from scratch,
  //   - CUDBG_IPC_FLAG_NAME -> 0, clearing the "client ready" flag.
  const uint32_t zero = 0;
  if (Error err = WriteToHostSymbol(get_symbol_address, linux_process,
                                    Symbols::CUDBG_DEBUGGER_CAPABILITIES, zero))
    return err;

  if (get_symbol_address(Symbols::CUDBG_DEBUGGER_INITIALIZED)) {
    if (Error err =
            WriteToHostSymbol(get_symbol_address, linux_process,
                              Symbols::CUDBG_DEBUGGER_INITIALIZED, zero))
      return err;
  }

  if (Error err = WriteToHostSymbol(get_symbol_address, linux_process,
                                    Symbols::CUDBG_IPC_FLAG_NAME, zero))
    return err;

  return Error::success();
}

Expected<bool>
CUDADebuggerAPI::IsLateAttachSupported(SymbolAddressProvider get_symbol_address,
                                       NativeProcessProtocol &linux_process) {
  // The driver sets CUDBG_ATTACH_HANDLER_AVAILABLE once it can service an
  // attach request. If the symbol cannot be resolved at all the running process
  // is not a (debuggable) CUDA process.
  Expected<uint32_t> available = ReadUInt32FromHost(
      get_symbol_address, linux_process, Symbols::CUDBG_ATTACH_HANDLER_AVAILABLE);
  if (!available)
    return available.takeError();
  return *available != 0;
}

Error CUDADebuggerAPI::SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                  NativeProcessProtocol &linux_process,
                                  bool enabled) {
  const uint32_t value = enabled ? 1 : 0;
  return WriteToHostSymbol(get_symbol_address, linux_process,
                           Symbols::CUDBG_IPC_FLAG_NAME, value);
}

Expected<uint32_t> CUDADebuggerAPI::ReadResumeForAttachDetach(
    SymbolAddressProvider get_symbol_address,
    NativeProcessProtocol &linux_process) {
  return ReadUInt32FromHost(get_symbol_address, linux_process,
                            Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH);
}

Error CUDADebuggerAPI::InitiateSafeAttach(
    SymbolAddressProvider get_symbol_address,
    NativeProcessProtocol &linux_process) {
  Log *log = GetLog(GDBRLog::Plugin);
  const uint32_t pid = getpid();
  // session_id 0 matches the launch path (WriteInitializationSymbolsToHost).
  // The revision is the compiled API revision: the driver's version cannot be
  // negotiated yet (the debug engine is not loaded until the procedure runs).
  // These handshake globals are rewritten with the negotiated values when the
  // API is initialized at CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED (see
  // Initialize), so any difference is transient and same-major by construction.
  const uint32_t session_id = 0;
  const uint32_t revision = nvgpu::CudbgApiVersion::Compiled().revision;

  // Publish the client handshake globals into the running process so the driver
  // knows who is attaching before it injects the debug engine. The IPC "client
  // ready" flag is deliberately NOT published here: on the attach path it must
  // stay clear until the attach procedure has finished, and is then only set
  // when the driver does not want the application resumed to replay its state.
  // See CUDADebuggerAPI::SetIpcFlag.
  if (Error err = WriteInitializationSymbolsToHost(
          get_symbol_address, linux_process, pid, session_id, revision,
          /*set_ipc_flag=*/false))
    return err;

  // If an alternate debug engine library is configured, publish it into the
  // inferior before triggering so the driver injects that library rather than
  // the default. This mirrors the launch path (WriteInjectionPathToLibcuda) and
  // must happen before the magic byte, since the driver reads the path when it
  // services the attach request.
  if (const char *injection_path = getenv("CUDBG_INJECTION_PATH")) {
    if (Error err = WriteInjectionPathToInferior(
            get_symbol_address, linux_process, llvm::StringRef(injection_path)))
      return err;
  }

  // Read the attach-procedure file descriptor exported by the driver. This is
  // an int that lives in the inferior's address space and refers to a pipe in
  // the inferior's fd table.
  std::optional<uint64_t> fd_symbol_address =
      get_symbol_address(Symbols::CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD);
  if (!fd_symbol_address)
    return createStringErrorFmt(
        "Couldn't find address for symbol {0}",
        Symbols::CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD);

  int32_t attach_fd = -1;
  size_t bytes_read = 0;
  Status status = linux_process.ReadMemory(*fd_symbol_address, &attach_fd,
                                           sizeof(attach_fd), bytes_read);
  if (status.Fail() || bytes_read != sizeof(attach_fd))
    return createStringErrorFmt("Failed to read the attach procedure FD: {0}",
                                status.AsCString());

  if (attach_fd < 0)
    return createStringError(
        "The driver did not publish a safe attach procedure FD "
        "(cudbgInitiateDebuggerAttachProcedureFd is -1): it is either not ready "
        "yet or this driver is too old to support the safe attach mechanism. "
        "LLDB only supports the safe debugger attach mechanism for CUDA late "
        "attach; the legacy cudbgApiAttach() injection path is not supported. "
        "Use a CUDA driver new enough to provide the safe attach procedure.");

  // The fd belongs to the inferior's fd table; reach it through procfs and
  // write the magic byte to wake the driver's interrupt handler so it can
  // safely inject the debug engine.
  std::string fd_path =
      llvm::formatv("/proc/{0}/fd/{1}", linux_process.GetID(), attach_fd).str();
  LLDB_LOG(log, "CUDADebuggerAPI::InitiateSafeAttach(). Writing magic byte to {0}",
           fd_path);

  int host_fd = ::open(fd_path.c_str(), O_WRONLY | O_CLOEXEC);
  if (host_fd < 0)
    return createStringErrorFmt("Failed to open {0}: {1}", fd_path,
                                ::strerror(errno));

  const uint8_t magic = ATTACH_PROCEDURE_MAGIC_BYTE;
  ssize_t written = 0;
  int write_errno = 0;
  while (true) {
    written = ::write(host_fd, &magic, sizeof(magic));
    // Retry only when interrupted before any byte was written; preserve errno
    // from the failing syscall (the next loop iteration may clobber it).
    if (written < 0 && errno == EINTR)
      continue;
    write_errno = errno;
    break;
  }
  ::close(host_fd);
  if (written != static_cast<ssize_t>(sizeof(magic)))
    return createStringErrorFmt(
        "Failed to write the magic byte to {0}: {1}", fd_path,
        written < 0 ? ::strerror(write_errno) : "short write");

  return Error::success();
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
  return bp;
}

bool CUDADebuggerAPI::IsAttachFinishedBreakpoint(StringRef function_name) {
  return function_name == Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED;
}

GPUBreakpointInfo
CUDADebuggerAPI::GetAttachFinishedBreakpointInfo(StringRef library_name) {
  GPUBreakpointInfo bp;
  bp.name_info = {library_name.str(),
                  Symbols::CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED};
  // When the attach procedure finishes the debug engine has been injected, so
  // resolving these symbols at the breakpoint lets the plugin initialize the
  // API exactly like the launch path does. CUDBG_RESUME_FOR_ATTACH_DETACH is
  // additionally needed to decide, once the API is up, whether the driver will
  // replay pre-existing state (and send CUDBG_EVENT_ATTACH_COMPLETE) or whether
  // the client must publish the IPC flag and finish the attach itself.
  bp.symbol_names.push_back(Symbols::CUDBG_RESUME_FOR_ATTACH_DETACH);
  bp.symbol_names.push_back(Symbols::CUDBG_IPC_FLAG_NAME);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_PID);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_REVISION);
  bp.symbol_names.push_back(Symbols::CUDBG_SESSION_ID);
  bp.symbol_names.push_back(Symbols::CUDBG_DEBUGGER_CAPABILITIES);
  bp.symbol_names.push_back(Symbols::CUDBG_INJECTION_PATH);
  return bp;
}
