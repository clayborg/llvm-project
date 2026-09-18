//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
#define LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H

#include "cudadebugger.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
// The runtime CudbgApiVersion type, used to pick an API version the attached
// driver supports.
#include "lldb/Utility/NVGPU/CUDADebuggerAPIVersion.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>
#include <vector>

namespace lldb_private::lldb_server {

/// Custom deleter for CUDBGAPI.
void CUDBGAPIDeleter(CUDBGAPI api);

/// Resolves the load address of a named symbol in the native (CPU) process.
///
/// Symbol-name to load-address resolution requires the ELF symbol table. The
/// launch path obtains addresses from a breakpoint hit (\ref
/// GPUPluginBreakpointHitArgs), which carries the symbol values resolved by the
/// LLDB client. The late attach path resolves them entirely server-side by
/// parsing the inferior's libcuda image (see ResolveInferiorAttachSymbols), so
/// that no gdb-remote round-trip is issued on the attach hot path. Both adapt
/// to this common callback so the initialization code does not care where the
/// addresses came from. Returns std::nullopt when the symbol could not be
/// resolved.
using SymbolAddressProvider =
    llvm::function_ref<std::optional<uint64_t>(llvm::StringRef)>;

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The names of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *LIBCUDA_LIBRARY_NAME = "libcuda.so";
  static constexpr const char *LIBCUDA_LIBRARY_NAME_ALT = "libcuda.so.1";

  /// Magic byte written to the attach-procedure FD to request that the driver
  /// safely inject the debug engine. The driver accepts any value; this matches
  /// the value used by cuda-gdb (CUOS_EVENT_MAGIC_BYTE).
  static constexpr uint8_t ATTACH_PROCEDURE_MAGIC_BYTE = 0xAB;

  /// Maximum number of bytes (including the trailing NUL) that may be written
  /// to the driver's CUDBG_INJECTION_PATH buffer. The driver declares this
  /// global as a fixed-size path buffer sized at PATH_MAX, so we assume
  /// PATH_MAX (4096) as the maximum rather than deriving the bound from the
  /// ELF symbol's st_size. We never write more than this many bytes into either
  /// the inferior's copy or this server's locally loaded libcuda image, and
  /// reject any longer path, so an attacker-controlled CUDBG_INJECTION_PATH
  /// cannot overrun those buffers.
  static constexpr size_t CUDBG_INJECTION_PATH_MAX_SIZE = 4096;

  /// Which path is bringing the debugger API up. The two differ only in how
  /// the driver's IPC "client ready" flag is published; see \ref SetIpcFlag.
  enum class InitContext {
    /// cuInit-style initialization. The debugger is present before CUDA comes
    /// up and there is no attach procedure to sequence against, so the IPC
    /// flag is published as the last step of initialization.
    eLaunch,
    /// Late attach. The IPC flag must stay clear until the attach procedure
    /// has finished, and even then it is only published conditionally, so
    /// initialization leaves it alone entirely and the caller decides.
    eLateAttach,
  };

  /// Initialize the CUDA debugger API.
  ///
  /// \param get_symbol_address
  ///     Resolver for native-process symbol load addresses. The launch path
  ///     supplies these from a breakpoint hit; the late attach path resolves
  ///     them server-side (see ResolveInferiorAttachSymbols).
  /// \param libcuda_library_name
  ///     The name of the CUDA library that contains the CUDA debugger API.
  /// \param linux_process
  ///     The native (CPU) process being debugged.
  /// \param init_context
  ///     Whether this initialization is for a launch or a late attach.
  static llvm::Expected<CUDADebuggerAPI>
  Initialize(SymbolAddressProvider get_symbol_address,
             llvm::StringRef libcuda_library_name,
             NativeProcessProtocol &linux_process, InitContext init_context);

  /// Publish (or clear) the driver's CUDBG_IPC_FLAG_NAME, the master "an API
  /// client is ready, emit callbacks" flag.
  ///
  /// Sequencing matters, because the driver emits no callbacks at all until a
  /// client has declared itself ready. The flag is written as the last step of
  /// initialization so the driver never observes a ready client whose PID,
  /// revision, session and capabilities have not been published yet. The attach
  /// path additionally keeps it clear until the attach procedure has finished
  /// and the new-event callback is registered, so no notification can be
  /// emitted before there is anything able to observe it. It is then set
  /// unconditionally, mirroring cuda-gdb's cuda_initialize_target, which
  /// publishes the flag during API bring-up on both paths and before it reads
  /// CUDBG_RESUME_FOR_ATTACH_DETACH.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  /// \param[in] enabled
  ///     Whether to set (true) or clear (false) the flag.
  ///
  /// \return
  ///     Error::success() on success, or an error describing the failed write.
  static llvm::Error SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                NativeProcessProtocol &linux_process,
                                bool enabled);

  /// \return the set of native-process symbol names needed for the late attach
  /// handshake, used by ResolveInferiorAttachSymbols.
  static std::vector<std::string> GetAttachSymbolNames();

  /// Resolve the inferior load addresses of the late-attach handshake symbols
  /// without any gdb-remote round-trip.
  ///
  /// Issuing a client->server packet during attach stop processing corrupts the
  /// in-flight gdb-remote continue cycle, so the late attach path resolves these
  /// symbols entirely server-side instead: it locates libcuda in the inferior's
  /// address space via /proc/<pid>/maps and reads its dynamic symbol table from
  /// disk, mapping each symbol's st_value onto the library's load base.
  ///
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  ///
  /// \return
  ///     A map from symbol name to inferior load address, or an error if
  ///     libcuda is not (yet) present in the inferior or could not be parsed.
  static llvm::Expected<llvm::StringMap<uint64_t>>
  ResolveInferiorAttachSymbols(NativeProcessProtocol &linux_process);

  /// Like ResolveInferiorAttachSymbols, but also resolves the OPTIONAL
  /// CUDBG_DEBUGGER_INITIALIZED flag used by the detach path (gap 7). The flag
  /// is included when libcuda exports it and silently omitted otherwise, so a
  /// libcuda lacking it does not regress detach.
  ///
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  ///
  /// \return
  ///     A map from symbol name to inferior load address, or an error if
  ///     libcuda is not present or a required handshake symbol is missing.
  static llvm::Expected<llvm::StringMap<uint64_t>>
  ResolveInferiorDetachSymbols(NativeProcessProtocol &linux_process);

  /// Reset the driver's handshake globals during detach so a later debugger can
  /// re-attach cleanly: clear the requested debugger capabilities, clear the
  /// OPTIONAL CUDBG_DEBUGGER_INITIALIZED flag (skipped when absent), and clear
  /// the IPC ready flag. Mirrors cuda-gdb's cuda_do_detach.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  ///
  /// \return
  ///     Error::success() on success, or an error describing the first failed
  ///     write.
  static llvm::Error
  ResetDetachSymbols(SymbolAddressProvider get_symbol_address,
                     NativeProcessProtocol &linux_process);

  /// \return true if the running process advertises a usable safe-attach
  /// handler (i.e. CUDBG_ATTACH_HANDLER_AVAILABLE is set). Returns false (not an
  /// error) when CUDA is present but the handler is unavailable, and an error
  /// when the required symbols could not be read.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  static llvm::Expected<bool>
  IsLateAttachSupported(SymbolAddressProvider get_symbol_address,
                        NativeProcessProtocol &linux_process);

  /// Read the value of the CUDBG_RESUME_FOR_ATTACH_DETACH flag from the running
  /// process. When set, the application (including the CPU) must keep running
  /// for the driver to complete the attach procedure.
  static llvm::Expected<bool>
  ShouldResumeForAttachDetach(SymbolAddressProvider get_symbol_address,
                              NativeProcessProtocol &linux_process);

  /// Initiate the safe debugger attach procedure on a running process.
  ///
  /// Writes the client handshake globals (PID, revision, session, capabilities)
  /// into the inferior, reads the attach-procedure file descriptor exported by
  /// the driver, and writes the magic byte to it so the driver injects the
  /// debug engine at a safe point. After this completes the driver eventually
  /// calls CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED, which the plugin observes via
  /// a breakpoint.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  /// \param[in] linux_process
  ///     The native (CPU) process being debugged.
  static llvm::Error
  InitiateSafeAttach(SymbolAddressProvider get_symbol_address,
                     NativeProcessProtocol &linux_process);

  /// \return breakpoint info for CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED in the
  /// given library. The plugin sets this breakpoint so it is notified when the
  /// driver finishes injecting the debug engine during attach.
  static GPUBreakpointInfo
  GetAttachFinishedBreakpointInfo(llvm::StringRef library_name);

  /// \return true if \a function_name is the attach-procedure-finished symbol,
  /// meaning a breakpoint hit on it came from the late attach path rather than
  /// from launch-style cuInit initialization. The two paths sequence the
  /// driver's IPC flag differently, so they must be told apart.
  static bool IsAttachFinishedBreakpoint(llvm::StringRef function_name);

  CUDBGAPI operator->() const { return m_api_up.get(); }

  CUDBGAPI GetRawAPI() const { return m_api_up.get(); }

  /// The CUDA debugger API version this session operates at: the lesser of
  /// this build's compiled version and the live driver's reported version.
  /// Any driver of the same CUDA major release works -- older OR newer than
  /// the compiled header; only the major must match (a different major is
  /// rejected at initialization, see Initialize). Taking the lesser of the
  /// two just means we never request an entry point the running driver
  /// doesn't provide. Carried so it can be handed to `ProcessNVGPU` for
  /// runtime gating of version-specific API calls (see
  /// `ProcessNVGPU::GetAPIVersion`).
  nvgpu::CudbgApiVersion GetAPIVersion() const { return m_api_version; }

  static GPUBreakpointInfo
  GetInitializationBreakpointInfo(llvm::StringRef library_name);

private:
  CUDADebuggerAPI(CUDBGAPI api, nvgpu::CudbgApiVersion api_version)
      : m_api_up(api, CUDBGAPIDeleter), m_api_version(api_version) {}

  std::unique_ptr<const CUDBGAPI_st, decltype(&CUDBGAPIDeleter)> m_api_up;
  nvgpu::CudbgApiVersion m_api_version;

  static llvm::Expected<CUDADebuggerAPI>
  InitializeImpl(SymbolAddressProvider get_symbol_address,
                 llvm::StringRef libcuda_library_name,
                 NativeProcessProtocol &linux_process,
                 InitContext init_context);
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
