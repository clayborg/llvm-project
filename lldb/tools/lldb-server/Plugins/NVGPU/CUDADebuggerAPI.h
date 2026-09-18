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

/// Resolves the load address of a named symbol in the native (CPU) process,
/// returning std::nullopt when it could not be resolved.
///
/// The launch path takes these from a breakpoint hit, where the LLDB client has
/// already resolved them; late attach resolves them server-side (see
/// ResolveInferiorAttachSymbols). This callback is what lets the rest of the
/// code ignore the difference.
///
/// Every function below taking one of these also takes the native (CPU)
/// process the addresses belong to.
using SymbolAddressProvider =
    llvm::function_ref<std::optional<uint64_t>(llvm::StringRef)>;

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The names of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *LIBCUDA_LIBRARY_NAME = "libcuda.so";
  static constexpr const char *LIBCUDA_LIBRARY_NAME_ALT = "libcuda.so.1";

  /// Written to the attach-procedure FD to ask the driver to inject the debug
  /// engine. Any value will do; this is the one cuda-gdb uses.
  static constexpr uint8_t ATTACH_PROCEDURE_MAGIC_BYTE = 0xAB;

  /// Assumed size of the driver's fixed CUDBG_INJECTION_PATH buffer, including
  /// the trailing NUL. Longer paths are rejected rather than truncated, so a
  /// hostile value cannot overrun the inferior's buffer or our own.
  static constexpr size_t CUDBG_INJECTION_PATH_MAX_SIZE = 4096;

  /// Which path is bringing the API up. They differ only in when the driver's
  /// IPC flag is published; see SetIpcFlag.
  enum class InitContext {
    /// cuInit-style initialization, which publishes the IPC flag itself.
    eLaunch,
    /// Late attach, which leaves the IPC flag to the caller.
    eLateAttach,
  };

  static llvm::Expected<CUDADebuggerAPI>
  Initialize(SymbolAddressProvider get_symbol_address,
             llvm::StringRef libcuda_library_name,
             NativeProcessProtocol &linux_process, InitContext init_context);

  /// Publish (or clear) CUDBG_IPC_FLAG_NAME, the master "an API client is
  /// ready, emit callbacks" flag.
  ///
  /// The driver emits no callbacks until this is set, and the order matters. It
  /// is written last during initialization, so the driver never sees a ready
  /// client whose PID, revision, session and capabilities have not landed. On
  /// the attach path it additionally stays clear until the attach procedure has
  /// finished and the event callback is registered, so no notification can
  /// arrive before anything can observe it.
  static llvm::Error SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                NativeProcessProtocol &linux_process,
                                bool enabled);

  /// \return the native-process symbol names needed for the late attach
  /// handshake.
  static std::vector<std::string> GetAttachSymbolNames();

  /// Resolve the handshake symbols from the inferior without a gdb-remote
  /// round-trip, which during attach stop processing would corrupt the
  /// in-flight continue cycle. Locates libcuda via /proc/<pid>/maps and maps
  /// each st_value from its on-disk dynamic symbol table onto the load base.
  static llvm::Expected<llvm::StringMap<uint64_t>>
  ResolveInferiorAttachSymbols(NativeProcessProtocol &linux_process);

  /// Like ResolveInferiorAttachSymbols, plus the OPTIONAL
  /// CUDBG_DEBUGGER_INITIALIZED flag the detach path resets. That one is
  /// omitted rather than fatal when libcuda does not export it.
  static llvm::Expected<llvm::StringMap<uint64_t>>
  ResolveInferiorDetachSymbols(NativeProcessProtocol &linux_process);

  /// Clear the requested capabilities, CUDBG_DEBUGGER_INITIALIZED (when
  /// present) and the IPC flag, so a later debugger re-negotiates from scratch.
  /// Must run only after the driver has finished its own detach cleanup, which
  /// depends on the IPC flag still being set.
  static llvm::Error
  ResetDetachSymbols(SymbolAddressProvider get_symbol_address,
                     NativeProcessProtocol &linux_process);

  /// \return true if CUDBG_ATTACH_HANDLER_AVAILABLE is set. False means CUDA is
  /// present but cannot service an attach; an error means the symbols could not
  /// be read.
  static llvm::Expected<bool>
  IsLateAttachSupported(SymbolAddressProvider get_symbol_address,
                        NativeProcessProtocol &linux_process);

  /// Read CUDBG_RESUME_FOR_ATTACH_DETACH, non-zero meaning the application must
  /// keep running for the driver to finish attaching or detaching.
  ///
  /// This is a flag word, not a boolean, and requestCleanupOnDetach is
  /// documented to take it verbatim, so it is returned raw; callers wanting the
  /// yes/no answer compare against zero themselves.
  static llvm::Expected<uint32_t>
  ReadResumeForAttachDetach(SymbolAddressProvider get_symbol_address,
                            NativeProcessProtocol &linux_process);

  /// Publish the client handshake globals and write the magic byte to the
  /// driver's attach-procedure FD, asking it to inject the debug engine at a
  /// point of its choosing. It reports completion by calling
  /// CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED, where the plugin has a breakpoint.
  static llvm::Error
  InitiateSafeAttach(SymbolAddressProvider get_symbol_address,
                     NativeProcessProtocol &linux_process);

  static GPUBreakpointInfo
  GetInitializationBreakpointInfo(llvm::StringRef library_name);

  static GPUBreakpointInfo
  GetAttachFinishedBreakpointInfo(llvm::StringRef library_name);

  /// \return true if a breakpoint on \a function_name came from the late attach
  /// path rather than launch-style cuInit initialization. The two sequence the
  /// IPC flag differently, so they must be told apart.
  static bool IsAttachFinishedBreakpoint(llvm::StringRef function_name);

  CUDBGAPI operator->() const { return m_api_up.get(); }

  CUDBGAPI GetRawAPI() const { return m_api_up.get(); }

  /// The version this session operates at: the lesser of the compiled and the
  /// driver's own, so we never request an entry point the driver lacks. Any
  /// driver of the same CUDA major release works; a different major is rejected
  /// in Initialize. Handed to ProcessNVGPU to gate version-specific calls.
  nvgpu::CudbgApiVersion GetAPIVersion() const { return m_api_version; }

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
