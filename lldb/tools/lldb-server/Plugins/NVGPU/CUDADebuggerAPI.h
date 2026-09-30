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
/// returning std::nullopt when it could not be resolved. The values come from
/// the client, either with a breakpoint hit or through qSymbol.
using SymbolAddressProvider =
    llvm::function_ref<std::optional<uint64_t>(llvm::StringRef)>;

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The names of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *LIBCUDA_LIBRARY_NAME = "libcuda.so";
  static constexpr const char *LIBCUDA_LIBRARY_NAME_ALT = "libcuda.so.1";

  /// Which path is bringing the API up. They differ only in when the driver's
  /// IPC flag is published; see SetIpcFlag.
  enum class InitContext { eLaunch, eLateAttach };

  static llvm::Expected<CUDADebuggerAPI>
  Initialize(SymbolAddressProvider get_symbol_address,
             llvm::StringRef libcuda_library_name,
             NativeProcessProtocol &linux_process, InitContext init_context);

  /// Publish CUDBG_IPC_FLAG_NAME, the master "an API client is ready, emit
  /// callbacks" flag. Written last during initialization so the driver never
  /// sees a ready client whose PID, revision, session and capabilities have not
  /// landed; the attach path additionally holds it back until the attach
  /// procedure has finished and the event callback exists.
  static llvm::Error SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                NativeProcessProtocol &linux_process);

  /// The symbols InitiateSafeAttach needs. They are needed before any
  /// breakpoint has been hit, so they are requested from the client through
  /// qSymbol instead.
  static std::vector<std::string> GetSafeAttachSymbolNames();

  /// \return true if libcuda is in the inferior's loaded library list. Without
  /// it CUDA cannot have been initialized yet, so the launch-style
  /// initialization breakpoints cover the attach.
  static llvm::Expected<bool>
  IsLibcudaLoaded(NativeProcessProtocol &linux_process);

  /// The writes that clear the requested capabilities,
  /// CUDBG_DEBUGGER_INITIALIZED and the IPC flag, so a later debugger
  /// re-negotiates. Valid only after the driver has finished its own detach
  /// cleanup, which needs the IPC flag still set, and with the process
  /// stopped, since ptrace refuses to write to a running tracee.
  static std::vector<GPUMemoryWrite>
  GetDetachResetWrites(SymbolAddressProvider get_symbol_address);

  /// Read CUDBG_RESUME_FOR_ATTACH_DETACH, non-zero meaning the application must
  /// keep running for the driver to finish. Returned raw because it is a flag
  /// word that requestCleanupOnDetach takes verbatim.
  static llvm::Expected<uint32_t>
  ReadResumeForAttachDetach(SymbolAddressProvider get_symbol_address,
                            NativeProcessProtocol &linux_process);

  /// Publish the client handshake globals and write the magic byte to the
  /// driver's attach-procedure FD, asking it to inject the debug engine at a
  /// point of its choosing. It reports completion by calling
  /// CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED. The globals are written with
  /// ptrace, so the process must be stopped.
  ///
  /// \return false, having written nothing, if the driver has not published
  /// the FD yet (it is still -1 until the driver finishes initializing).
  static llvm::Expected<bool>
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

  /// The lesser of the compiled and the driver's own version, so we never
  /// request an entry point the driver lacks. Initialize rejects a driver from
  /// a different CUDA major release outright.
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
