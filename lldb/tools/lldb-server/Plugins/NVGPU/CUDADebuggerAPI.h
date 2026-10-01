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
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/NVGPU/AttachHandshake.h"
// The runtime CudbgApiVersion type, used to pick an API version the attached
// driver supports.
#include "lldb/Utility/NVGPU/CUDADebuggerAPIVersion.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <memory>

namespace lldb_private::process_gdb_remote {
class GDBRemoteCommunicationServerLLGS;
} // namespace lldb_private::process_gdb_remote

namespace lldb_private::lldb_server {

/// Custom deleter for CUDBGAPI.
void CUDBGAPIDeleter(CUDBGAPI api);

/// Resolves the load address of a named symbol in the native (CPU) process,
/// returning std::nullopt when it could not be resolved. The values come from
/// the client with a breakpoint hit.
using SymbolAddressProvider =
    llvm::function_ref<std::optional<uint64_t>(llvm::StringRef)>;

/// The server debugging the native (CPU) process. The handshake reads and
/// writes that process only through this server's process actions, on its
/// MainLoop thread.
using HostServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;

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

  /// Initialize the CUDA debugger API.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the symbols of the GPU plugin breakpoint that triggered the
  ///     initialization.
  ///
  /// \param[in] libcuda_library_name
  ///     The name of the CUDA library that contains the CUDA debugger API.
  ///
  /// \param[in] host_server
  ///     The server debugging the process that spawned the GPU process.
  ///
  /// \param[in] init_context
  ///     Whether the API comes up on launch or at the end of a late attach.
  static llvm::Expected<CUDADebuggerAPI>
  Initialize(SymbolAddressProvider get_symbol_address,
             llvm::StringRef libcuda_library_name, HostServer &host_server,
             InitContext init_context);

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

  static GPUBreakpointInfo
  GetAttachFinishedBreakpointInfo(llvm::StringRef library_name);

  /// \return
  ///     true if a breakpoint on \a function_name came from the late attach
  ///     path rather than launch-style cuInit initialization. The two sequence
  ///     the IPC flag differently, so they must be told apart.
  static bool IsAttachFinishedBreakpoint(llvm::StringRef function_name);

  /// Publish CUDBG_IPC_FLAG_NAME, the master "an API client is ready, emit
  /// callbacks" flag. Written last during initialization so the driver never
  /// sees a ready client whose PID, revision, session and capabilities have not
  /// landed; the attach path additionally holds it back until the attach
  /// procedure has finished and the event callback exists.
  static llvm::Error SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                HostServer &host_server);

  /// The values the client writes into the application's libcuda before it
  /// asks the driver to attach to a running process. Only this server knows
  /// them: its own pid and environment, and the capabilities it requires.
  static nvgpu::AttachHandshake GetAttachHandshake();

  /// Read CUDBG_RESUME_FOR_ATTACH_DETACH, non-zero meaning the application must
  /// keep running for the driver to finish. Returned raw because it is a flag
  /// word that requestCleanupOnDetach takes verbatim.
  static llvm::Expected<uint32_t>
  ReadResumeForAttachDetach(SymbolAddressProvider get_symbol_address,
                            HostServer &host_server);

private:
  CUDADebuggerAPI(CUDBGAPI api, nvgpu::CudbgApiVersion api_version)
      : m_api_up(api, CUDBGAPIDeleter), m_api_version(api_version) {}

  std::unique_ptr<const CUDBGAPI_st, decltype(&CUDBGAPIDeleter)> m_api_up;
  nvgpu::CudbgApiVersion m_api_version;

  static llvm::Expected<CUDADebuggerAPI>
  InitializeImpl(SymbolAddressProvider get_symbol_address,
                 llvm::StringRef libcuda_library_name, HostServer &host_server,
                 InitContext init_context);
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
