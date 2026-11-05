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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lldb_private::lldb_server {

/// Custom deleter for CUDBGAPI.
void CUDBGAPIDeleter(CUDBGAPI api);

/// Returns the load address of a symbol of the debugged process by name, or
/// std::nullopt if it did not resolve. lldb sends the addresses with a
/// breakpoint hit or a jGPUPluginSymbols packet.
using SymbolAddressProvider =
    llvm::function_ref<std::optional<uint64_t>(llvm::StringRef)>;

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The names of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *LIBCUDA_LIBRARY_NAME = "libcuda.so";
  static constexpr const char *LIBCUDA_LIBRARY_NAME_ALT = "libcuda.so.1";

  /// Initialize the CUDA debugger API. This writes the API client's identity
  /// into the debugged process, but not the IPC flag; see SetIpcFlag.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the symbols of the breakpoint that triggered the
  ///     initialization.
  ///
  /// \param[in] libcuda_library_name
  ///     The name of the CUDA library that contains the CUDA debugger API.
  ///
  /// \param[in] process
  ///     The debugged process. Only the thread that debugs it may use it, as
  ///     the breakpoint that triggered the initialization does.
  static llvm::Expected<CUDADebuggerAPI>
  Initialize(SymbolAddressProvider get_symbol_address,
             llvm::StringRef libcuda_library_name,
             NativeProcessProtocol &process);

  /// Set the IPC flag in the debugged process, which tells its CUDA driver
  /// that a debugger is ready to serve it. A driver that sees the flag can
  /// wait for this debugger indefinitely, so set it only after every other
  /// initialization step has succeeded. If a step fails first, the process
  /// runs on without GPU debugging.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the address of the flag.
  ///
  /// \param[in] process
  ///     The debugged process.
  static llvm::Error SetIpcFlag(SymbolAddressProvider get_symbol_address,
                                NativeProcessProtocol &process);

  /// Clear the IPC flag that SetIpcFlag set, when the driver did not pick it
  /// up.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the address of the flag.
  ///
  /// \param[in] process
  ///     The debugged process.
  static llvm::Error ClearIpcFlag(SymbolAddressProvider get_symbol_address,
                                  NativeProcessProtocol &process);

  /// Read CUDBG_DEBUGGER_INITIALIZED, which the driver's debug engine sets
  /// once it has initialized. The driver initializes it in the call to cuInit
  /// that initializes the driver, if the IPC flag is set by then.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the symbols of the breakpoint that GetCuInitBreakpointInfo
  ///     describes.
  ///
  /// \param[in] process
  ///     The debugged process. Only the thread that debugs it may use it.
  static llvm::Expected<bool>
  ReadDebuggerInitialized(SymbolAddressProvider get_symbol_address,
                          NativeProcessProtocol &process);

  /// \return
  ///     The names of the libcuda symbols that RequestLateAttach uses. It
  ///     needs CUDBG_INJECTION_PATH's only when that variable is set.
  static llvm::ArrayRef<llvm::StringLiteral> GetLateAttachSymbolNames();

  /// Ask the CUDA driver in \a process to attach to this debugger, if it can,
  /// as it can once the application has a CUDA context: write who is
  /// attaching, with the debugger API revision that Initialize negotiates,
  /// into libcuda, but not the IPC flag, then a byte to the driver's attach
  /// procedure descriptor. The driver attaches once the process runs, and then
  /// calls the function that GetAttachFinishedBreakpointInfo names.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the symbols that GetLateAttachSymbolNames names.
  ///
  /// \param[in] process
  ///     The process to attach to. Only the thread that debugs it may use it.
  ///
  /// \return
  ///     Whether the driver was asked to attach, which it is not while the
  ///     application has no CUDA context. An error if a symbol is missing,
  ///     which it is from a libcuda that predates the driver's attach
  ///     procedure, if the driver's debugger API is from another CUDA major
  ///     release, or if the process cannot be read or written.
  static llvm::Expected<bool>
  RequestLateAttach(SymbolAddressProvider get_symbol_address,
                    NativeProcessProtocol &process);

  /// Read CUDBG_RESUME_FOR_ATTACH_DETACH, which is non-zero while the driver
  /// needs the application to run to finish an attach.
  ///
  /// \param[in] get_symbol_address
  ///     Resolves the symbols of the breakpoint that
  ///     GetAttachFinishedBreakpointInfo describes.
  ///
  /// \param[in] process
  ///     The debugged process. Only the thread that debugs it may use it.
  static llvm::Expected<uint32_t>
  ReadResumeForAttachDetach(SymbolAddressProvider get_symbol_address,
                            NativeProcessProtocol &process);

  CUDBGAPI operator->() const { return m_api_up.get(); }

  CUDBGAPI GetRawAPI() const { return m_api_up.get(); }

  /// Let go of the API without finalizing it, for when the driver would crash
  /// in finalize.
  void Abandon() { m_api_up.release(); }

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

  /// The functions that the breakpoints these functions describe are on.
  enum class BreakpointFunction {
    /// cuInit: GetCuInitBreakpointInfo.
    CuInit,
    /// The driver's CUDBG_PRE_INIT hook: GetPreInitBreakpointInfo.
    PreInit,
    /// CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED: GetAttachFinishedBreakpointInfo.
    AttachFinished,
  };

  /// \return
  ///     A breakpoint at the first instruction of cuInit in \a library_name,
  ///     with the symbols that the debugger API's initialization needs, the
  ///     address of cuInit itself, of CUDBG_PRE_INIT if libcuda has it, and of
  ///     CUDBG_DEBUGGER_INITIALIZED.
  static GPUBreakpointInfo
  GetCuInitBreakpointInfo(llvm::StringRef library_name);

  /// \return
  ///     A breakpoint on CUDBG_PRE_INIT in \a library_name, which every call
  ///     to cuInit calls before the driver initializes, with the symbols that
  ///     the debugger API's initialization needs. The libcuda of CUDA 13.0
  ///     lacks the hook, so the breakpoint does not resolve there.
  static GPUBreakpointInfo
  GetPreInitBreakpointInfo(llvm::StringRef library_name);

  /// \return
  ///     A breakpoint on the function that the driver calls once it has
  ///     attached at RequestLateAttach's request, with the symbols that the
  ///     debugger API's initialization needs.
  static GPUBreakpointInfo
  GetAttachFinishedBreakpointInfo(llvm::StringRef library_name);

  /// \return
  ///     Which function a breakpoint on \a function_name is on, if it is one
  ///     of those that BreakpointFunction names.
  static std::optional<BreakpointFunction>
  GetBreakpointFunction(llvm::StringRef function_name);

  /// \return
  ///     The address of cuInit, from the symbols of the breakpoint that
  ///     GetCuInitBreakpointInfo describes.
  static std::optional<uint64_t>
  GetCuInitAddress(SymbolAddressProvider get_symbol_address);

  /// \return
  ///     Whether libcuda has CUDBG_PRE_INIT, from the symbols of the
  ///     breakpoint that GetCuInitBreakpointInfo describes.
  static bool HasPreInit(SymbolAddressProvider get_symbol_address);

private:
  CUDADebuggerAPI(CUDBGAPI api, nvgpu::CudbgApiVersion api_version)
      : m_api_up(api, CUDBGAPIDeleter), m_api_version(api_version) {}

  std::unique_ptr<const CUDBGAPI_st, decltype(&CUDBGAPIDeleter)> m_api_up;
  nvgpu::CudbgApiVersion m_api_version;

  static llvm::Expected<CUDADebuggerAPI>
  InitializeImpl(SymbolAddressProvider get_symbol_address,
                 llvm::StringRef libcuda_library_name,
                 NativeProcessProtocol &process);
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
