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
// Shared single-major version policy: the LLDB_NVGPU_CUDBG_API_* compile-time
// macros, the single-major static_assert, and the runtime CudbgApiVersion
// type used to pick a driver-compatible API version.
#include "lldb/Utility/NVGPU/CUDADebuggerVersion.h"

#include <memory>

namespace lldb_private::lldb_server {

/// Custom deleter for CUDBGAPI.
void CUDBGAPIDeleter(CUDBGAPI api);

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The names of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *LIBCUDA_LIBRARY_NAME = "libcuda.so";
  static constexpr const char *LIBCUDA_LIBRARY_NAME_ALT = "libcuda.so.1";

  /// Initialize the CUDA debugger API.
  ///
  /// \param bp_args The GPU plugin breakpoint that triggered the
  /// initialization.
  /// \param libcuda_library_name The name of the CUDA library that contains the
  /// CUDA debugger API.
  /// \param linux_process The Linux process that spawned the GPU process.
  static llvm::Expected<CUDADebuggerAPI>
  Initialize(const GPUPluginBreakpointHitArgs &bp_args,
             llvm::StringRef libcuda_library_name,
             NativeProcessProtocol &linux_process);

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
  InitializeImpl(const GPUPluginBreakpointHitArgs &bp_args,
                 llvm::StringRef libcuda_library_name,
                 NativeProcessProtocol &linux_process);
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
