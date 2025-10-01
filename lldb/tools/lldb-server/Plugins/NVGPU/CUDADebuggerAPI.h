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

#include <memory>

namespace lldb_private::lldb_server {

/// Custom deleter for CUDBGAPI.
void CUDBGAPIDeleter(CUDBGAPI api);

/// RAII wrapper class for CUDA debugger API instances.
/// The API methods are accessed through the -> operator.
class CUDADebuggerAPI {
public:
  /// The name of the CUDA library that contains the CUDA debugger API.
  static constexpr const char *CUDA_API_LIBRARY_NAME = "libcuda.so.1";

  /// Initialize the CUDA debugger API.
  ///
  /// \param bp_args The GPU plugin breakpoint that triggered the
  /// initialization.
  /// \param linux_process The Linux process that spawned the GPU process.
  static llvm::Expected<CUDADebuggerAPI>
  Initialize(const GPUPluginBreakpointHitArgs &bp_args,
             NativeProcessProtocol &linux_process);

  CUDBGAPI operator->() const { return m_api_up.get(); }

  CUDBGAPI GetRawAPI() const { return m_api_up.get(); }

  static GPUBreakpointInfo GetInitializationBreakpointInfo();

private:
  CUDADebuggerAPI(CUDBGAPI api) : m_api_up(api, CUDBGAPIDeleter) {}

  std::unique_ptr<const CUDBGAPI_st, decltype(&CUDBGAPIDeleter)> m_api_up;

  static llvm::Expected<CUDADebuggerAPI>
  InitializeImpl(const GPUPluginBreakpointHitArgs &bp_args,
                 NativeProcessProtocol &linux_process);
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_CUDADDEBUGGERAPI_H
