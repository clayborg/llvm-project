//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The build-time check that the `cudadebugger.h` the compiler reached is from
/// the CUDA major release this build targets.
///
/// A translation unit of its own, rather than a line in
/// `CUDADebuggerAPIVersion.h`, so it is compiled once instead of in every file
/// that includes the header. This one, rather than one in lldb-server, because
/// lldbUtilityNVGPU is a link dependency of both lldbServerPluginNVGPU and
/// lldbPluginProcessNVGPUCore  so anything that can reach the CUDA debugger API
/// compiles this.
///
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/CUDADebuggerAPIVersion.h"

/// The CUDA major release this LLDB build targets. A build constant rather
/// than something derived from the header, so that an accidental cross-major
/// build is caught by the assert below rather than producing cryptic
/// missing-symbol errors. Bump it when moving to the next CUDA major.
#define LLDB_NVGPU_CUDA_TARGET_MAJOR 13

static_assert(CUDBG_API_VERSION_MAJOR == LLDB_NVGPU_CUDA_TARGET_MAJOR,
              "LLDB's NVGPU plugins support building against a single CUDA "
              "debugger-API major release only (see "
              "LLDB_NVGPU_CUDA_TARGET_MAJOR). The cudadebugger.h on the "
              "include path is from a different major release.");
