//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared thread display name formatting for NVIDIA GPU threads.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_THREADNAME_H
#define LLDB_UTILITY_NVGPU_THREADNAME_H

#include <cstdint>
#include <string>

namespace lldb_private::nvgpu {

/// Format a GPU thread name from block and thread coordinates.
///
/// \return
///     A string like "blockIdx(x=0 y=0 z=0) threadIdx(x=3 y=0 z=0)".
std::string FormatThreadName(uint32_t blockIdxX, uint32_t blockIdxY,
                             uint32_t blockIdxZ, uint32_t threadIdxX,
                             uint32_t threadIdxY, uint32_t threadIdxZ);

/// Format the name of a thread standing in for an SM-level exception that
/// has no surviving kernel context. Every warp on the SM had exited by the
/// time the state was captured, so there are no block / thread coordinates
/// to name it by.
///
/// \param[in] device_idx
///     Index of the device the SM belongs to.
///
/// \param[in] sm_id
///     Hardware SM id, as reported by the debug API.
///
/// \return
///     A string like "device 0 SM 3 (no kernel context)".
std::string FormatSMExceptionThreadName(uint32_t device_idx, uint32_t sm_id);

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_THREADNAME_H
