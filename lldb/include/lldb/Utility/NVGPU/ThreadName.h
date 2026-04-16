//===-- ThreadName.h --------------------------------------------*- C++ -*-===//
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

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_THREADNAME_H
