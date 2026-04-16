//===-- ThreadName.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/ThreadName.h"
#include "llvm/Support/FormatVariadic.h"

std::string lldb_private::nvgpu::FormatThreadName(
    uint32_t blockIdxX, uint32_t blockIdxY, uint32_t blockIdxZ,
    uint32_t threadIdxX, uint32_t threadIdxY, uint32_t threadIdxZ) {
  return llvm::formatv(
      "blockIdx(x={0} y={1} z={2}) threadIdx(x={3} y={4} z={5})", blockIdxX,
      blockIdxY, blockIdxZ, threadIdxX, threadIdxY, threadIdxZ);
}
