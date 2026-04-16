//===-- CUDAException.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared CUDA exception utilities used by both lldb-server (live GPU
/// debugging) and liblldb (GPU corefile debugging).
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_CUDAEXCEPTION_H
#define LLDB_UTILITY_NVGPU_CUDAEXCEPTION_H

#include "cudadebugger.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace lldb_private {

/// Convert a CUDBGException_t value to a human-readable string.
///
/// \param[in] exception
///     The CUDA exception type.
///
/// \return
///     A string describing the exception, or "Device Unknown Exception"
///     for unrecognized values.
llvm::StringRef CUDAExceptionToString(CUDBGException_t exception);

/// Represents information about an exception that occurred during CUDA kernel
/// execution.
struct CUDAExceptionInfo {
  /// The type of CUDA exception that occurred.
  CUDBGException_t exception;

  /// The program counter address where the exception occurred, if available.
  std::optional<uint64_t> errorPC;

  /// Construct exception information.
  ///
  /// \param[in] exception
  ///     The CUDA exception type that occurred. Must not be
  ///     CUDBG_EXCEPTION_NONE.
  ///
  /// \param[in] errorPC
  ///     Optional program counter where the exception occurred.
  CUDAExceptionInfo(CUDBGException_t exception,
                    std::optional<uint64_t> errorPC);

  /// Convert the exception information to a human-readable string.
  ///
  /// \return
  ///     A string representation including the exception name and
  ///     optionally the error PC.
  std::string ToString() const;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_NVGPU_CUDAEXCEPTION_H
