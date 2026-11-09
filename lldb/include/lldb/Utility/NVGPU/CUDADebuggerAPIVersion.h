//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Version-compatibility helpers for the CUDA debugger API
/// (`cudadebugger.h`), shared by the live lldb-server NVGPU plugin and the
/// NVGPU corefile reader.
///
/// LLDB's NVGPU plugins are built against a single CUDA debugger-API header
/// but must work against any CUDA driver -- or read any coredump -- within
/// that same CUDA *major* release. There is no cross-major-release
/// compatibility. The vendored header is the default; compile-time revision
/// checks allow building against older headers within that major. The driver
/// varies at run time, which `CudbgApiVersion` below is for.
///
/// Which major that is, and the check that enforces it, live in
/// `CUDADebuggerAPIVersion.cpp`.
///
/// The header is the copy vendored in `lldb/third-party/cuda`, unless
/// `NVGPU_DEBUGGER_INCLUDE_DIR` points elsewhere; `SetupCUDA.cmake` makes
/// that choice.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_CUDADEBUGGERAPIVERSION_H
#define LLDB_UTILITY_NVGPU_CUDADEBUGGERAPIVERSION_H

#include "cudadebugger.h"

#include <cstdint>
#include <tuple>

/// True if the compiled CUDA debugger API revision is at least `revision`.
/// CUDADebuggerAPIVersion.cpp rejects cross-major builds, and the revision
/// increases monotonically within that major.
#define LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(revision)                              \
  (CUDBG_API_VERSION_REVISION >= (revision))

namespace lldb_private::nvgpu {

/// A CUDA debugger API version (major.minor.revision). The live plugin
/// discovers the driver's version with `cudbgGetAPIVersion` and picks the
/// revision to use (the lesser of the driver's and the compiled version), so
/// it needs a value comparable at runtime.
struct CudbgApiVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t revision = 0;

  /// The version this build was compiled against.
  static CudbgApiVersion Compiled() {
    return {CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
            CUDBG_API_VERSION_REVISION};
  }

  /// Lexicographic comparison over (major, minor, revision), used to pick the
  /// lesser of the compiled and driver versions.
  bool operator<(const CudbgApiVersion &rhs) const {
    return std::tie(major, minor, revision) <
           std::tie(rhs.major, rhs.minor, rhs.revision);
  }

  /// True if this version is at least (maj, min, rev). Use this to gate calls
  /// to API entry points introduced after the major's baseline: a driver
  /// older than the compiled header returns an API table that does not
  /// contain newer (appended) entry points, so calling -- or even reading the
  /// function pointer of -- such an entry is undefined unless the in-use API
  /// version guarantees it exists. Gate first, then call.
  bool AtLeast(uint32_t maj, uint32_t min, uint32_t rev) const {
    return !(*this < CudbgApiVersion{maj, min, rev});
  }
};

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_CUDADEBUGGERAPIVERSION_H
