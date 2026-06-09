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
/// compatibility. These macros gate version-specific symbols on the compiled
/// header's `CUDBG_API_VERSION_*` values; `CUDBG_API_VERSION_REVISION` is a
/// monotonic build counter and is the reliable in-major discriminator.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_CUDADEBUGGERVERSION_H
#define LLDB_UTILITY_NVGPU_CUDADEBUGGERVERSION_H

#include "cudadebugger.h"

#include <cstdint>
#include <tuple>

/// The CUDA major release this LLDB build targets. Cross-major-release
/// compatibility is explicitly out of scope: a build works against any
/// driver/coredump within this major only. It is a build constant rather
/// than something derived from the header so that an accidental cross-major
/// build is caught by the static_assert below instead of producing cryptic
/// missing-symbol errors.
#define LLDB_NVGPU_CUDA_TARGET_MAJOR 13

static_assert(CUDBG_API_VERSION_MAJOR == LLDB_NVGPU_CUDA_TARGET_MAJOR,
              "LLDB's NVGPU plugins support building against a single CUDA "
              "debugger-API major release only (see "
              "LLDB_NVGPU_CUDA_TARGET_MAJOR). The cudadebugger.h on the "
              "include path is from a different major release.");

/// True if the compiled CUDA debugger API revision is at least `revision`.
/// Because cross-major builds are rejected above and
/// `CUDBG_API_VERSION_REVISION` is monotonic within a major, the revision
/// alone is a sufficient discriminator for version-specific symbols.
#define LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(revision)                            \
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

#endif // LLDB_UTILITY_NVGPU_CUDADEBUGGERVERSION_H
