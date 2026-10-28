//===-- ThreadCoords.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared CUDA thread coordinate value types.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_THREADCOORDS_H
#define LLDB_UTILITY_NVGPU_THREADCOORDS_H

#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>

namespace lldb_private::nvgpu {

/// A three-dimensional unsigned coordinate.
struct Dim3u {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

template <typename T> Dim3u ToDim3u(const T &dim) {
  return {dim.x, dim.y, dim.z};
}

/// CUDA cluster index and dimensions.
struct ClusterInfo {
  Dim3u idx;
  Dim3u dim;
};

/// The physical location of a CUDA thread. An absent field is unknown.
struct PhysicalCoords {
  std::optional<uint32_t> device;
  std::optional<uint32_t> sm;
  std::optional<uint32_t> warp;
  std::optional<uint32_t> lane;
};

/// The logical location of a CUDA thread. An absent field is unknown.
struct LogicalCoords {
  std::optional<uint64_t> context_id;
  std::optional<uint64_t> grid_id;
  std::optional<ClusterInfo> cluster;
  std::optional<Dim3u> block_idx;
  std::optional<Dim3u> thread_idx;
};

/// The available fields in a CUDA thread coordinate snapshot.
///
/// An absent optional is always unknown; zero and false are known values.
struct ThreadCoords {
  PhysicalCoords physical;
  LogicalCoords logical;
  std::optional<bool> is_active;
};

/// The thread extended-info key whose value is the coordinate object.
inline constexpr llvm::StringLiteral g_thread_coords_key = "cuda";

/// Serialize the available fields in a CUDA thread coordinate snapshot.
///
/// The returned value contains the contents of the "cuda" object, without the
/// outer wrapper. Its fields are shared by live jThreadExtendedInfo and
/// core-file thread extended info.
///
/// Unknown fields are omitted. Cluster index and dimensions are emitted
/// together when cluster information is present, without classifying or
/// validating their values.
llvm::json::Value toJSON(const ThreadCoords &coords);

/// Serialize the available fields as the contents of a "cuda" dictionary.
StructuredData::ObjectSP toStructuredData(const ThreadCoords &coords);

/// Decode a CUDA coordinate object; missing fields are unknown.
/// Returns an error for a non-dictionary, or an error that names every
/// malformed field.
llvm::Expected<ThreadCoords>
FromStructuredData(const StructuredData::Object &cuda);

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_THREADCOORDS_H
