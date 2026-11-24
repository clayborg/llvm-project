//===-- ThreadCoords.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/ThreadCoords.h"

#include "llvm/ADT/StringRef.h"

#include <type_traits>
#include <utility>

using namespace lldb_private;

namespace {

/// The schema of the "cuda" object: each wire key and the member it holds.
/// `map` receives each key and the member's std::optional; one overload per
/// value type does the conversion.
template <typename Mapper>
void MapThreadCoords(const nvgpu::ThreadCoords &coords, Mapper &&map) {
  map("device", coords.physical.device);
  map("sm", coords.physical.sm);
  map("warp", coords.physical.warp);
  map("lane", coords.physical.lane);
  map("context_id", coords.logical.context_id);
  map("grid_id", coords.logical.grid_id);

  // A cluster travels as two keys that are present together.
  std::optional<nvgpu::Dim3u> cluster_idx;
  std::optional<nvgpu::Dim3u> cluster_dim;
  if (coords.logical.cluster) {
    cluster_idx = coords.logical.cluster->idx;
    cluster_dim = coords.logical.cluster->dim;
  }
  map("cluster_idx", cluster_idx);
  map("cluster_dim", cluster_dim);

  map("block_idx", coords.logical.block_idx);
  map("thread_idx", coords.logical.thread_idx);
  map("is_active", coords.is_active);
}

template <typename T> llvm::json::Value ToJSONValue(const T &value) {
  return llvm::json::Value(value);
}

llvm::json::Value ToJSONValue(const nvgpu::Dim3u &dim) {
  return llvm::json::Value(
      llvm::json::Object{{"x", dim.x}, {"y", dim.y}, {"z", dim.z}});
}

template <typename T> StructuredData::ObjectSP MakeStructuredValue(T value) {
  static_assert(std::is_unsigned_v<T>);
  return std::make_shared<StructuredData::UnsignedInteger>(
      static_cast<uint64_t>(value));
}

StructuredData::ObjectSP MakeStructuredValue(bool value) {
  return std::make_shared<StructuredData::Boolean>(value);
}

StructuredData::ObjectSP MakeStructuredValue(const nvgpu::Dim3u &value) {
  auto value_sp = std::make_shared<StructuredData::Dictionary>();
  value_sp->AddIntegerItem("x", value.x);
  value_sp->AddIntegerItem("y", value.y);
  value_sp->AddIntegerItem("z", value.z);
  return value_sp;
}

} // namespace

llvm::json::Value nvgpu::toJSON(const ThreadCoords &coords) {
  llvm::json::Object object;
  MapThreadCoords(coords, [&](llvm::StringRef key, const auto &value) {
    if (value)
      object.insert({key, ToJSONValue(*value)});
  });
  return llvm::json::Value(std::move(object));
}

StructuredData::ObjectSP nvgpu::toStructuredData(const ThreadCoords &coords) {
  auto object_sp = std::make_shared<StructuredData::Dictionary>();
  MapThreadCoords(coords, [&](llvm::StringRef key, const auto &value) {
    if (value)
      object_sp->AddItem(key, MakeStructuredValue(*value));
  });
  return object_sp;
}
