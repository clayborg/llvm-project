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

using namespace lldb_private;

namespace {

template <typename Visitor>
void VisitThreadCoords(const nvgpu::ThreadCoords &coords, Visitor &&visit) {
  if (coords.physical.device)
    visit(llvm::StringRef("device"), *coords.physical.device);
  if (coords.physical.sm)
    visit(llvm::StringRef("sm"), *coords.physical.sm);
  if (coords.physical.warp)
    visit(llvm::StringRef("warp"), *coords.physical.warp);
  if (coords.physical.lane)
    visit(llvm::StringRef("lane"), *coords.physical.lane);
  if (coords.logical.context_id)
    visit(llvm::StringRef("context_id"), *coords.logical.context_id);
  if (coords.logical.grid_id)
    visit(llvm::StringRef("grid_id"), *coords.logical.grid_id);

  if (coords.logical.cluster) {
    visit(llvm::StringRef("cluster_idx"), coords.logical.cluster->idx);
    visit(llvm::StringRef("cluster_dim"), coords.logical.cluster->dim);
  }
  if (coords.logical.block_idx)
    visit(llvm::StringRef("block_idx"), *coords.logical.block_idx);
  if (coords.logical.thread_idx)
    visit(llvm::StringRef("thread_idx"), *coords.logical.thread_idx);
  if (coords.is_active)
    visit(llvm::StringRef("is_active"), *coords.is_active);
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

  VisitThreadCoords(coords, [&](llvm::StringRef name, const auto &value) {
    object.insert({name, ToJSONValue(value)});
  });

  return llvm::json::Value(std::move(object));
}

StructuredData::ObjectSP nvgpu::toStructuredData(const ThreadCoords &coords) {
  auto object_sp = std::make_shared<StructuredData::Dictionary>();

  VisitThreadCoords(coords, [&](llvm::StringRef name, const auto &value) {
    object_sp->AddItem(name, MakeStructuredValue(value));
  });

  return object_sp;
}
