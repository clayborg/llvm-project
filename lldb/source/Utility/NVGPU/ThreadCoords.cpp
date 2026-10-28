//===-- ThreadCoords.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/ThreadCoords.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <type_traits>
#include <utility>

using namespace lldb_private;

namespace {

/// The schema of the "cuda" object: each wire key and the member it holds.
/// The encoders pass a const ThreadCoords and the decoder a mutable one, so a
/// field listed here is both written and read. `map` receives each key and the
/// member's std::optional; one overload per value type does the conversion.
template <typename Coords, typename Mapper>
void MapThreadCoords(Coords &coords, Mapper &&map) {
  map("device", coords.physical.device);
  map("sm", coords.physical.sm);
  map("warp", coords.physical.warp);
  map("lane", coords.physical.lane);
  map("context_id", coords.logical.context_id);
  map("grid_id", coords.logical.grid_id);

  // A cluster travels as two keys that are present together. Split it for the
  // mapper and, when decoding, join it back; one key without the other is
  // ignored.
  std::optional<nvgpu::Dim3u> cluster_idx;
  std::optional<nvgpu::Dim3u> cluster_dim;
  if (coords.logical.cluster) {
    cluster_idx = coords.logical.cluster->idx;
    cluster_dim = coords.logical.cluster->dim;
  }
  map("cluster_idx", cluster_idx);
  map("cluster_dim", cluster_dim);
  if constexpr (!std::is_const_v<Coords>) {
    if (cluster_idx && cluster_dim)
      coords.logical.cluster = nvgpu::ClusterInfo{*cluster_idx, *cluster_dim};
  }

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

/// `path` is a Twine so that it is only rendered for an invalid field.
llvm::Error InvalidField(const llvm::Twine &path, const llvm::Twine &reason) {
  return llvm::createStringError("invalid CUDA thread coordinate field 'cuda." +
                                 path + "': " + reason);
}

template <typename UInt>
llvm::Error DecodeValue(StructuredData::Object &object, const llvm::Twine &path,
                        UInt &value) {
  static_assert(std::is_unsigned_v<UInt> && !std::is_same_v<UInt, bool>,
                "DecodeValue requires an unsigned integer other than bool");
  StructuredData::UnsignedInteger *integer = object.GetAsUnsignedInteger();
  if (!integer || !llvm::isUInt<sizeof(UInt) * 8>(integer->GetValue()))
    return InvalidField(path, "expected a " + llvm::Twine(sizeof(UInt) * 8) +
                                  "-bit unsigned integer");
  value = static_cast<UInt>(integer->GetValue());
  return llvm::Error::success();
}

llvm::Error DecodeValue(StructuredData::Object &object, const llvm::Twine &path,
                        bool &value) {
  StructuredData::Boolean *boolean = object.GetAsBoolean();
  if (!boolean)
    return InvalidField(path, "expected a boolean");
  value = boolean->GetValue();
  return llvm::Error::success();
}

llvm::Error DecodeValue(StructuredData::Object &object, const llvm::Twine &path,
                        nvgpu::Dim3u &dim) {
  StructuredData::Dictionary *dictionary = object.GetAsDictionary();
  if (!dictionary)
    return InvalidField(path, "expected a dictionary");
  for (const auto &[component, destination] :
       {std::pair{"x", &dim.x}, std::pair{"y", &dim.y},
        std::pair{"z", &dim.z}}) {
    StructuredData::ObjectSP value_sp = dictionary->GetValueForKey(component);
    if (!value_sp)
      return InvalidField(path + "." + component, "missing");
    if (llvm::Error error =
            DecodeValue(*value_sp, path + "." + component, *destination))
      return error;
  }
  return llvm::Error::success();
}

/// Decode `key` into `destination` when present; a missing key stays unknown.
template <typename T>
llvm::Error DecodeOptional(const StructuredData::Dictionary &cuda,
                           llvm::StringRef key, std::optional<T> &destination) {
  StructuredData::ObjectSP value_sp = cuda.GetValueForKey(key);
  if (!value_sp)
    return llvm::Error::success();
  T value{};
  if (llvm::Error error = DecodeValue(*value_sp, key, value))
    return error;
  destination = value;
  return llvm::Error::success();
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

llvm::Expected<nvgpu::ThreadCoords>
nvgpu::FromStructuredData(const StructuredData::Object &cuda) {
  if (cuda.GetType() != lldb::eStructuredDataTypeDictionary)
    return llvm::createStringError(
        "invalid CUDA thread coordinate object 'cuda': expected a dictionary");
  const auto &dictionary =
      static_cast<const StructuredData::Dictionary &>(cuda);

  // Report every malformed field; any one of them fails the decode.
  ThreadCoords result;
  llvm::Error error = llvm::Error::success();
  MapThreadCoords(result, [&](llvm::StringRef key, auto &destination) {
    error = llvm::joinErrors(std::move(error),
                             DecodeOptional(dictionary, key, destination));
  });
  if (error)
    return std::move(error);
  return result;
}
