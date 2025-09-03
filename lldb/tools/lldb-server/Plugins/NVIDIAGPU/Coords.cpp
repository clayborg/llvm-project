//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Coords.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;
using namespace lldb_private::lldb_server;

namespace {
enum CoordSpecialIDs : uint32_t {
  INVALID_ID = std::numeric_limits<uint32_t>::max(),
  WILDCARD_ID = INVALID_ID - 1,
  CURRENT_ID = INVALID_ID - 2,
  IGNORE_ID = INVALID_ID - 3,
};
} // namespace

PhysicalCoords::PhysicalCoords()
    : dev_id(CoordSpecialIDs::INVALID_ID), sm_id(CoordSpecialIDs::INVALID_ID),
      warp_id(CoordSpecialIDs::INVALID_ID),
      lane_id(CoordSpecialIDs::INVALID_ID) {}

PhysicalCoords::PhysicalCoords(int64_t dev_id, int64_t sm_id, int64_t warp_id,
                               int64_t lane_id)
    : dev_id(dev_id), sm_id(sm_id), warp_id(warp_id), lane_id(lane_id) {}

bool PhysicalCoords::IsValid() const {
  return dev_id != CoordSpecialIDs::INVALID_ID &&
         sm_id != CoordSpecialIDs::INVALID_ID &&
         warp_id != CoordSpecialIDs::INVALID_ID &&
         lane_id != CoordSpecialIDs::INVALID_ID;
}

std::string PhysicalCoords::AsThreadName() const {
  if (IsValid())
    return llvm::formatv("GPU Thread ({}, {}, {})", sm_id, warp_id, lane_id);
  return "NVIDIA GPU";
}

std::string PhysicalCoords::Dump() const {
  return llvm::formatv("dev_id = {} sm_id = {} warp_id = {} lane_id = {}",
                       dev_id, sm_id, warp_id, lane_id);
}
