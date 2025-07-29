//===-- Coords.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_COORDS_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_COORDS_H

#include <cstdint>
#include <string>

namespace lldb_private::lldb_server {
/// This struct represents the physical coordinates of a HW thread in a GPU.
struct PhysicalCoords {
  int64_t dev_id;
  int64_t sm_id;
  int64_t warp_id;
  int64_t lane_id;

  PhysicalCoords();

  PhysicalCoords(int64_t dev_id, int64_t sm_id, int64_t warp_id,
                 int64_t lane_id);

  /// \return \b
  //    true if the coordinates correspond to a valid HW thread.
  bool IsValid() const;

  /// \return
  //    A string representation of the coordinates that can be used to name a
  //    thread.
  std::string AsThreadName() const;

  /// \return
  //    A string representation of the coordinates used for logging.
  std::string Dump() const;
};
} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_COORDS_H
