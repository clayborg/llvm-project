//===-- IntelGTArch.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntelGTArch.h"

namespace lldb_private {
namespace lldb_server {
namespace intelgt {

// ---------------------------------------------------------------------------
// Device ID to xe_version mapping
// ---------------------------------------------------------------------------

uint32_t get_xe_version(uint32_t device_id) {
  // Xe_HP devices.
  switch (device_id) {
  // Xe HP SDV
  case 0x4f80:
  case 0x4f81:
  case 0x4f82:
  case 0x4f83:
  case 0x4f84:
  case 0x4f85:
  case 0x4f86:
  case 0x4f87:
  case 0x4f88:
  // Xe HP PVC (XL)
  case 0x0201:
  case 0x0202:
  case 0x0203:
  case 0x0204:
  case 0x0205:
  case 0x0206:
  case 0x0207:
  case 0x0208:
  case 0x0209:
  case 0x020a:
  case 0x020b:
  case 0x020c:
  case 0x020d:
  case 0x020e:
  case 0x020f:
  case 0x0210:
    return XE_HP;

  // Xe HPG devices (Arc Alchemist family).
  case 0x5694:
  case 0x5695:
  case 0x5696:
  case 0x5697:
  case 0x56a0:
  case 0x56a1:
  case 0x56a2:
  case 0x56b0:
  case 0x56b1:
  case 0x56c0:
  case 0x56c1:
  case 0x56a5:
  case 0x56a6:
  case 0x5698:
  case 0x5699:
  case 0x569a:
  case 0x569b:
  case 0x569c:
  case 0x56cf:
  case 0x56b2:
  case 0x56b3:
    return XE_HPG;

  // Xe HPC (Ponte Vecchio).
  case 0x0bd0:
  case 0x0bd5:
  case 0x0bd6:
  case 0x0bd7:
  case 0x0bd8:
  case 0x0bd9:
  case 0x0bda:
  case 0x0bdb:
  case 0x0b69:
  case 0x0b6e:
    return XE_HPC;

  // Xe2 (Battlemage family).
  case 0xe202:
  case 0xe20b:
  case 0xe20c:
  case 0xe20d:
  case 0xe212:
  case 0xe215:
  case 0xe216:
    return XE2;

  default:
    // Unknown device; assume XE_HPG as a safe default for EU debugging.
    return XE_HPG;
  }
}

// ---------------------------------------------------------------------------
// Breakpoint bit offset
// ---------------------------------------------------------------------------

uint32_t breakpoint_bit_offset(const uint8_t *inst, uint32_t device_id) {
  // Compact (8B) instructions carry the breakpoint bit at bit 7; full (16B)
  // instructions carry it at bit 30. CmptCtrl (bit 29 of DWORD[0]) selects.
  (void)device_id;
  bool is_compact = (inst[3] & 0x20) != 0;
  return is_compact ? 7 : 30;
}

bool is_xe2_or_later(uint32_t device_id) {
  return (get_xe_version(device_id) & XE2) != 0;
}

} // namespace intelgt
} // namespace lldb_server
} // namespace lldb_private
