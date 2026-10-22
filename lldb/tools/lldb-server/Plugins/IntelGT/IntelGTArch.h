//===-- IntelGTArch.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_INTELGTARCH_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_INTELGTARCH_H

#include <cstdint>
#include <string>

namespace lldb_private {
namespace lldb_server {
namespace intelgt {

// ---------------------------------------------------------------------------
// Feature name constants
// ---------------------------------------------------------------------------

static constexpr const char *feature_grf = "org.gnu.gdb.intelgt.grf";
static constexpr const char *feature_addr = "org.gnu.gdb.intelgt.addr";
static constexpr const char *feature_flag = "org.gnu.gdb.intelgt.flag";
static constexpr const char *feature_ce = "org.gnu.gdb.intelgt.ce";
static constexpr const char *feature_sr = "org.gnu.gdb.intelgt.sr";
static constexpr const char *feature_cr = "org.gnu.gdb.intelgt.cr";
static constexpr const char *feature_tdr = "org.gnu.gdb.intelgt.tdr";
static constexpr const char *feature_acc = "org.gnu.gdb.intelgt.acc";
static constexpr const char *feature_mme = "org.gnu.gdb.intelgt.mme";
static constexpr const char *feature_sp = "org.gnu.gdb.intelgt.sp";
static constexpr const char *feature_sba = "org.gnu.gdb.intelgt.sba";
static constexpr const char *feature_dbg = "org.gnu.gdb.intelgt.dbg";
static constexpr const char *feature_fc = "org.gnu.gdb.intelgt.fc";
static constexpr const char *feature_msg = "org.gnu.gdb.intelgt.msg";
static constexpr const char *feature_mf = "org.gnu.gdb.intelgt.mf";
static constexpr const char *feature_debugger = "org.gnu.gdb.intelgt.debugger";
static constexpr const char *feature_scratch = "org.gnu.gdb.intelgt.scratch";
static constexpr const char *feature_scalar = "org.gnu.gdb.intelgt.scalar";

// ---------------------------------------------------------------------------
// Instruction size constants
// ---------------------------------------------------------------------------

static constexpr uint32_t MAX_INST_LENGTH = 16;
static constexpr uint32_t COMPACT_INST_LENGTH = 8;

// ---------------------------------------------------------------------------
// CR0 bit position constants
// ---------------------------------------------------------------------------

static constexpr uint32_t cr0_0_breakpoint_suppress = 15;
static constexpr uint32_t cr0_1_breakpoint_status = 31;
static constexpr uint32_t cr0_1_external_halt_status = 30;
static constexpr uint32_t cr0_1_software_exception_control = 29;
static constexpr uint32_t cr0_1_illegal_opcode_status = 28;
static constexpr uint32_t cr0_1_systolic_exception_status = 27;
static constexpr uint32_t cr0_1_oob_status = 27;
static constexpr uint32_t cr0_1_force_exception_status = 26;
static constexpr uint32_t cr0_1_shared_function_exception_status = 23;
static constexpr uint32_t cr0_1_pagefault_status = 16;

// ---------------------------------------------------------------------------
// DWARF register set enumeration
// ---------------------------------------------------------------------------

enum dwarf_regsets : uint32_t {
  dwarf_regset_sba = 0,
  dwarf_regset_grf,
  dwarf_regset_addr,
  dwarf_regset_flag,
  dwarf_regset_acc,
  dwarf_regset_mme,
  dwarf_regset_count,
};

// ---------------------------------------------------------------------------
// Xe version
// ---------------------------------------------------------------------------

enum xe_version : uint32_t {
  XE_HP = 0x00000001,
  XE_HPG = 0x00000002,
  XE_HPC = 0x00000004,
  XE2 = 0x00000008,
};

#define XE_VERSION(maj, min) (((maj) << 16) | (min))

// ---------------------------------------------------------------------------
// Breakpoint kind
// ---------------------------------------------------------------------------

enum breakpoint_kind : uint32_t {
  INTELGT_BP_KIND_BREAKPOINT = 1,
};

// ---------------------------------------------------------------------------
// Instruction bit helpers
// ---------------------------------------------------------------------------

/// Return the value of bit \a bit_pos in instruction \a inst.
inline uint32_t get_inst_bit(const uint8_t *inst, uint32_t bit_pos) {
  uint32_t byte_pos = bit_pos / 8;
  uint32_t bit_in_byte = bit_pos % 8;
  return (inst[byte_pos] >> bit_in_byte) & 1;
}

/// Set bit \a bit_pos in instruction \a inst.
inline void set_inst_bit(uint8_t *inst, uint32_t bit_pos) {
  uint32_t byte_pos = bit_pos / 8;
  uint32_t bit_in_byte = bit_pos % 8;
  inst[byte_pos] |= (1u << bit_in_byte);
}

/// Clear bit \a bit_pos in instruction \a inst.
inline void clear_inst_bit(uint8_t *inst, uint32_t bit_pos) {
  uint32_t byte_pos = bit_pos / 8;
  uint32_t bit_in_byte = bit_pos % 8;
  inst[byte_pos] &= ~(1u << bit_in_byte);
}

/// Return the bit offset for the breakpoint bit in the instruction.
uint32_t breakpoint_bit_offset(const uint8_t *inst, uint32_t device_id);

// ---------------------------------------------------------------------------
// Device-ID to xe_version mapping
// ---------------------------------------------------------------------------

/// Map a Level Zero device ID to the corresponding xe_version bitmask.
uint32_t get_xe_version(uint32_t device_id);

/// Return true if the device is Xe2 or later (systolic exception support).
bool is_xe2_or_later(uint32_t device_id);

// ---------------------------------------------------------------------------
// Register set info
// ---------------------------------------------------------------------------

struct ze_regset_info {
  uint32_t type;     ///< ZET_DEBUG_REGSET_TYPE_* value.
  uint32_t size;     ///< Total byte size of this register set.
  uint32_t begin;    ///< First register index in this set.
  uint32_t end;      ///< One-past-last register index in this set.
  bool is_writeable; ///< Whether registers in this set are writeable.
};

// ---------------------------------------------------------------------------
// Device topology enumerations
// ---------------------------------------------------------------------------

enum ze_node_level_t {
  ZE_NODE_DEVICE = 0,
  ZE_NODE_SUBDEVICE,
  ZE_NODE_SLICE,
  ZE_NODE_SUBSLICE,
  ZE_NODE_EU,
  ZE_NODE_THREAD,
  ZE_NODE_COUNT,
};

enum ze_node_state_t {
  ZE_NODE_STATE_UNKNOWN = 0,
  ZE_NODE_STATE_STOPPED,
  ZE_NODE_STATE_RUNNING,
  ZE_NODE_STATE_UNAVAILABLE,
};

} // namespace intelgt
} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_INTELGTARCH_H
