//===-- SASSRegisterInfo.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"
#include "lldb/lldb-defines.h"

#include "cudadebugger.h"

using namespace lldb;
using namespace lldb_private;

namespace {

// Canonical storage layout for computing byte_offset values in RegisterInfo.
// This layout matches the server-side ThreadRegistersValues struct so that
// byte offsets are consistent between live debugging and post-mortem analysis.
struct SASSRegisterLayout {
  uint64_t PC;
  uint64_t errorPC;
  uint32_t regular[sass::kNumRRegs];
  uint32_t regular_zero;
  uint32_t predicate[sass::kNumPRegs];
  uint32_t uniform[sass::kNumURRegs];
  uint32_t uniform_zero;
  uint32_t uniform_predicate[sass::kNumUPRegs];
};

} // anonymous namespace

#define REG_OFFSET(Reg) offsetof(SASSRegisterLayout, Reg)

#define R_REG_OFFSET(Index)                                                    \
  offsetof(SASSRegisterLayout, regular) +                                      \
      (Index) * sizeof(SASSRegisterLayout::regular[0])

#define P_REG_OFFSET(Index)                                                    \
  offsetof(SASSRegisterLayout, predicate) +                                    \
      (Index) * sizeof(SASSRegisterLayout::predicate[0])

#define UR_REG_OFFSET(Index)                                                   \
  offsetof(SASSRegisterLayout, uniform) +                                      \
      (Index) * sizeof(SASSRegisterLayout::uniform[0])

#define UP_REG_OFFSET(Index)                                                   \
  offsetof(SASSRegisterLayout, uniform_predicate) +                            \
      (Index) * sizeof(SASSRegisterLayout::uniform_predicate[0])

#include "RegisterDefinitionsSASS.inc"

#define EXPAND_REGULAR_REGISTERS(PREFIX) EXPAND_REGULAR_REGISTERS_255(PREFIX)
#define EXPAND_UNIFORM_REGISTERS(PREFIX) EXPAND_UNIFORM_REGISTERS_255(PREFIX)
#define EXPAND_PREDICATE_REGISTERS(PREFIX) EXPAND_PREDICATE_REGISTERS_8(PREFIX)
#define EXPAND_UNIFORM_PREDICATE_REGISTERS(PREFIX)                             \
  EXPAND_UNIFORM_PREDICATE_REGISTERS_8(PREFIX)

// LLDB register numbers must start at 0 and be contiguous with no gaps.
enum LLDBRegNum : uint32_t {
  LLDB_PC = sass::LLDB_PC,
  LLDB_ERROR_PC = sass::LLDB_ERROR_PC,
  LLDB_SP = sass::LLDB_SP,
  LLDB_FP = sass::LLDB_FP,
  LLDB_RA = sass::LLDB_RA,
  EXPAND_REGULAR_REGISTERS(LLDB),
  LLDB_RZ,
  EXPAND_PREDICATE_REGISTERS(LLDB),
  EXPAND_UNIFORM_REGISTERS(LLDB),
  LLDB_URZ,
  EXPAND_UNIFORM_PREDICATE_REGISTERS(LLDB),
  kNumRegs,
};

// DWARF register numbers use CUDA-encoded values where the register class is
// in the upper 8 bits and register number in the lower 24 bits.
enum DWARFRegNum : uint32_t {
  DWARF_PC = sass::DWARF_PSEUDO_PC,
  DWARF_ERROR_PC = sass::DWARF_PSEUDO_ERROR_PC,
  DWARF_SP = sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL,
                                           sass::SASS_SP_REG),
  DWARF_FP = sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL,
                                           sass::SASS_FP_REG),
  GENERATE_DWARF_REGULAR_DEFS(),
  DWARF_RZ = sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL,
                                           sass::SASS_ZERO_REG),
  GENERATE_DWARF_PREDICATE_DEFS(),
  GENERATE_DWARF_UNIFORM_DEFS(),
  DWARF_URZ = sass::GetDWARFEncodedRegister(REG_CLASS_UREG_FULL,
                                            sass::SASS_ZERO_REG),
  GENERATE_DWARF_UNIFORM_PREDICATE_DEFS()
};

// EH_FRAME register numbers for .eh_frame unwind info.
enum CompilerRegNum : uint32_t {
  EH_FRAME_PC = 1000,
  EH_FRAME_ERROR_PC,
  EH_FRAME_SP,
  EH_FRAME_FP,
  EH_FRAME_RA,
  EXPAND_REGULAR_REGISTERS(EH_FRAME),
  EH_FRAME_RZ = 2500,
  EXPAND_PREDICATE_REGISTERS(EH_FRAME),
  EXPAND_UNIFORM_REGISTERS(EH_FRAME),
  EH_FRAME_URZ = 2501,
  EXPAND_UNIFORM_PREDICATE_REGISTERS(EH_FRAME),
};

static uint32_t g_gpr_regnums[] = {LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP,
                                   LLDB_RA};
static uint32_t g_regular_regnums[] = {EXPAND_REGULAR_REGISTERS(LLDB), LLDB_RZ};
static uint32_t g_predicate_regnums[] = {EXPAND_PREDICATE_REGISTERS(LLDB)};
static uint32_t g_uniform_regnums[] = {EXPAND_UNIFORM_REGISTERS(LLDB),
                                       LLDB_URZ};
static uint32_t g_uniform_predicate_regnums[] = {
    EXPAND_UNIFORM_PREDICATE_REGISTERS(LLDB)};

static lldb_private::RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     sizeof(g_gpr_regnums) / sizeof(g_gpr_regnums[0]), g_gpr_regnums},
    {"Regular Registers", "r", LLDB_RZ - LLDB_R0 + 1, g_regular_regnums},
    {"Predicate Registers", "p", sass::kNumPRegs, g_predicate_regnums},
    {"Uniform Registers", "ur", LLDB_URZ - LLDB_UR0 + 1, g_uniform_regnums},
    {"Uniform Predicate Registers", "up", sass::kNumUPRegs,
     g_uniform_predicate_regnums}};

// RA is a composite of R20 (low) and R21 (high), forming a 64-bit address.
static uint32_t g_ra_value_regs[] = {LLDB_R20, LLDB_R21, LLDB_INVALID_REGNUM};

static const lldb_private::RegisterInfo g_reg_infos[LLDBRegNum::kNumRegs] = {
    // PC
    {"PC", nullptr, 8, REG_OFFSET(PC), eEncodingUint, eFormatAddressInfo,
     {EH_FRAME_PC, DWARF_PC, LLDB_REGNUM_GENERIC_PC, LLDB_PC, LLDB_PC},
     nullptr, nullptr, nullptr},
    // errorPC
    {"errorPC", nullptr, 8, REG_OFFSET(errorPC), eEncodingUint,
     eFormatAddressInfo,
     {EH_FRAME_ERROR_PC, DWARF_ERROR_PC, LLDB_INVALID_REGNUM, LLDB_ERROR_PC,
      LLDB_ERROR_PC},
     nullptr, nullptr, nullptr},
    // SP (alias for R1)
    {"SP", "R[1]", 4, R_REG_OFFSET(1), eEncodingUint, eFormatAddressInfo,
     {EH_FRAME_SP, DWARF_R1, LLDB_REGNUM_GENERIC_SP, LLDB_SP, LLDB_SP},
     nullptr, nullptr, nullptr},
    // FP (alias for R2)
    {"FP", "R[2]", 4, R_REG_OFFSET(2), eEncodingUint, eFormatAddressInfo,
     {EH_FRAME_FP, DWARF_R2, LLDB_REGNUM_GENERIC_FP, LLDB_FP, LLDB_FP},
     nullptr, nullptr, nullptr},
    // RA (composite of R20-R21)
    {"RA", "R[20-21]", 8, R_REG_OFFSET(20), eEncodingUint, eFormatAddressInfo,
     {EH_FRAME_RA, LLDB_INVALID_REGNUM, LLDB_REGNUM_GENERIC_RA, LLDB_RA,
      LLDB_RA},
     g_ra_value_regs, nullptr, nullptr},
    // R0-R254
    GENERATE_ALL_REGULAR_REGISTER_INFO(),
    // RZ (R255 zero register)
    {"RZ", "R255", 4, REG_OFFSET(regular_zero), eEncodingUint, eFormatHex,
     {2500,
      sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL, sass::SASS_ZERO_REG),
      LLDB_INVALID_REGNUM, LLDB_RZ, LLDB_RZ},
     nullptr, nullptr, nullptr},
    // P0-P7
    GENERATE_ALL_PREDICATE_REGISTER_INFO(),
    // UR0-UR254
    GENERATE_ALL_UNIFORM_REGISTER_INFO(),
    // URZ (UR255 uniform zero register)
    {"URZ", "UR255", 4, REG_OFFSET(uniform_zero), eEncodingUint, eFormatHex,
     {2501,
      sass::GetDWARFEncodedRegister(REG_CLASS_UREG_FULL, sass::SASS_ZERO_REG),
      LLDB_INVALID_REGNUM, LLDB_URZ, LLDB_URZ},
     nullptr, nullptr, nullptr},
    // UP0-UP7
    GENERATE_ALL_UNIFORM_PREDICATE_REGISTER_INFO()};

static_assert(kNumRegs == sass::kNumSASSRegs,
              "LLDBRegNum::kNumRegs must match sass::kNumSASSRegs");

llvm::ArrayRef<lldb_private::RegisterInfo> sass::GetRegisterInfos() {
  return llvm::ArrayRef(g_reg_infos);
}

llvm::ArrayRef<lldb_private::RegisterSet> sass::GetRegisterSets() {
  return llvm::ArrayRef(g_reg_sets);
}
