//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "cudadebugger.h"

using namespace lldb;
using namespace lldb_private;

using namespace sass;
using namespace sass::regnum;

// Include cudadebugger.h in this file to prevent exposing it in the register
// header. Check our register class mirror (sass::regnum::RegClass) against the
// CUDBGRegClass values in cudadebugger.h to prevent mismatches.
#define ASSERT_REG_CLASS_MATCHES(NAME)                                         \
  static_assert(static_cast<uint32_t>(sass::regnum::NAME) ==                   \
                    static_cast<uint32_t>(CUDBGRegClass::NAME),                \
                #NAME " must match cudadebugger.h")
ASSERT_REG_CLASS_MATCHES(REG_CLASS_INVALID);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_REG_CC);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_REG_PRED);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_REG_ADDR);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_REG_HALF);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_REG_FULL);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_MEM_LOCAL);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_LMEM_REG_OFFSET);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_UREG_PRED);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_UREG_HALF);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_UREG_FULL);
ASSERT_REG_CLASS_MATCHES(REG_CLASS_TEMP_REG_SPILL);
#undef ASSERT_REG_CLASS_MATCHES

// All `byte_offset` values below are computed against `sass::ThreadRegisters`
// (declared in SASSRegisterInfo.h); consumers index a buffer of that type using
// each `RegisterInfo`'s `byte_offset` / `byte_size`.
#define REG_OFFSET(Reg) offsetof(lldb_private::sass::ThreadRegisters, Reg)

#define R_REG_OFFSET(Index)                                                    \
  offsetof(::lldb_private::sass::ThreadRegisters, regular) +                   \
      (Index) * sizeof(lldb_private::sass::ThreadRegisters::regular[0])

#define P_REG_OFFSET(Index)                                                    \
  offsetof(::lldb_private::sass::ThreadRegisters, predicate) +                 \
      (Index) * sizeof(lldb_private::sass::ThreadRegisters::predicate[0])

#define UR_REG_OFFSET(Index)                                                   \
  offsetof(::lldb_private::sass::ThreadRegisters, uniform) +                   \
      (Index) * sizeof(lldb_private::sass::ThreadRegisters::uniform[0])

#define UP_REG_OFFSET(Index)                                                   \
  offsetof(::lldb_private::sass::ThreadRegisters, uniform_predicate) +         \
      (Index) *                                                                \
          sizeof(lldb_private::sass::ThreadRegisters::uniform_predicate[0])

// TODO: Move all these to consteval when we switch to C++20

// Update this if a _generated_ register name needs to be longer than 6 bytes
// ("UR254" + NUL).
static constexpr size_t kMaxRegNameLength = 6;
using RegName = std::array<char, kMaxRegNameLength>;

static constexpr RegName BuildRegName(const char *prefix, uint32_t n) {
  RegName name{};
  uint32_t i = 0;
  uint32_t ndigits = 1;

  for (; prefix[i]; ++i)
    name[i] = prefix[i];

  for (uint32_t t = n; t >= 10; t /= 10)
    ++ndigits;

  if (i + ndigits >= kMaxRegNameLength)
    llvm_unreachable("Reg name too long");

  for (uint32_t d = 0; d < ndigits; ++d, n /= 10)
    name[i + ndigits - 1 - d] = static_cast<char>('0' + n % 10);

  return name;
}

// We're reusing the uniform names for the regular regs.
// Make sure the counts are the same.
static_assert(kNumRRegs == kNumURRegs, "R and UR must share a name table");
static_assert(kNumPRegs == kNumUPRegs, "P and UP must share a name table");

static constexpr auto g_uniform_names = [] {
  std::array<RegName, kNumURRegs> names{};
  for (uint32_t i = 0; i < kNumURRegs; ++i)
    names[i] = BuildRegName("UR", i);
  return names;
}();

static constexpr auto g_uniform_predicate_names = [] {
  std::array<RegName, kNumUPRegs> names{};
  for (uint32_t i = 0; i < kNumUPRegs; ++i)
    names[i] = BuildRegName("UP", i);
  return names;
}();

static constexpr RegisterInfo BuildRegularInfo(uint32_t i, const char *name) {
  return {name,
          nullptr,
          4,
          static_cast<uint32_t>(R_REG_OFFSET(i)),
          lldb::eEncodingUint,
          lldb::eFormatHex,
          {LLDB_INVALID_REGNUM, regnum::GetRegularDWARF(i), LLDB_INVALID_REGNUM,
           regnum::GetRegularLLDB(i), regnum::GetRegularLLDB(i)},
          nullptr,
          nullptr,
          nullptr};
}
static constexpr RegisterInfo BuildUniformInfo(uint32_t i, const char *name) {
  return {name,
          nullptr,
          4,
          static_cast<uint32_t>(UR_REG_OFFSET(i)),
          lldb::eEncodingUint,
          lldb::eFormatHex,
          {LLDB_INVALID_REGNUM, regnum::GetUniformDWARF(i), LLDB_INVALID_REGNUM,
           regnum::GetUniformLLDB(i), regnum::GetUniformLLDB(i)},
          nullptr,
          nullptr,
          nullptr};
}
static constexpr RegisterInfo BuildPredicateInfo(uint32_t i, const char *name) {
  return {name,
          nullptr,
          1,
          static_cast<uint32_t>(P_REG_OFFSET(i)),
          lldb::eEncodingUint,
          lldb::eFormatBoolean,
          {LLDB_INVALID_REGNUM, regnum::GetPredicateDWARF(i),
           LLDB_INVALID_REGNUM, regnum::GetPredicateLLDB(i),
           regnum::GetPredicateLLDB(i)},
          nullptr,
          nullptr,
          nullptr};
}
static constexpr RegisterInfo BuildUniformPredicateInfo(uint32_t i,
                                                        const char *name) {
  return {name,
          nullptr,
          1,
          static_cast<uint32_t>(UP_REG_OFFSET(i)),
          lldb::eEncodingUint,
          lldb::eFormatBoolean,
          {LLDB_INVALID_REGNUM, regnum::GetUniformPredicateDWARF(i),
           LLDB_INVALID_REGNUM, regnum::GetUniformPredicateLLDB(i),
           regnum::GetUniformPredicateLLDB(i)},
          nullptr,
          nullptr,
          nullptr};
}

using RegisterInfoTable = std::array<RegisterInfo, regnum::LLDB_REG_COUNT>;

// Fill the regular register class: R0..R254 plus the RZ zero register.
static constexpr void BuildRegularRange(RegisterInfoTable &infos) {
  for (uint32_t i = 0; i < kNumRRegs; ++i)
    infos[regnum::GetRegularLLDB(i)] =
        BuildRegularInfo(i, g_uniform_names[i].data() + 1);

  infos[regnum::LLDB_RZ] = {
      "RZ",
      "R255",
      4,
      REG_OFFSET(regular_zero),
      lldb::eEncodingUint,
      lldb::eFormatHex,
      {LLDB_INVALID_REGNUM, regnum::GetRegularDWARF(regnum::SASS_ZERO),
       LLDB_INVALID_REGNUM, regnum::LLDB_RZ, regnum::LLDB_RZ},
      nullptr,
      nullptr,
      nullptr};
}

// Fill the predicate register class: P0..P7.
static constexpr void BuildPredicateRange(RegisterInfoTable &infos) {
  for (uint32_t i = 0; i < kNumPRegs; ++i)
    infos[regnum::GetPredicateLLDB(i)] =
        BuildPredicateInfo(i, g_uniform_predicate_names[i].data() + 1);
}

// Fill the uniform register class: UR0..UR254 plus the URZ zero register.
static constexpr void BuildUniformRange(RegisterInfoTable &infos) {
  for (uint32_t i = 0; i < kNumURRegs; ++i)
    infos[regnum::GetUniformLLDB(i)] =
        BuildUniformInfo(i, g_uniform_names[i].data());

  infos[regnum::LLDB_URZ] = {
      "URZ",
      "UR255",
      4,
      REG_OFFSET(uniform_zero),
      lldb::eEncodingUint,
      lldb::eFormatHex,
      {LLDB_INVALID_REGNUM, regnum::GetUniformDWARF(regnum::SASS_ZERO),
       LLDB_INVALID_REGNUM, regnum::LLDB_URZ, regnum::LLDB_URZ},
      nullptr,
      nullptr,
      nullptr};
}

// Fill the uniform predicate register class: UP0..UP7.
static constexpr void BuildUniformPredicateRange(RegisterInfoTable &infos) {
  for (uint32_t i = 0; i < kNumUPRegs; ++i)
    infos[regnum::GetUniformPredicateLLDB(i)] =
        BuildUniformPredicateInfo(i, g_uniform_predicate_names[i].data());
}

// RA is a composite of R20 (low) and R21 (high), forming a 64-bit address.
// The register dependency list must be terminated with LLDB_INVALID_REGNUM.
static std::array<uint32_t, 3> g_ra_value_regs = {
    regnum::GetRegularLLDB(regnum::SASS_RA_LO),
    regnum::GetRegularLLDB(regnum::SASS_RA_HI), LLDB_INVALID_REGNUM};

static constexpr void BuildCommonRegisters(RegisterInfoTable &infos) {
  infos[regnum::LLDB_PC] = {"PC",
                            nullptr,
                            8,
                            REG_OFFSET(PC),
                            lldb::eEncodingUint,
                            lldb::eFormatAddressInfo,
                            {LLDB_INVALID_REGNUM, regnum::DWARF_PSEUDO_PC,
                             LLDB_REGNUM_GENERIC_PC, regnum::LLDB_PC,
                             regnum::LLDB_PC},
                            nullptr,
                            nullptr,
                            nullptr};
  infos[regnum::LLDB_ERROR_PC] = {
      "errorPC",
      nullptr,
      8,
      REG_OFFSET(errorPC),
      lldb::eEncodingUint,
      lldb::eFormatAddressInfo,
      {LLDB_INVALID_REGNUM, regnum::DWARF_PSEUDO_ERROR_PC, LLDB_INVALID_REGNUM,
       regnum::LLDB_ERROR_PC, regnum::LLDB_ERROR_PC},
      nullptr,
      nullptr,
      nullptr};
  infos[regnum::LLDB_SP] = {
      "SP",
      "R[1]",
      4,
      R_REG_OFFSET(1),
      lldb::eEncodingUint,
      lldb::eFormatAddressInfo,
      {LLDB_INVALID_REGNUM, regnum::GetRegularDWARF(regnum::SASS_SP),
       LLDB_REGNUM_GENERIC_SP, regnum::LLDB_SP, regnum::LLDB_SP},
      nullptr,
      nullptr,
      nullptr};
  infos[regnum::LLDB_FP] = {
      "FP",
      "R[2]",
      4,
      R_REG_OFFSET(2),
      lldb::eEncodingUint,
      lldb::eFormatAddressInfo,
      {LLDB_INVALID_REGNUM, regnum::GetRegularDWARF(regnum::SASS_FP),
       LLDB_REGNUM_GENERIC_FP, regnum::LLDB_FP, regnum::LLDB_FP},
      nullptr,
      nullptr,
      nullptr};
  infos[regnum::LLDB_RA] = {"RA",
                            "R[20-21]",
                            8,
                            R_REG_OFFSET(20),
                            lldb::eEncodingUint,
                            lldb::eFormatAddressInfo,
                            {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                             LLDB_REGNUM_GENERIC_RA, regnum::LLDB_RA,
                             regnum::LLDB_RA},
                            g_ra_value_regs.data(),
                            nullptr,
                            nullptr};
}

// Register-set membership arrays. Each register class is a contiguous run of
// LLDB register numbers, so the arrays are generated at compile time.
static constexpr std::array<uint32_t, 5> g_gpr_regnums = {
    LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP, LLDB_RA};

static constexpr auto g_regular_regnums = [] {
  std::array<uint32_t, sass::kNumRRegs + 1> regs{};
  for (uint32_t i = 0; i < sass::kNumRRegs; ++i)
    regs[i] = GetRegularLLDB(i);
  regs[sass::kNumRRegs] = LLDB_RZ; // RZ follows R0..R254.
  return regs;
}();

static constexpr auto g_predicate_regnums = [] {
  std::array<uint32_t, sass::kNumPRegs> regs{};
  for (uint32_t i = 0; i < sass::kNumPRegs; ++i)
    regs[i] = GetPredicateLLDB(i);
  return regs;
}();

static constexpr auto g_uniform_regnums = [] {
  std::array<uint32_t, sass::kNumURRegs + 1> regs{};
  for (uint32_t i = 0; i < sass::kNumURRegs; ++i)
    regs[i] = GetUniformLLDB(i);
  regs[sass::kNumURRegs] = LLDB_URZ; // URZ follows UR0..UR254.
  return regs;
}();

static constexpr auto g_uniform_predicate_regnums = [] {
  std::array<uint32_t, sass::kNumUPRegs> regs{};
  for (uint32_t i = 0; i < sass::kNumUPRegs; ++i)
    regs[i] = GetUniformPredicateLLDB(i);
  return regs;
}();

static const lldb_private::RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr", g_gpr_regnums.size(),
     g_gpr_regnums.data()},
    {"Regular Registers", "r", g_regular_regnums.size(),
     g_regular_regnums.data()},
    {"Predicate Registers", "p", g_predicate_regnums.size(),
     g_predicate_regnums.data()},
    {"Uniform Registers", "ur", g_uniform_regnums.size(),
     g_uniform_regnums.data()},
    {"Uniform Predicate Registers", "up", g_uniform_predicate_regnums.size(),
     g_uniform_predicate_regnums.data()}};

static constexpr auto g_reg_infos = [] {
  RegisterInfoTable infos{};

  BuildCommonRegisters(infos);
  BuildRegularRange(infos);
  BuildPredicateRange(infos);
  BuildUniformRange(infos);
  BuildUniformPredicateRange(infos);

  return infos;
}();

llvm::ArrayRef<lldb_private::RegisterInfo> sass::GetRegisterInfos() {
  return llvm::ArrayRef(g_reg_infos);
}

llvm::ArrayRef<lldb_private::RegisterSet> sass::GetRegisterSets() {
  return llvm::ArrayRef(g_reg_sets);
}
