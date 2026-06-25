//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_SASSREGISTERNUMBERS_H
#define LLDB_UTILITY_NVGPU_SASSREGISTERNUMBERS_H

#include <stdint.h>

namespace lldb_private {

/// The \c sass namespace contains SASS (GPU instruction set) architecture
/// definitions: register numbering, DWARF encoding, and register info tables.
/// General GPU utilities (thread naming, address spaces, exception handling)
/// live in the \c nvgpu namespace instead.
namespace sass {

/// Register counts for each register class.
constexpr uint32_t kNumRRegs = 255;  /// R0-R254
constexpr uint32_t kNumPRegs = 8;    /// P0-P7
constexpr uint32_t kNumURRegs = 255; /// UR0-UR254
constexpr uint32_t kNumUPRegs = 8;   /// UP0-UP7
/// Field count of CUDA's uint3/dim3 vectors (x, y, z).
constexpr uint32_t kNumXYZComponents = 3;

namespace regnum {

/// Declare a contiguous register-number range inside an enum: `<NAME>_BASE`
/// takes the next enum value and `<NAME>_LAST` is the last of `Count`
/// consecutive registers, so register i is `<NAME>_BASE + i`.
#define SASS_REG_RANGE(NAME, Count)                                            \
  NAME##_BASE, NAME##_LAST = NAME##_BASE + (Count) - 1

// Mirror of the CUDBGRegClass values from cudadebugger.h that are needed
// for DWARF register encoding. This is intended to avoid cudadebugger.h
// from being included through the register header.
// We check the values against the ones in cudadebugger.h in
// SASSRegisterInfo.cpp.
/// Physical location of DWARF register.
enum DWARFRegClass : uint32_t {
  REG_CLASS_INVALID = 0x00,         ///< No location, used for pseudo-registers.
  REG_CLASS_REG_CC = 0x01,          ///< Condition code register.
  REG_CLASS_REG_PRED = 0x02,        ///< Predicate register.
  REG_CLASS_REG_ADDR = 0x03,        ///< Address register.
  REG_CLASS_REG_HALF = 0x04,        ///< 16-bit register.
  REG_CLASS_REG_FULL = 0x05,        ///< 32-bit register.
  REG_CLASS_MEM_LOCAL = 0x06,       ///< Local memory register.
  REG_CLASS_LMEM_REG_OFFSET = 0x07, ///< Local memory register offset.
  REG_CLASS_UREG_PRED = 0x09,       ///< Uniform predicate register.
  REG_CLASS_UREG_HALF = 0x0a,       ///< 16-bit uniform register.
  REG_CLASS_UREG_FULL = 0x0b,       ///< 32-bit uniform register.
  REG_CLASS_TEMP_REG_SPILL = 0x0c,  ///< Temp register spill.
};

/// DWARF pseudo register numbers, used for virtual variables (metadata)
/// not backed by any physical register.
/// These must be used in the REG_CLASS_INVALID (most significant byte 0) class
/// space. Pseudo register starts at 1 so that null DWARF reg is invalid.
enum DWARFPseudoRegNum : uint32_t {
  DWARF_PSEUDO_INVALID,
  DWARF_PSEUDO_PC,
  DWARF_PSEUDO_ERROR_PC,
  DWARF_PSEUDO_THREAD_IDX,
  DWARF_PSEUDO_BLOCK_IDX,
  DWARF_PSEUDO_BLOCK_DIM,
  DWARF_PSEUDO_GRID_DIM,
  DWARF_PSEUDO_WARP_SIZE,
};

/// Special SASS hardware register indices.
enum SASSSpecialRegNum : uint32_t {
  SASS_SP = 1,     ///< R1 is the stack pointer.
  SASS_FP = 2,     ///< R2 is the frame pointer.
  SASS_RA_LO = 20, ///< R20-R21 store the return address.
  SASS_RA_HI = 21,
  SASS_ZERO = 255, ///< R255/UR255 are the zero registers.
};

/// LLDB register numbers for SASS.
/// They start at 0 and are contiguous with no gaps, since they are used as
/// indices into the register-info table. Each register class occupies a
/// contiguous [_BASE, _LAST] range.
enum LLDBRegNum : uint32_t {
  LLDB_PC,
  LLDB_ERROR_PC,
  LLDB_SP,
  LLDB_FP,
  LLDB_RA,
  SASS_REG_RANGE(LLDB_R, kNumRRegs),
  LLDB_RZ,
  SASS_REG_RANGE(LLDB_P, kNumPRegs),
  SASS_REG_RANGE(LLDB_UR, kNumURRegs),
  LLDB_URZ,
  SASS_REG_RANGE(LLDB_UP, kNumUPRegs),
  LLDB_VREG_THREAD_IDX,
  LLDB_VREG_BLOCK_IDX,
  LLDB_VREG_BLOCK_DIM,
  LLDB_VREG_GRID_DIM,
  LLDB_VREG_WARP_SIZE,
  LLDB_REG_COUNT,
};

#undef SASS_REG_RANGE

// ------------------------------------------------------------
// DWARF register number helpers
// ------------------------------------------------------------

/// Encode register class and number into a single DWARF register value.
///
/// \param[in] reg_class
///     The register class (stored in upper 8 bits).
///
/// \param[in] reg_num
///     The register number (stored in lower 24 bits).
///
/// \return
///     A DWARF encoded register value.
constexpr uint32_t GetDWARFEncodedRegister(uint32_t reg_class,
                                           uint32_t reg_num) {
  return ((reg_class << 24) | (reg_num & 0x00FFFFFF));
}
/// Extract the register class from a DWARF encoded register value.
///
/// \param[in] encoded_reg
///     The DWARF encoded register value.
///
/// \return
///     The register class (upper 8 bits).
constexpr uint32_t GetDWARFRegisterClass(uint32_t encoded_reg) {
  return (encoded_reg >> 24) & 0xFF;
}
/// Extract the register number from a DWARF encoded register value.
///
/// \param[in] encoded_reg
///     The DWARF encoded register value.
///
/// \return
///     The register number (lower 24 bits).
constexpr uint32_t GetDWARFRegisterNumber(uint32_t encoded_reg) {
  return encoded_reg & 0x00FFFFFF;
}
/// \return the DWARF number for regular register R\a n.
constexpr uint32_t GetRegularDWARF(uint32_t n) {
  return GetDWARFEncodedRegister(REG_CLASS_REG_FULL, n);
}
/// \return the DWARF number for uniform register UR\a n.
constexpr uint32_t GetUniformDWARF(uint32_t n) {
  return GetDWARFEncodedRegister(REG_CLASS_UREG_FULL, n);
}
/// \return the DWARF number for predicate register P\a n.
constexpr uint32_t GetPredicateDWARF(uint32_t n) {
  return GetDWARFEncodedRegister(REG_CLASS_REG_PRED, n);
}
/// \return the DWARF number for uniform predicate register UP\a n.
constexpr uint32_t GetUniformPredicateDWARF(uint32_t n) {
  return GetDWARFEncodedRegister(REG_CLASS_UREG_PRED, n);
}

// ------------------------------------------------------------
// LLDB register number helpers
// ------------------------------------------------------------

/// \return the LLDB register number for regular register R\a n.
constexpr uint32_t GetRegularLLDB(uint32_t n) { return LLDB_R_BASE + n; }
/// \return the LLDB register number for uniform register UR\a n.
constexpr uint32_t GetUniformLLDB(uint32_t n) { return LLDB_UR_BASE + n; }
/// \return the LLDB register number for predicate register P\a n.
constexpr uint32_t GetPredicateLLDB(uint32_t n) { return LLDB_P_BASE + n; }
/// \return the LLDB register number for uniform predicate register UP\a n.
constexpr uint32_t GetUniformPredicateLLDB(uint32_t n) {
  return LLDB_UP_BASE + n;
}
/// \return true if \a reg is an LLDB number for a regular register R\a n.
constexpr bool IsRegularLLDB(uint32_t reg) {
  return reg >= LLDB_R_BASE && reg <= LLDB_R_LAST;
}
/// \return true if \a reg is an LLDB number for a uniform register UR\a n.
constexpr bool IsUniformLLDB(uint32_t reg) {
  return reg >= LLDB_UR_BASE && reg <= LLDB_UR_LAST;
}
/// \return true if \a reg is an LLDB number for a predicate register P\a n.
constexpr bool IsPredicateLLDB(uint32_t reg) {
  return reg >= LLDB_P_BASE && reg <= LLDB_P_LAST;
}
/// \return true if \a reg is an LLDB number for a uniform predicate
/// register UP\a n.
constexpr bool IsUniformPredicateLLDB(uint32_t reg) {
  return reg >= LLDB_UP_BASE && reg <= LLDB_UP_LAST;
}
} // namespace regnum
} // namespace sass
} // namespace lldb_private

#endif // LLDB_UTILITY_NVGPU_SASSREGISTERNUMBERS_H
