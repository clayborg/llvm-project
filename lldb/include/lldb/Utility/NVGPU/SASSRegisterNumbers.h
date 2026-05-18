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
inline constexpr uint32_t GetDWARFEncodedRegister(uint32_t reg_class,
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
inline constexpr uint32_t GetDWARFRegisterClass(uint32_t encoded_reg) {
  return (encoded_reg >> 24) & 0xFF;
}

/// Extract the register number from a DWARF encoded register value.
///
/// \param[in] encoded_reg
///     The DWARF encoded register value.
///
/// \return
///     The register number (lower 24 bits).
inline constexpr uint32_t GetDWARFRegisterNumber(uint32_t encoded_reg) {
  return encoded_reg & 0x00FFFFFF;
}

// LLDB common register numbers for SASS architecture
static constexpr uint32_t LLDB_PC = 0;       // Program Counter
static constexpr uint32_t LLDB_ERROR_PC = 1; // Error PC
static constexpr uint32_t LLDB_SP = 2;       // Stack Pointer
static constexpr uint32_t LLDB_FP = 3;       // Frame Pointer
static constexpr uint32_t LLDB_RA = 4;       // Return Address

// Pseudo-DWARF register numbers for 64-bit registers that don't have
// standard DWARF representations in SASS
static constexpr uint32_t DWARF_PSEUDO_PC = 128;
static constexpr uint32_t DWARF_PSEUDO_ERROR_PC = 129;

// Special register constants
static constexpr uint32_t SASS_SP_REG = 1;     // R1 is the stack pointer
static constexpr uint32_t SASS_FP_REG = 2;     // R2 is the frame pointer
static constexpr uint32_t SASS_RA_REG_LO = 20; // R20-R21 store return address
static constexpr uint32_t SASS_RA_REG_HI = 21;
static constexpr uint32_t SASS_ZERO_REG = 255; // R255 is the zero register

// Register counts for each register class
static constexpr uint32_t kNumRRegs = 255;  // R0-R254
static constexpr uint32_t kNumPRegs = 8;    // P0-P7
static constexpr uint32_t kNumURRegs = 255; // UR0-UR254
static constexpr uint32_t kNumUPRegs = 8;   // UP0-UP7

// Computed LLDB register number offsets for each register class.
// Layout: PC, ErrorPC, SP, FP, RA, R0..R254, RZ, P0..P7,
//         UR0..UR254, URZ, UP0..UP7
static constexpr uint32_t LLDB_R0 = LLDB_RA + 1;
static constexpr uint32_t LLDB_RZ = LLDB_R0 + kNumRRegs;
static constexpr uint32_t LLDB_P0 = LLDB_RZ + 1;
static constexpr uint32_t LLDB_UR0 = LLDB_P0 + kNumPRegs;
static constexpr uint32_t LLDB_URZ = LLDB_UR0 + kNumURRegs;
static constexpr uint32_t LLDB_UP0 = LLDB_URZ + 1;
static constexpr uint32_t kNumSASSRegs = LLDB_UP0 + kNumUPRegs;

} // namespace sass
} // namespace lldb_private

#endif // LLDB_UTILITY_NVGPU_SASSREGISTERNUMBERS_H
