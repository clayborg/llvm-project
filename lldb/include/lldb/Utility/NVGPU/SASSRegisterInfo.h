//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_SASSREGISTERINFO_H
#define LLDB_UTILITY_NVGPU_SASSREGISTERINFO_H

#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"
#include "lldb/lldb-private-types.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace lldb_private {
namespace sass {

/// Canonical packed register buffer for SASS. Every `RegisterInfo::byte_offset`
/// returned by `GetRegisterInfos()` is computed against this struct, so a
/// buffer of size `sizeof(RegisterLayout)` -- indexed by `byte_offset` -- is
/// the format LLDB's `RegisterValue::SetFromMemoryData` /
/// `RegisterValue::GetAsMemoryData` expect when reading or writing a register
/// from its raw bytes.
struct RegisterLayout {
  uint64_t PC;
  uint64_t errorPC;
  uint32_t regular[kNumRRegs];            ///< R0..R254
  uint32_t regular_zero;                  ///< RZ (R255, always reads 0)
  uint32_t predicate[kNumPRegs];          ///< P0..P7
  uint32_t uniform[kNumURRegs];           ///< UR0..UR254
  uint32_t uniform_zero;                  ///< URZ (UR255, always reads 0)
  uint32_t uniform_predicate[kNumUPRegs]; ///< UP0..UP7
};

/// Get the canonical register info table for SASS architecture.
///
/// The returned array contains all SASS registers (PC, errorPC, SP, FP, RA,
/// R0-R254, RZ, P0-P7, UR0-UR254, URZ, UP0-UP7) with proper DWARF,
/// EH_FRAME, and generic register kind mappings. Entries are indexed by
/// LLDB register number as defined in SASSRegisterNumbers.h. Each entry's
/// `byte_offset` / `byte_size` describe its placement inside a
/// `RegisterLayout` buffer.
llvm::ArrayRef<lldb_private::RegisterInfo> GetRegisterInfos();

/// Get the register sets for SASS architecture.
///
/// Returns 5 sets: General Purpose (PC, errorPC, SP, FP, RA), Regular
/// (R0-RZ), Predicate (P0-P7), Uniform (UR0-URZ), Uniform Predicate
/// (UP0-UP7).
llvm::ArrayRef<lldb_private::RegisterSet> GetRegisterSets();

} // namespace sass
} // namespace lldb_private

#endif // LLDB_UTILITY_NVGPU_SASSREGISTERINFO_H
