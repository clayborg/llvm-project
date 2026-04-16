//===-- SASSRegisterInfo.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_SASSREGISTERINFO_H
#define LLDB_UTILITY_NVGPU_SASSREGISTERINFO_H

#include "lldb/lldb-private-types.h"
#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {
namespace sass {

/// Get the canonical register info table for SASS architecture.
///
/// The returned array contains all SASS registers (PC, errorPC, SP, FP, RA,
/// R0-R254, RZ, P0-P7, UR0-UR254, URZ, UP0-UP7) with proper DWARF,
/// EH_FRAME, and generic register kind mappings. Entries are indexed by
/// LLDB register number as defined in SASSRegisterNumbers.h.
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
