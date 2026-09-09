//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileIntelGT.h"

#include "lldb/Core/Architecture.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

SymbolFileIntelGT::SymbolFileIntelGT(ObjectFileSP objfile_sp,
                                     SectionList *dwo_section_list)
    : SymbolFileDWARF(std::move(objfile_sp), dwo_section_list) {}

SymbolFileIntelGT::~SymbolFileIntelGT() = default;

lldb::offset_t
SymbolFileIntelGT::GetVendorDWARFOpcodeSize(const DataExtractor &,
                                            const lldb::offset_t,
                                            const uint8_t op) const {
  switch (op) {
  case DW_OP_INTEL_push_simd_lane:
    return 0;
  case DW_OP_INTEL_regval_bits:
    return 1;
  default:
    return LLDB_INVALID_OFFSET;
  }
}

bool SymbolFileIntelGT::ParseVendorDWARFOpcode(uint8_t op,
                                               const DataExtractor &opcodes,
                                               lldb::offset_t &offset,
                                               ExecutionContext *exe_ctx,
                                               RegisterContext *reg_ctx,
                                               lldb::RegisterKind reg_kind,
                                               std::vector<Value> &stack) const {
  // Delegate the dispatch so that SymbolFile (variable-location) path and
  // the CFI (unwind) path share one implementation.
  if (!exe_ctx)
    return false;
  Target *target = exe_ctx->GetTargetPtr();
  if (!target)
    return false;
  Architecture *arch = target->GetArchitecturePlugin();
  if (!arch)
    return false;
  return arch->ParseVendorDWARFOpcode(op, opcodes, offset, exe_ctx, reg_ctx,
                                      reg_kind, stack);
}
