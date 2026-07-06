//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEINTELGT_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEINTELGT_H

#include "SymbolFileDWARF.h"

namespace lldb_private::plugin {
namespace dwarf {

// Intel GT vendor DWARF opcodes in the DW_OP_lo_user..DW_OP_hi_user range
// (0xe0..0xff). 0xed collides with DW_OP_WASM_location in Dwarf.def so both
// live here instead. Selected for spirv64 triples in
// SymbolFileDWARF::CreateInstance.
static constexpr uint8_t DW_OP_INTEL_push_simd_lane = 0xed;
static constexpr uint8_t DW_OP_INTEL_regval_bits     = 0xfe;

class SymbolFileIntelGT : public SymbolFileDWARF {
public:
  SymbolFileIntelGT(lldb::ObjectFileSP objfile_sp,
                    SectionList *dwo_section_list);

  ~SymbolFileIntelGT() override;

  lldb::offset_t GetVendorDWARFOpcodeSize(const DataExtractor &data,
                                          lldb::offset_t data_offset,
                                          uint8_t op) const override;

  bool ParseVendorDWARFOpcode(uint8_t op, const DataExtractor &opcodes,
                              lldb::offset_t &offset,
                              ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
                              lldb::RegisterKind reg_kind,
                              std::vector<Value> &stack) const override;
};

} // namespace dwarf
} // namespace lldb_private::plugin

#endif
