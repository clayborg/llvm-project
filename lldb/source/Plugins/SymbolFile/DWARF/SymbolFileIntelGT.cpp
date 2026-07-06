//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileIntelGT.h"

#include "LogChannelDWARF.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"

#include "llvm/Support/FormatVariadic.h"

#include <cinttypes>

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

bool SymbolFileIntelGT::ParseVendorDWARFOpcode(
    uint8_t op, const DataExtractor &opcodes, lldb::offset_t &offset,
    ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
    lldb::RegisterKind reg_kind, std::vector<Value> &stack) const {
  Log *log = GetLog(DWARFLog::DebugInfo);

  switch (op) {

  // Push the current SIMD lane onto the stack (no inline operands).
  case DW_OP_INTEL_push_simd_lane: {
    uint32_t simd_lane = 0;
    if (exe_ctx) {
      if (Thread *thread = exe_ctx->GetThreadPtr()) {
        std::optional<lldb::tid_t> lane = thread->GetLaneID();
        if (lane)
          simd_lane = static_cast<uint32_t>(*lane);
      }
    }
    stack.push_back(Scalar(simd_lane));
    LLDB_LOGF(log, "DW_OP_INTEL_push_simd_lane: lane=%u", simd_lane);
    return true;
  }

  // Extract bit_size bits from a register. Operand: u8 bit_size.
  // Stack: [dwarf_reg, bit_offset] -> [value].
  case DW_OP_INTEL_regval_bits: {
    if (stack.size() < 2)
      return false;

    uint8_t bit_size = opcodes.GetU8(&offset);

    uint64_t bit_offset = stack.back().GetScalar().ULongLong();
    stack.pop_back();
    uint32_t dwarf_reg_num = stack.back().GetScalar().UInt();
    stack.pop_back();

    if (!reg_ctx)
      return false;

    uint32_t lldb_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(
        reg_kind, dwarf_reg_num);
    if (lldb_reg_num == LLDB_INVALID_REGNUM)
      return false;

    const RegisterInfo *reg_info =
        reg_ctx->GetRegisterInfoAtIndex(lldb_reg_num);
    if (!reg_info)
      return false;

    uint64_t reg_bit_size = reg_info->byte_size * 8;
    if (bit_offset + bit_size > reg_bit_size)
      return false;

    RegisterValue reg_value;
    if (!reg_ctx->ReadRegister(reg_info, reg_value))
      return false;

    // Intel GT is little-endian.
    constexpr lldb::ByteOrder byte_order = lldb::eByteOrderLittle;
    uint8_t reg_bytes[256];
    if (reg_info->byte_size > sizeof(reg_bytes))
      return false;

    Status error;
    if (reg_value.GetAsMemoryData(*reg_info, reg_bytes, reg_info->byte_size,
                                  byte_order, error) == 0)
      return false;

    DataExtractor reg_data(reg_bytes, reg_info->byte_size, byte_order,
                           /*addr_size=*/8);

    uint64_t byte_offset = bit_offset / 8;
    uint64_t byte_bit_offset = bit_offset % 8;
    uint64_t bytes_needed = (byte_bit_offset + bit_size + 7) / 8;

    if (byte_offset + bytes_needed > reg_info->byte_size)
      return false;

    lldb::offset_t data_offset = byte_offset;
    uint64_t extracted = reg_data.GetMaxU64_unchecked(&data_offset, bytes_needed);
    extracted >>= byte_bit_offset;
    uint64_t mask = (bit_size >= 64) ? ~0ULL : ((1ULL << bit_size) - 1);
    extracted &= mask;

    stack.push_back(Scalar(extracted));

    LLDB_LOGF(log,
              "DW_OP_INTEL_regval_bits: reg=%s[%u] bits[%" PRIu64
              ":%" PRIu64 ") = 0x%" PRIx64,
              reg_info->name, dwarf_reg_num, bit_offset,
              bit_offset + bit_size, extracted);
    return true;
  }

  default:
    return false;
  }
}
