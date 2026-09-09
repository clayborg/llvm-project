//===-- ArchitectureIntelGT.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArchitectureIntelGT.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/RegisterContextUnwind.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/UnwindLLDB.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/FormatVariadic.h"
#include <cinttypes>
#include <cstring>

// Intel GT vendor DWARF opcodes.
#define DW_OP_INTEL_push_simd_lane 0xed
#define DW_OP_INTEL_regval_bits    0xfe

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ArchitectureIntelGT)

void ArchitectureIntelGT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Intel GT GPU Architecture Plugin",
                                &ArchitectureIntelGT::Create);
}

void ArchitectureIntelGT::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureIntelGT::Create);
}

std::unique_ptr<Architecture>
ArchitectureIntelGT::Create(const ArchSpec &arch) {
  if (arch.GetTriple().getArch() == llvm::Triple::intelgt)
    return std::make_unique<ArchitectureIntelGT>();
  return nullptr;
}

lldb::UnwindPlanSP ArchitectureIntelGT::GetArchitectureUnwindPlan(
    Thread &thread, RegisterContextUnwind *regctx,
    std::shared_ptr<const UnwindPlan> current_unwindplan) {
  if (!regctx)
    return nullptr;

  Log *log = GetLog(LLDBLog::Unwind);
  LLDB_LOGF(log, "ArchitectureIntelGT::%s thread=0x%" PRIx64, __FUNCTION__,
            (uint64_t)thread.GetID());

  // r127 is the framedesc register; absent means no framedesc support.
  const RegisterInfo *r127_info = regctx->GetRegisterInfoByName("r127");
  if (!r127_info)
    return nullptr;

  uint32_t r127_regnum = r127_info->kinds[eRegisterKindLLDB];
  if (r127_regnum == LLDB_INVALID_REGNUM)
    return nullptr;

  // Use the previous frame's UnwindPlan location to avoid circularity.
  UnwindLLDB::ConcreteRegisterLocation r127_regloc = {};
  if (regctx->SavedLocationForRegister(r127_regnum, r127_regloc) !=
      UnwindLLDB::RegisterSearchResult::eRegisterFound)
    return nullptr;

  // r127 is a 256-bit framedesc register.
  RegisterValue r127_value;
  if (!regctx->ReadRegisterValueFromRegisterLocation(r127_regloc, r127_info,
                                                     r127_value))
    return nullptr;

  // Framedesc layout: return_ip[0:32] callmask[32:64] be_sp[64:96]
  // be_fp[96:128] fe_fp[128:192] fe_sp[192:256].
  uint8_t framedesc[32];
  Status error;
  size_t bytes_read = r127_value.GetAsMemoryData(*r127_info, framedesc, 32,
                                                 eByteOrderLittle, error);
  if (bytes_read != 32) {
    LLDB_LOGF(log,
              "ArchitectureIntelGT: framedesc extraction failed "
              "(got %zu bytes)",
              bytes_read);
    return nullptr;
  }

  uint32_t return_ip;
  memcpy(&return_ip, framedesc + 0, 4);

  if (return_ip == 0)
    return nullptr; // top of stack

  uint32_t be_fp;
  memcpy(&be_fp, framedesc + 12, 4); // bits 96:128

  // scrbase (scratch base) is needed for the CFA.
  uint64_t scrbase = 0;
  const RegisterInfo *scrbase_info = regctx->GetRegisterInfoByName("scrbase0");
  if (scrbase_info) {
    RegisterValue scrbase_value;
    if (regctx->ReadRegister(scrbase_info, scrbase_value))
      scrbase = scrbase_value.GetAsUInt64();
  }

  if (scrbase == 0) {
    LLDB_LOGF(log, "ArchitectureIntelGT: scrbase == 0, can't compute CFA");
    return nullptr;
  }

  uint64_t cfa = scrbase + be_fp;
  if (cfa == 0) {
    LLDB_LOGF(log, "ArchitectureIntelGT: CFA == 0, invalid");
    return nullptr;
  }

  // isabase (fallback: genstbase) is needed for the caller PC.
  uint64_t isabase = 0;
  const RegisterInfo *isabase_info = regctx->GetRegisterInfoByName("isabase");
  if (isabase_info) {
    RegisterValue isabase_value;
    if (regctx->ReadRegister(isabase_info, isabase_value))
      isabase = isabase_value.GetAsUInt64();
  }

  if (isabase == 0) {
    const RegisterInfo *genstbase_info =
        regctx->GetRegisterInfoByName("genstbase");
    if (genstbase_info) {
      RegisterValue genstbase_value;
      if (regctx->ReadRegister(genstbase_info, genstbase_value))
        isabase = genstbase_value.GetAsUInt64();
    }
  }

  if (isabase == 0) {
    LLDB_LOGF(log,
              "ArchitectureIntelGT: isabase == 0, can't compute caller PC");
    return nullptr;
  }

  uint64_t caller_pc = isabase + return_ip;

  LLDB_LOGF(log,
            "ArchitectureIntelGT: return_ip=0x%08x be_fp=0x%08x "
            "scrbase=0x%" PRIx64 " CFA=0x%" PRIx64 " isabase=0x%" PRIx64
            " caller_pc=0x%" PRIx64,
            return_ip, be_fp, scrbase, cfa, isabase, caller_pc);

  const RegisterInfo *pc_info = regctx->GetRegisterInfoByName("pc");
  if (!pc_info)
    return nullptr;

  // Synthetic UnwindPlan from the framedesc data.
  UnwindPlanSP unwind_plan_sp = std::make_shared<UnwindPlan>(eRegisterKindLLDB);
  unwind_plan_sp->SetSourceName("Intel GT framedesc");
  unwind_plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  unwind_plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  unwind_plan_sp->SetRegisterKind(eRegisterKindLLDB);

  UnwindPlan::Row row;
  row.GetCFAValue().SetIsConstant(cfa);

  UnwindPlan::Row::AbstractRegisterLocation pc_loc;
  pc_loc.SetIsConstant(caller_pc);
  row.SetRegisterInfo(pc_info->kinds[eRegisterKindLLDB], pc_loc);

  // Caller's framedesc lives at [CFA+0] in scratch memory.
  UnwindPlan::Row::AbstractRegisterLocation r127_loc;
  r127_loc.SetAtCFAPlusOffset(0);
  row.SetRegisterInfo(r127_info->kinds[eRegisterKindLLDB], r127_loc);

  row.SetOffset(0);
  unwind_plan_sp->AppendRow(row);

  return unwind_plan_sp;
}

bool ArchitectureIntelGT::ParseVendorDWARFOpcode(
    uint8_t op, const DataExtractor &opcodes, lldb::offset_t &offset,
    ExecutionContext *exe_ctx, RegisterContext *reg_ctx,
    lldb::RegisterKind reg_kind, std::vector<Value> &stack) const {
  Log *log = GetLog(LLDBLog::Expressions);

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

    uint32_t lldb_reg_num =
        reg_ctx->ConvertRegisterKindToRegisterNumber(reg_kind, dwarf_reg_num);
    if (lldb_reg_num == LLDB_INVALID_REGNUM)
      return false;

    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoAtIndex(lldb_reg_num);
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
    uint64_t extracted =
        reg_data.GetMaxU64_unchecked(&data_offset, bytes_needed);
    extracted >>= byte_bit_offset;
    uint64_t mask = (bit_size >= 64) ? ~0ULL : ((1ULL << bit_size) - 1);
    extracted &= mask;

    stack.push_back(Scalar(extracted));

    LLDB_LOGF(log,
              "DW_OP_INTEL_regval_bits: reg=%s[%u] bits[%" PRIu64 ":%" PRIu64
              ") = 0x%" PRIx64,
              reg_info->name, dwarf_reg_num, bit_offset, bit_offset + bit_size,
              extracted);
    return true;
  }

  default:
    return false;
  }
}
