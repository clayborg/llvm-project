//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABISASS.h"

#include <optional>

#include "cudadebugger.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "tools/lldb-server/Plugins/NVGPU/SASSRegisterNumbers.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ABISASS)

std::optional<uint64_t>
ABISASS::GetDefaultAddressSpaceForSavedRegisters() const {
  // SASS uses local address space for saved registers
  return ptxLocalStorage;
}

Status ABISASS::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                     lldb::ValueObjectSP &new_value) {
  return Status::FromErrorString(
      "Setting return values not supported for SASS");
}

ValueObjectSP ABISASS::GetReturnValueObjectImpl(Thread &thread,
                                                CompilerType &ast_type) const {
  return ValueObjectSP();
}

UnwindPlanSP ABISASS::CreateFunctionEntryUnwindPlan() {
  UnwindPlan::Row row;

  // Use LLDB register numbers to support the virtual RA register
  const uint32_t lldb_sp = sass::LLDB_SP;
  const uint32_t lldb_ra = sass::LLDB_RA;

  // At function entry, CFA is the stack pointer
  row.GetCFAValue().SetIsRegisterPlusOffset(lldb_sp, 0);
  row.SetUnspecifiedRegistersAreUndefined(true);

  // Return address is in the RA register (R20-R21)
  row.SetRegisterLocationToRegister(lldb_ra, lldb_ra, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindLLDB);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("sass at-func-entry default");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  plan_sp->SetUnwindPlanForSignalTrap(eLazyBoolNo);
  plan_sp->SetReturnAddressRegister(lldb_ra);

  return plan_sp;
}

UnwindPlanSP ABISASS::CreateDefaultUnwindPlan() {
  // Per SASS ABI: Before a function modifies a preserved (callee-save)
  // register, it must save the value. This includes R2 (when used as FP)
  // and R20-R21 (return address), which are stored in the Local Storage area.
  // SASS frame pointer usage is optional - FP may be 0 if omitted.
  // Since we can't detect FP=0 at plan creation time (no access to register
  // values), and an FP-based plan would fail if FP=0, we return nullptr.
  //
  // This forces LLDB to rely on:
  // 1. DWARF CFI (if available) - most reliable.
  // 2. Function entry unwind plan - uses SP, works without FP.
  // 3. Architecture heuristics.
  //
  // This approach is similar to other architectures with optional frame
  // pointers (e.g., SystemZ).
  return nullptr;
}

bool ABISASS::RegisterIsVolatile(const RegisterInfo *reg_info) {
  // Default SASS Register ABI:
  // - Scratch (volatile/caller-save): r0, r3, r4-r15, r32-r35, r40-r43,
  //   r48-r51, r56-r59, and groups r64-r67, r72-r75, etc (every other group of
  //   4).
  // - Preserved (non-volatile/callee-save): r2, r16-r31, r36-r39, r44-r47,
  //   r52-r55, r60-r62, and groups r68-r71, r76-r79, etc (every other group of
  //   4).
  // - Special: r1 (stack pointer), r2 (frame pointer when used),
  //   r20-r21 (64-bit return address), r255 (zero).
  //
  // NOTE: This implements the DEFAULT ABI. SASS allows customization of:
  // - Parameter registers (default: start at r4).
  // - Return address register (default: r20-r21).
  // - Caller/callee save classification.
  //
  // TODO: Custom ABIs require runtime detection.

  if (!reg_info)
    return true;

  const uint32_t dwarf_regnum = reg_info->kinds[eRegisterKindDWARF];
  if (dwarf_regnum == LLDB_INVALID_REGNUM)
    return true;

  // For CUDA-encoded registers, extract the register number.
  const uint32_t reg_class = sass::GetDWARFRegisterClass(dwarf_regnum);
  const uint32_t reg_num = sass::GetDWARFRegisterNumber(dwarf_regnum);

  // Handle non-CUDA encoded special registers.
  if (reg_class == 0) {
    // PC, ErrorPC, etc are considered volatile.
    return true;
  }

  // Handle regular registers.
  if (reg_class == REG_CLASS_REG_FULL) {
    // Special registers.
    if (reg_num == sass::SASS_SP_REG)   // r1: Stack pointer.
      return false;                     // Preserved
    if (reg_num == sass::SASS_FP_REG)   // r2: Frame pointer (when used).
      return false;                     // Preserved
    if (reg_num == sass::SASS_ZERO_REG) // r255: Zero register.
      return false;                     // Special, never changes

    // Preserved (callee-save) registers.
    if (reg_num >= 16 && reg_num <= 31) // r16-r31
      return false;
    if (reg_num >= 36 && reg_num <= 39) // r36-r39
      return false;
    if (reg_num >= 44 && reg_num <= 47) // r44-r47
      return false;
    if (reg_num >= 52 && reg_num <= 55) // r52-r55
      return false;
    if (reg_num >= 60 && reg_num <= 62) // r60-r62
      return false;

    // Preserved registers in groups of 4.
    for (uint32_t base = 68; base <= 252; base += 8) {
      if (reg_num >= base && reg_num <= base + 3)
        return false; // r68-r71, r76-r79, etc.
    }

    // All other registers are scratch (caller-save/volatile).
    return true;
  }

  // Handle uniform registers.
  if (reg_class == REG_CLASS_UREG_FULL) {
    // UR255 is the uniform zero register.
    if (reg_num == sass::SASS_ZERO_REG)
      return false; // Special, never changes.

    // All other uniform registers are considered volatile.
    return true;
  }

  // For any other register class, assume volatile.
  return true;
}

// Static Functions.

ABISP
ABISASS::CreateInstance(lldb::ProcessSP process_sp, const ArchSpec &arch) {

  if (arch.IsNVPTX())
    return ABISP(new ABISASS(std::move(process_sp), MakeMCRegisterInfo(arch)));

  return ABISP();
}

void ABISASS::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "ABI for SASS targets (64-bit)", CreateInstance);
}

void ABISASS::Terminate() { PluginManager::UnregisterPlugin(CreateInstance); }
