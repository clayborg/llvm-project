//===-- ABIIntelGT.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIIntelGT.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ABIIntelGT)

UnwindPlanSP ABIIntelGT::CreateFunctionEntryUnwindPlan() {
  // Empty plan; the framedesc plugin does the real unwinding.
  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindLLDB);
  plan_sp->SetSourceName("intelgt empty function-entry unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  return plan_sp;
}

UnwindPlanSP ABIIntelGT::CreateDefaultUnwindPlan() {
  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindLLDB);
  plan_sp->SetSourceName("intelgt empty default unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  return plan_sp;
}

void ABIIntelGT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Intel GT GPU ABI",
                                &ABIIntelGT::CreateInstance);
}

void ABIIntelGT::Terminate() {
  PluginManager::UnregisterPlugin(&ABIIntelGT::CreateInstance);
}

lldb::ABISP ABIIntelGT::CreateInstance(lldb::ProcessSP process_sp,
                                       const ArchSpec &arch) {
  if (arch.GetTriple().getArch() == llvm::Triple::spirv64) {
    // Registers are discovered dynamically; a dummy MCRegisterInfo suffices.
    auto mc_reg_info = std::make_unique<llvm::MCRegisterInfo>();
    mc_reg_info->InitMCRegisterInfo(nullptr, // MCRegisterDesc *D
                                    0,       // unsigned NR (NumRegs)
                                    0,       // unsigned RA (RAReg)
                                    0,       // unsigned PC (PCReg)
                                    nullptr, // MCRegisterClass *C
                                    0,       // unsigned NC (NumClasses)
                                    nullptr, // MCPhysReg (*RURoots)[2]
                                    0,       // unsigned NRU
                                    nullptr, // int16_t *DL
                                    nullptr, // LaneBitmask *RUMS
                                    nullptr, // const char *Strings
                                    nullptr, // const char *ClassStrings
                                    nullptr, // uint16_t *SubIndices
                                    0,       // unsigned NumIndices
                                    nullptr  // uint16_t *RET
    );
    return ABISP(new ABIIntelGT(std::move(process_sp), std::move(mc_reg_info)));
  }
  return ABISP();
}
