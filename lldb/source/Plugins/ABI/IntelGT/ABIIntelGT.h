//===-- ABIIntelGT.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ABI_INTELGT_ABIINTELGT_H
#define LLDB_SOURCE_PLUGINS_ABI_INTELGT_ABIINTELGT_H

#include "lldb/Target/ABI.h"

namespace lldb_private {

class ABIIntelGT : public lldb_private::MCBasedABI {
public:
  ~ABIIntelGT() override = default;

  size_t GetRedZoneSize() const override { return 0; }

  bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                          lldb::addr_t func_addr, lldb::addr_t return_addr,
                          llvm::ArrayRef<lldb::addr_t> args) const override {
    return false;
  }

  bool GetArgumentValues(lldb_private::Thread &thread,
                         lldb_private::ValueList &values) const override {
    return false;
  }

  lldb_private::Status
  SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                       lldb::ValueObjectSP &new_value_sp) override {
    return lldb_private::Status::FromErrorString("not implemented");
  }

  lldb::ValueObjectSP
  GetReturnValueObjectImpl(lldb_private::Thread &thread,
                           lldb_private::CompilerType &type) const override {
    return nullptr;
  }

  // Real unwinding is done by the ArchitectureIntelGT framedesc plugin;
  // return a non-null but empty UnwindPlan since core LLDB dereferences
  // the result without a null check.
  lldb::UnwindPlanSP CreateFunctionEntryUnwindPlan() override;

  lldb::UnwindPlanSP CreateDefaultUnwindPlan() override;

  bool RegisterIsVolatile(const lldb_private::RegisterInfo *reg_info) override {
    return false;
  }

  bool CallFrameAddressIsValid(lldb::addr_t cfa) override { return cfa != 0; }

  bool CodeAddressIsValid(lldb::addr_t pc) override { return true; }

  uint32_t GetGenericNum(llvm::StringRef reg) override {
    return LLDB_INVALID_REGNUM;
  }

  // Saved registers live in GPU global memory (address space 0).
  std::optional<uint64_t>
  GetDefaultAddressSpaceForSavedRegisters() const override {
    return 0;
  }

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb::ABISP CreateInstance(lldb::ProcessSP process_sp,
                                    const lldb_private::ArchSpec &arch);

  static llvm::StringRef GetPluginNameStatic() { return "intelgt"; }

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  void AugmentRegisterInfo(
      std::vector<lldb_private::DynamicRegisterInfo::Register> &regs) override {
  }

private:
  using lldb_private::MCBasedABI::MCBasedABI;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_ABI_INTELGT_ABIINTELGT_H
