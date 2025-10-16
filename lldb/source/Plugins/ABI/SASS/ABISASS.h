//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ABI_SASS_ABISASS_H
#define LLDB_SOURCE_PLUGINS_ABI_SASS_ABISASS_H

#include "lldb/Target/ABI.h"

class ABISASS : public lldb_private::RegInfoBasedABI {
public:
  ~ABISASS() override = default;

  size_t GetRedZoneSize() const override {
    // SASS doesn't have a red zone concept like x86.
    return 0;
  }

  bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                          lldb::addr_t func_addr, lldb::addr_t returnAddress,
                          llvm::ArrayRef<lldb::addr_t> args) const override {
    // TODO: SASS doesn't support traditional function calls.
    return false;
  }

  bool GetArgumentValues(lldb_private::Thread &thread,
                         lldb_private::ValueList &values) const override {
    // TODO: SASS argument passing.
    return false;
  }

  lldb_private::Status
  SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                       lldb::ValueObjectSP &new_value) override;

  lldb::UnwindPlanSP CreateFunctionEntryUnwindPlan() override;

  lldb::UnwindPlanSP CreateDefaultUnwindPlan() override;

  bool RegisterIsVolatile(const lldb_private::RegisterInfo *reg_info) override;

  bool GetFallbackRegisterLocation(
      const lldb_private::RegisterInfo *reg_info,
      lldb_private::UnwindPlan::Row::AbstractRegisterLocation &unwind_regloc)
      override;

  bool CallFrameAddressIsValid(lldb::addr_t cfa) override {
    // Make sure the stack call frame addresses are 8 byte aligned.
    // SASS always uses 64-bit addressing.
    if (cfa & 0x7ull)
      return false; // Not 8 byte aligned.
    if (cfa == 0)
      return false; // Zero is not a valid stack address.
    return true;
  }

  bool CodeAddressIsValid(lldb::addr_t pc) override {
    // TODO: Lookup address in cubin module to ensure it is valid.
    return true;
  }

  lldb::addr_t FixCodeAddress(lldb::addr_t pc) override { return pc; }

  const lldb_private::RegisterInfo *
  GetRegisterInfoArray(uint32_t &count) override {
    // Register information is provided by the server plugin
    // and passed to the client dynamically, so we don't maintain static
    // definitions.
    count = 0;
    return nullptr;
  }

  std::optional<uint64_t>
  GetDefaultAddressSpaceForSavedRegisters() const override;

  // Static Functions.

  static void Initialize();

  static void Terminate();

  static lldb::ABISP CreateInstance(lldb::ProcessSP process_sp,
                                    const lldb_private::ArchSpec &arch);

  static llvm::StringRef GetPluginNameStatic() { return "sass"; }

  // PluginInterface protocol.

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  lldb::ValueObjectSP
  GetReturnValueObjectImpl(lldb_private::Thread &thread,
                           lldb_private::CompilerType &ast_type) const override;

private:
  using lldb_private::RegInfoBasedABI::RegInfoBasedABI;
};

#endif // LLDB_SOURCE_PLUGINS_ABI_SASS_ABISASS_H
