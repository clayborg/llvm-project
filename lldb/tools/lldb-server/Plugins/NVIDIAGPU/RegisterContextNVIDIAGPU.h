//===-- RegisterContextNVIDIAGPU.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVIDIAGPU_H
#define LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVIDIAGPU_H

#include "cudadebugger.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/lldb-forward.h"

#include <bitset>

namespace lldb_private::lldb_server {

class ThreadNVIDIAGPU;

class RegisterContextNVIDIAGPU : public NativeRegisterContext {
public:
  /// Only PC and errorPC are 64-bit, all the R registers are 32-bit, but we
  /// store them in a 64-bit array for simplicity.
  struct RegisterContext {
    int64_t PC;
    int64_t errorPC;
    int64_t SP; // r1
    int64_t FP; // r2
  };

  RegisterContextNVIDIAGPU(ThreadNVIDIAGPU &thread);

  uint32_t GetRegisterCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterInfo *GetRegisterInfoAtIndex(uint32_t reg) const override;

  uint32_t GetRegisterSetCount() const override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  std::vector<uint32_t>
  GetExpeditedRegisters(ExpeditedRegs expType) const override;

  /// Invalidate all registers. Future accessess will cause reads from the
  /// device.
  void InvalidateAllRegisters();

private:
  /// Read the registers from the device. The results are cached. Any failures
  /// to read individual registers are signaled in invalid states of the
  /// registers.
  void ReadAllRegsFromDevice();

  CUDBGAPI GetDebuggerAPI();

  ThreadNVIDIAGPU &GetGPUThread();

  union {
    int64_t data[sizeof(RegisterContext)]; // Allow for indexed access.
    RegisterContext regs;
  } m_regs;

  std::bitset<sizeof(RegisterContext) / sizeof(int64_t)> m_regs_value_is_valid;
  bool m_did_read_already;
};

} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVIDIAGPU_H
