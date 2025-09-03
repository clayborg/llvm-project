//===----------------------------------------------------------------------===//
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

namespace lldb_private::lldb_server {

class ThreadNVIDIAGPU;

static constexpr size_t kNumRRegs = 32;

/// Store all the registers for a single thread.
struct ThreadRegistersValues {
  uint64_t PC;
  uint64_t errorPC;
  uint32_t R[kNumRRegs];
};

/// Store the validity of the registers.
struct ThreadRegisterValidity {
  bool PC;
  bool errorPC;
  bool R[kNumRRegs];

  ThreadRegisterValidity();
};

struct ThreadRegistersWithValidity {
  ThreadRegisterValidity is_valid;
  ThreadRegistersValues val;
};

class RegisterContextNVIDIAGPU : public NativeRegisterContext {
public:
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

  std::optional<uint64_t> ReadErrorPC();

  /// Invalidate all registers. Future accessess will cause reads from the
  /// device.
  void InvalidateAllRegisters();

private:
  /// Read the registers from the device. The results are cached. Any failures
  /// to read individual registers are signaled in invalid states of the
  /// registers.
  const ThreadRegistersWithValidity &ReadAllRegsFromDevice();

  CUDBGAPI GetDebuggerAPI();

  ThreadNVIDIAGPU &GetGPUThread();

  std::optional<ThreadRegistersWithValidity> m_regs;
};

} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVIDIAGPU_H
