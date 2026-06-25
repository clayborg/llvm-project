//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVGPU_H
#define LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVGPU_H

#include "cudadebugger.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/lldb-forward.h"

namespace lldb_private::lldb_server {

class ThreadNVGPU;

/// Store the values and validity of the thread registers.
struct ThreadRegisterCache {
  sass::ThreadRegisters val;
  sass::ThreadRegistersValidity is_valid;
};

/// Store the values and validity of warp-wide registers.
struct WarpSharedRegisterCache {
  sass::WarpSharedRegisters val;
  sass::WarpSharedRegistersValidity is_valid;
};

class RegisterContextNVGPU : public NativeRegisterContext {
public:
  RegisterContextNVGPU(ThreadNVGPU &thread);

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
  const ThreadRegisterCache &ReadAllRegsFromDevice();

  CUDBGAPI GetDebuggerAPI();

  ThreadNVGPU &GetGPUThread();

  std::optional<ThreadRegisterCache> m_regs;
};

} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVGPU_H
