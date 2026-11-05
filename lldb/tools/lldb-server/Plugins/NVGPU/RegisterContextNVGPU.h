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
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace lldb_private::lldb_server {

class ThreadNVGPU;
class ThreadState;

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

  /// Invalidate this lane's cached regular and predicate registers.
  void InvalidateAllRegisters();

private:
  llvm::Error ReadRegisterValue(const RegisterInfo &reg_info,
                                RegisterValue &value);

  /// \return
  ///     Cached R0-based values; empty on failure.
  llvm::ArrayRef<uint32_t> GetRegularRegisters(const ThreadState &thread_state);

  /// \return
  ///     Cached P0-based values; empty on failure.
  llvm::ArrayRef<uint32_t>
  GetPredicateRegisters(const ThreadState &thread_state);

  CUDBGAPI GetDebuggerAPI();

  ThreadNVGPU &GetGPUThread();

  /// Unset until read; empty after failure or unavailability.
  /// Reset on lane decode, allowing failed reads to be retried.
  /// Allocate on demand: unused lane slots also have register contexts.
  std::optional<llvm::SmallVector<uint32_t, 0>> m_regular_registers;
  std::optional<llvm::SmallVector<uint32_t, 0>> m_predicate_registers;
};

} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_REGISTERCONTEXTNVGPU_H
