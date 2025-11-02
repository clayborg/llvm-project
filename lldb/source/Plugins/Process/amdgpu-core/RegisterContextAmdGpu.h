//===-- RegisterContextAmdGpu.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AMDGPU_CORE_REGISTERCONTEXTAMDGPU_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AMDGPU_CORE_REGISTERCONTEXTAMDGPU_H

#include "Plugins/Process/Utility/RegisterContextAmdGpuImpl.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-forward.h"
#include <memory>

namespace lldb_private {

class RegisterContextAmdGpu : public RegisterContext {
public:
  RegisterContextAmdGpu(Thread &thread);

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

private:
  std::unique_ptr<RegisterContextAmdGpuImpl> m_impl;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_AMDGPU_CORE_REGISTERCONTEXTAMDGPU_H
