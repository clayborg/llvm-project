//===-- RegisterContextNVGPUCore.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H

#include "NVGPUCoreData.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"

class ObjectFileELF;

namespace lldb_private {

class RegisterContextNVGPUCore : public RegisterContext {
public:
  RegisterContextNVGPUCore(Thread &thread, NVGPUCoreData &core_data,
                           const NVGPULaneCoords &coords,
                           ObjectFileELF *core);

  ~RegisterContextNVGPUCore() override;

  void InvalidateAllRegisters() override {}

  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

private:
  void LoadRegistersFromCore();

  NVGPUCoreData &m_core_data;
  ObjectFileELF *m_core;
  NVGPULaneCoords m_coords;

  uint64_t m_pc = 0;
  uint64_t m_error_pc = 0;
  uint32_t m_gp_regs[sass::kNumRRegs] = {};
  uint32_t m_pred_regs[sass::kNumPRegs] = {};
  uint32_t m_uniform_regs[sass::kNumURRegs] = {};
  uint32_t m_uniform_pred_regs[sass::kNumUPRegs] = {};
  uint32_t m_num_gp_regs = 0;
  uint32_t m_num_pred_regs = 0;
  uint32_t m_num_uniform_regs = 0;
  uint32_t m_num_uniform_pred_regs = 0;
  bool m_loaded = false;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
