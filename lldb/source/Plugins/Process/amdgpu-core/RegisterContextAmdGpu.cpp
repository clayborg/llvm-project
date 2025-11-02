//===-- RegisterContextAmdGpu.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextAmdGpu.h"

#include "ThreadAMDGPU.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextAmdGpu::RegisterContextAmdGpu(Thread &thread)
    : RegisterContext(thread, 0) {
  ThreadAMDGPU *amdgpu_thread = static_cast<ThreadAMDGPU *>(&thread);
  amd_dbgapi_architecture_id_t architecture_id =
      amdgpu_thread->GetArchitectureId();

  m_impl = std::make_unique<RegisterContextAmdGpuImpl>(architecture_id);
  m_impl->InitializeRegisterInfo();
  m_impl->InitializeRegisterData();
}

void RegisterContextAmdGpu::InvalidateAllRegisters() {
  m_impl->InvalidateAllRegisters();
}

size_t RegisterContextAmdGpu::GetRegisterCount() {
  return m_impl->GetRegisterCount();
}

size_t RegisterContextAmdGpu::GetRegisterSetCount() {
  return m_impl->GetRegisterSetCount();
}

const RegisterSet *RegisterContextAmdGpu::GetRegisterSet(size_t set_index) {
  return m_impl->GetRegisterSet(set_index);
}

const RegisterInfo *RegisterContextAmdGpu::GetRegisterInfoAtIndex(size_t reg) {
  return m_impl->GetRegisterInfoAtIndex(reg);
}

bool RegisterContextAmdGpu::ReadRegister(const RegisterInfo *reg_info,
                                         RegisterValue &reg_value) {
  ThreadAMDGPU *amdgpu_thread = static_cast<ThreadAMDGPU *>(&m_thread);
  auto wave_id = amdgpu_thread->GetWaveId();

  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  if (!m_impl->IsRegisterValid(lldb_reg_num)) {
    Status error = m_impl->ReadRegister(wave_id, reg_info);
    if (error.Fail())
      return false;
  }

  return m_impl->GetRegisterValue(reg_info, reg_value).Success();
}

bool RegisterContextAmdGpu::WriteRegister(const RegisterInfo *reg_info,
                                          const RegisterValue &reg_value) {
  return m_impl->WriteRegister(reg_info, reg_value).Success();
}

bool RegisterContextAmdGpu::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ThreadAMDGPU *amdgpu_thread = static_cast<ThreadAMDGPU *>(&m_thread);
  auto wave_id = amdgpu_thread->GetWaveId();

  Status error = m_impl->ReadAllRegisters(wave_id);
  if (error.Fail())
    return false;

  const size_t buffer_size = m_impl->GetRegisterBufferSize();
  data_sp.reset(new DataBufferHeap(buffer_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  memcpy(dst, m_impl->GetRegisterDataBuffer(), buffer_size);
  return true;
}

bool RegisterContextAmdGpu::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  const size_t buffer_size = m_impl->GetRegisterBufferSize();

  if (!data_sp || data_sp->GetByteSize() != buffer_size || !data_sp->GetBytes())
    return false;

  memcpy(m_impl->GetRegisterDataBuffer(), data_sp->GetBytes(), buffer_size);
  return true;
}
