//===-- RegisterContextAmdGpu.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextAmdGpu.h"

#include "LLDBServerPluginAMDGPU.h"
#include "ProcessAMDGPU.h"
#include "ThreadAMDGPU.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

RegisterContextAmdGpu::RegisterContextAmdGpu(
    NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread) {
  ThreadAMDGPU *thread = static_cast<ThreadAMDGPU *>(&native_thread);
  amd_dbgapi_architecture_id_t architecture_id =
      thread->GetProcess().m_debugger->m_architecture_id;

  m_impl = std::make_unique<RegisterContextAmdGpuImpl>(architecture_id);
}

void RegisterContextAmdGpu::InvalidateAllRegisters() {
  m_impl->InvalidateAllRegisters();
}

uint32_t RegisterContextAmdGpu::GetRegisterCount() const {
  return m_impl->GetRegisterCount();
}

uint32_t RegisterContextAmdGpu::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextAmdGpu::GetRegisterInfoAtIndex(uint32_t reg) const {
  return m_impl->GetRegisterInfoAtIndex(reg);
}

uint32_t RegisterContextAmdGpu::GetRegisterSetCount() const {
  return m_impl->GetRegisterSetCount();
}

const RegisterSet *
RegisterContextAmdGpu::GetRegisterSet(uint32_t set_index) const {
  return m_impl->GetRegisterSet(set_index);
}

Status RegisterContextAmdGpu::ReadRegister(const RegisterInfo *reg_info,
                                           RegisterValue &reg_value) {
  ThreadAMDGPU *thread = static_cast<ThreadAMDGPU *>(&m_thread);
  auto wave_id = thread->GetWaveID();

  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  if (!m_impl->IsRegisterValid(lldb_reg_num)) {
    Status error = m_impl->ReadRegister(wave_id, reg_info);
    if (error.Fail())
      return error;
  }

  return m_impl->GetRegisterValue(reg_info, reg_value);
}

Status RegisterContextAmdGpu::WriteRegister(const RegisterInfo *reg_info,
                                            const RegisterValue &reg_value) {
  return m_impl->WriteRegister(reg_info, reg_value);
}

Status RegisterContextAmdGpu::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ThreadAMDGPU *thread = static_cast<ThreadAMDGPU *>(&m_thread);
  auto wave_id = thread->GetWaveID();

  Status error = m_impl->ReadAllRegisters(wave_id);
  if (error.Fail())
    return error;

  const size_t buffer_size = m_impl->GetRegisterBufferSize();
  data_sp.reset(new DataBufferHeap(buffer_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  memcpy(dst, m_impl->GetRegisterDataBuffer(), buffer_size);
  return Status();
}

Status RegisterContextAmdGpu::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  const size_t buffer_size = m_impl->GetRegisterBufferSize();

  if (!data_sp) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAmdGpu::%s invalid data_sp provided", __FUNCTION__);
  }

  if (data_sp->GetByteSize() != buffer_size) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAmdGpu::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, buffer_size, data_sp->GetByteSize());
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextAmdGpu::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
  }
  memcpy(m_impl->GetRegisterDataBuffer(), src, buffer_size);
  return Status();
}

std::vector<uint32_t>
RegisterContextAmdGpu::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    // We can't expedite all registers because that would cause jThreadsInfo to
    // fetch registers from all stopped waves eagerly which would be too slow
    // and unnecessary.
    g_expedited_regs.push_back(m_impl->GetPCRegisterNumber());
  }
  return g_expedited_regs;
}
