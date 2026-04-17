//===-- RegisterContextNVGPUCore.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVGPUCore.h"
#include "ThreadNVGPUCore.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"
#include "lldb/Utility/RegisterValue.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;

RegisterContextNVGPUCore::RegisterContextNVGPUCore(Thread &thread,
                                                   ObjectFileELF *core)
    : RegisterContext(thread, 0), m_core(core) {
  // Borrow the pre-resolved hierarchy pointers from the owning thread.
  // They outlive this RegisterContext: they point into NVGPUCoreData's
  // lazy vectors, which are owned by the Process and never repopulated.
  auto &nvgpu_thread = static_cast<ThreadNVGPUCore &>(thread);
  m_warp = nvgpu_thread.GetWarp();
  m_lane = nvgpu_thread.GetLane();

  // If the thread didn't fully resolve a lane, leave m_num_* at zero; all
  // ReadRegister bounds checks will then fail and reads return 0.
  if (!m_core || !m_lane)
    return;

  // m_warp and GetDevice() are guaranteed non-null here via the thread's
  // nested-null invariant (m_lane => m_warp => ... => m_dev). Compute the
  // register counts as the minimum of device-advertised and what the sections
  // actually hold. After this, `idx < m_num_*` at every ReadRegister call
  // site is the single authoritative bound: it implies both that the pointer
  // is non-null and that the section has at least `(idx + 1) * 4` bytes.
  const CudbgDeviceTableEntry &dev_entry = nvgpu_thread.GetDevice()->GetEntry();
  auto clamp = [](uint32_t cap, const DataExtractor &data) {
    return std::min<uint32_t>(cap, data.GetByteSize() / sizeof(uint32_t));
  };
  m_num_gp_regs = clamp(std::min(dev_entry.numRegsPerLane, sass::kNumRRegs),
                        m_lane->GetRegisters(m_core));
  m_num_pred_regs =
      clamp(std::min(dev_entry.numPredicatesPrLane, sass::kNumPRegs),
            m_lane->GetPredicates(m_core));
  m_num_uniform_regs =
      clamp(std::min(dev_entry.numUniformRegsPrWarp, sass::kNumURRegs),
            m_warp->GetUniformRegisters(m_core));
  m_num_uniform_pred_regs =
      clamp(std::min(dev_entry.numUniformPredicatesPrWarp, sass::kNumUPRegs),
            m_warp->GetUniformPredicates(m_core));
}

RegisterContextNVGPUCore::~RegisterContextNVGPUCore() = default;

size_t RegisterContextNVGPUCore::GetRegisterCount() {
  return sass::GetRegisterInfos().size();
}

const lldb_private::RegisterInfo *
RegisterContextNVGPUCore::GetRegisterInfoAtIndex(size_t reg) {
  llvm::ArrayRef<lldb_private::RegisterInfo> infos = sass::GetRegisterInfos();
  if (reg < infos.size())
    return &infos[reg];
  return nullptr;
}

size_t RegisterContextNVGPUCore::GetRegisterSetCount() {
  return sass::GetRegisterSets().size();
}

const lldb_private::RegisterSet *
RegisterContextNVGPUCore::GetRegisterSet(size_t set) {
  llvm::ArrayRef<lldb_private::RegisterSet> sets = sass::GetRegisterSets();
  if (set < sets.size())
    return &sets[set];
  return nullptr;
}

bool RegisterContextNVGPUCore::ReadRegister(const RegisterInfo *reg_info,
                                            RegisterValue &reg_value) {
  if (!reg_info) {
    LLDB_LOG(GetLog(LLDBLog::Process),
             "RegisterContextNVGPUCore::ReadRegister called with null "
             "reg_info");
    return false;
  }

  uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

  if (reg == sass::LLDB_PC) {
    reg_value.SetUInt64(m_lane ? m_lane->GetEntry().virtualPC : 0);
    return true;
  }
  if (reg == sass::LLDB_ERROR_PC) {
    uint64_t err_pc = 0;
    if (m_warp && m_warp->GetEntry().errorPCValid)
      err_pc = m_warp->GetEntry().errorPC;
    reg_value.SetUInt64(err_pc);
    return true;
  }
  // Ctor-side clamping makes `idx < m_num_*` imply the section has at least
  // `(idx + 1) * 4` bytes, so GetU32_unchecked is safe at every call site.
  if (reg == sass::LLDB_SP) {
    uint32_t idx = sass::SASS_SP_REG;
    uint32_t val = 0;
    if (idx < m_num_gp_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_lane->GetRegisters(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }
  if (reg == sass::LLDB_FP) {
    uint32_t idx = sass::SASS_FP_REG;
    uint32_t val = 0;
    if (idx < m_num_gp_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_lane->GetRegisters(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }
  if (reg == sass::LLDB_RA) {
    uint32_t lo = sass::SASS_RA_REG_LO;
    uint32_t hi = sass::SASS_RA_REG_HI;
    uint64_t ra = 0;
    if (lo < m_num_gp_regs && hi < m_num_gp_regs) {
      const DataExtractor &regs = m_lane->GetRegisters(m_core);
      offset_t off_hi = hi * sizeof(uint32_t);
      offset_t off_lo = lo * sizeof(uint32_t);
      ra = static_cast<uint64_t>(regs.GetU32_unchecked(&off_hi)) << 32 |
           regs.GetU32_unchecked(&off_lo);
    }
    reg_value.SetUInt64(ra);
    return true;
  }
  if (reg >= sass::LLDB_R0 && reg < sass::LLDB_R0 + sass::kNumRRegs) {
    uint32_t idx = reg - sass::LLDB_R0;
    uint32_t val = 0;
    if (idx < m_num_gp_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_lane->GetRegisters(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }
  if (reg == sass::LLDB_RZ) {
    reg_value.SetUInt32(0);
    return true;
  }
  if (reg >= sass::LLDB_P0 && reg < sass::LLDB_P0 + sass::kNumPRegs) {
    uint32_t idx = reg - sass::LLDB_P0;
    uint32_t val = 0;
    if (idx < m_num_pred_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_lane->GetPredicates(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }
  if (reg >= sass::LLDB_UR0 && reg < sass::LLDB_UR0 + sass::kNumURRegs) {
    uint32_t idx = reg - sass::LLDB_UR0;
    uint32_t val = 0;
    if (idx < m_num_uniform_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_warp->GetUniformRegisters(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }
  if (reg == sass::LLDB_URZ) {
    reg_value.SetUInt32(0);
    return true;
  }
  if (reg >= sass::LLDB_UP0 && reg < sass::LLDB_UP0 + sass::kNumUPRegs) {
    uint32_t idx = reg - sass::LLDB_UP0;
    uint32_t val = 0;
    if (idx < m_num_uniform_pred_regs) {
      offset_t off = idx * sizeof(uint32_t);
      val = m_warp->GetUniformPredicates(m_core).GetU32_unchecked(&off);
    }
    reg_value.SetUInt32(val);
    return true;
  }

  LLDB_LOG(GetLog(LLDBLog::Process),
           "RegisterContextNVGPUCore::ReadRegister unhandled register {0}",
           reg);
  return false;
}

bool RegisterContextNVGPUCore::WriteRegister(const RegisterInfo *reg_info,
                                             const RegisterValue &reg_value) {
  return false;
}

bool RegisterContextNVGPUCore::ReadAllRegisterValues(
    WritableDataBufferSP &data_sp) {
  const size_t reg_count = GetRegisterCount();
  const size_t bytes_per_reg = 8;
  const size_t buf_size = reg_count * bytes_per_reg;
  data_sp = std::make_shared<DataBufferHeap>(buf_size, 0);
  uint8_t *dst = data_sp->GetBytes();

  for (size_t i = 0; i < reg_count; ++i) {
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(i);
    if (!reg_info)
      continue;
    RegisterValue reg_value;
    if (!ReadRegister(reg_info, reg_value))
      continue;
    Status error;
    if (reg_value.GetAsMemoryData(*reg_info, dst + i * bytes_per_reg,
                                  bytes_per_reg, eByteOrderLittle, error) == 0)
      LLDB_LOG(GetLog(LLDBLog::Process),
               "ReadAllRegisterValues: failed to serialize register {0}: {1}",
               reg_info->name, error);
  }
  return true;
}

bool RegisterContextNVGPUCore::WriteAllRegisterValues(
    const DataBufferSP &data_sp) {
  return false;
}
