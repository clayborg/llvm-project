//===-- RegisterContextNVGPUCore.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVGPUCore.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/RegisterValue.h"

#include <algorithm>

using namespace lldb;
using namespace lldb_private;

RegisterContextNVGPUCore::RegisterContextNVGPUCore(
    Thread &thread, NVGPUCoreData &core_data,
    const NVGPULaneCoords &coords, ObjectFileELF *core)
    : RegisterContext(thread, 0), m_core_data(core_data), m_core(core),
      m_coords(coords) {}

RegisterContextNVGPUCore::~RegisterContextNVGPUCore() = default;

void RegisterContextNVGPUCore::LoadRegistersFromCore() {
  if (m_loaded || !m_core)
    return;

  m_loaded = true;

  if (m_coords.dev_idx >= m_core_data.devices.size())
    return;
  
  DeviceData &dev = m_core_data.devices[m_coords.dev_idx];

  m_num_gp_regs = std::min(dev.entry.numRegsPerLane, sass::kNumRRegs);
  m_num_pred_regs = std::min(dev.entry.numPredicatesPrLane, sass::kNumPRegs);
  m_num_uniform_regs =
      std::min(dev.entry.numUniformRegsPrWarp, sass::kNumURRegs);
  m_num_uniform_pred_regs =
      std::min(dev.entry.numUniformPredicatesPrWarp, sass::kNumUPRegs);

  llvm::ArrayRef<SMData> sms = dev.GetSMs(m_core);
  if (m_coords.sm_idx >= sms.size())
    return;
  const SMData &sm = sms[m_coords.sm_idx];
  llvm::ArrayRef<CTAData> ctas = sm.GetCTAs(m_core);
  if (m_coords.cta_idx >= ctas.size())
    return;
  const CTAData &cta = ctas[m_coords.cta_idx];
  llvm::ArrayRef<WarpData> warps = cta.GetWarps(m_core);
  if (m_coords.warp_idx >= warps.size())
    return;
  const WarpData &warp = warps[m_coords.warp_idx];
  llvm::ArrayRef<LaneData> lanes = warp.GetLanes(m_core);
  if (m_coords.lane_idx >= lanes.size())
    return;
  const LaneData &lane = lanes[m_coords.lane_idx];

  m_pc = lane.entry.virtualPC;

  if (warp.entry.errorPCValid)
    m_error_pc = warp.entry.errorPC;

  // Per-lane registers (lazy, zero-copy from section tree)
  const DataExtractor &regs_data = lane.GetRegisters(m_core);
  if (regs_data.GetByteSize() > 0) {
    size_t avail = regs_data.GetByteSize() / sizeof(uint32_t);
    size_t count = std::min(avail, static_cast<size_t>(m_num_gp_regs));
    offset_t off = 0;
    regs_data.GetU32(&off, m_gp_regs, count);
  }

  const DataExtractor &preds_data = lane.GetPredicates(m_core);
  if (preds_data.GetByteSize() > 0) {
    size_t avail = preds_data.GetByteSize() / sizeof(uint32_t);
    size_t count = std::min(avail, static_cast<size_t>(m_num_pred_regs));
    offset_t off = 0;
    preds_data.GetU32(&off, m_pred_regs, count);
  }

  // Per-warp uniform registers (lazy, zero-copy from section tree)
  const DataExtractor &uregs_data = warp.GetUniformRegisters(m_core);
  if (uregs_data.GetByteSize() > 0) {
    size_t avail = uregs_data.GetByteSize() / sizeof(uint32_t);
    size_t count = std::min(avail, static_cast<size_t>(m_num_uniform_regs));
    offset_t off = 0;
    uregs_data.GetU32(&off, m_uniform_regs, count);
  }

  const DataExtractor &upreds_data = warp.GetUniformPredicates(m_core);
  if (upreds_data.GetByteSize() > 0) {
    size_t avail = upreds_data.GetByteSize() / sizeof(uint32_t);
    size_t count =
        std::min(avail, static_cast<size_t>(m_num_uniform_pred_regs));
    offset_t off = 0;
    upreds_data.GetU32(&off, m_uniform_pred_regs, count);
  }
}

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

  LoadRegistersFromCore();

  uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

  if (reg == sass::LLDB_PC) {
    reg_value.SetUInt64(m_pc);
    return true;
  }
  if (reg == sass::LLDB_ERROR_PC) {
    reg_value.SetUInt64(m_error_pc);
    return true;
  }
  if (reg == sass::LLDB_SP) {
    uint32_t idx = sass::SASS_SP_REG;
    reg_value.SetUInt32(idx < m_num_gp_regs ? m_gp_regs[idx] : 0);
    return true;
  }
  if (reg == sass::LLDB_FP) {
    uint32_t idx = sass::SASS_FP_REG;
    reg_value.SetUInt32(idx < m_num_gp_regs ? m_gp_regs[idx] : 0);
    return true;
  }
  if (reg == sass::LLDB_RA) {
    uint32_t lo = sass::SASS_RA_REG_LO;
    uint32_t hi = sass::SASS_RA_REG_HI;
    uint64_t ra = 0;
    if (lo < m_num_gp_regs && hi < m_num_gp_regs)
      ra = static_cast<uint64_t>(m_gp_regs[hi]) << 32 | m_gp_regs[lo];
    reg_value.SetUInt64(ra);
    return true;
  }
  if (reg >= sass::LLDB_R0 && reg < sass::LLDB_R0 + sass::kNumRRegs) {
    uint32_t idx = reg - sass::LLDB_R0;
    reg_value.SetUInt32(idx < m_num_gp_regs ? m_gp_regs[idx] : 0);
    return true;
  }
  if (reg == sass::LLDB_RZ) {
    reg_value.SetUInt32(0);
    return true;
  }
  if (reg >= sass::LLDB_P0 && reg < sass::LLDB_P0 + sass::kNumPRegs) {
    uint32_t idx = reg - sass::LLDB_P0;
    reg_value.SetUInt32(idx < m_num_pred_regs ? m_pred_regs[idx] : 0);
    return true;
  }
  if (reg >= sass::LLDB_UR0 && reg < sass::LLDB_UR0 + sass::kNumURRegs) {
    uint32_t idx = reg - sass::LLDB_UR0;
    reg_value.SetUInt32(idx < m_num_uniform_regs ? m_uniform_regs[idx] : 0);
    return true;
  }
  if (reg == sass::LLDB_URZ) {
    reg_value.SetUInt32(0);
    return true;
  }
  if (reg >= sass::LLDB_UP0 && reg < sass::LLDB_UP0 + sass::kNumUPRegs) {
    uint32_t idx = reg - sass::LLDB_UP0;
    reg_value.SetUInt32(
        idx < m_num_uniform_pred_regs ? m_uniform_pred_regs[idx] : 0);
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
  LoadRegistersFromCore();

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
