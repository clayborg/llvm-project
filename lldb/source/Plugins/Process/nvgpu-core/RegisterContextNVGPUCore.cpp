//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVGPUCore.h"
#include "CudbgEntryParser.h"
#include "SectionUtils.h"
#include "ThreadNVGPUCore.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
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
    : RegisterContext(thread, 0) {
  if (!core)
    return;

  auto &gpu_thread = static_cast<ThreadNVGPUCore &>(thread);
  SectionSP lane_sp = gpu_thread.GetLaneSection();
  SectionSP warp_sp = gpu_thread.GetWarpSection();

  Log *log = GetLog(LLDBLog::Process);

  // Read lane / warp / device rows.
  llvm::Expected<nvgpu_core::LaneEntry> lane_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::LaneEntry>(lane_sp, core);
  llvm::Expected<nvgpu_core::WarpEntry> warp_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::WarpEntry>(warp_sp, core);
  llvm::Expected<nvgpu_core::DeviceEntry> dev_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::DeviceEntry>(
          gpu_thread.GetDeviceSection(), core);
  if (!lane_or || !warp_or || !dev_or) {
    if (!lane_or)
      LLDB_LOG(log, "RegisterContextNVGPUCore: lane decode failed: {0}",
               llvm::toString(lane_or.takeError()));
    if (!warp_or)
      LLDB_LOG(log, "RegisterContextNVGPUCore: warp decode failed: {0}",
               llvm::toString(warp_or.takeError()));
    if (!dev_or)
      LLDB_LOG(log, "RegisterContextNVGPUCore: device decode failed: {0}",
               llvm::toString(dev_or.takeError()));
    return;
  }

  m_register_data.PC = lane_or->virtualPC;
  m_register_data.errorPC = warp_or->errorPCValid ? warp_or->errorPC : 0;

  // Fill each per-class slice of the canonical layout from its corresponding
  // section, clamped to the device-advertised register count. The device row
  // is the single authoritative bound; any extra section bytes (alignment
  // slop, format-version-skew tail) are ignored, and any unfilled tail of
  // the slice stays zero from value-init.
  auto copy_class = [core](uint32_t *dst, size_t dst_capacity_words,
                           uint32_t device_count, SectionSP section) {
    if (!section)
      return;
    DataExtractor data;
    core->ReadSectionData(section.get(), data);
    const size_t section_words = data.GetByteSize() / sizeof(uint32_t);
    const size_t n = std::min(
        {static_cast<size_t>(device_count), section_words, dst_capacity_words});
    if (n > 0)
      data.CopyData(0, n * sizeof(uint32_t), dst);
  };

  copy_class(m_register_data.regular, sass::kNumRRegs, dev_or->numRegsPerLane,
             nvgpu_core::FindChildByType(*lane_sp, eSectionTypeNVGPURegisters));
  copy_class(
      m_register_data.predicate, sass::kNumPRegs, dev_or->numPredicatesPrLane,
      nvgpu_core::FindChildByType(*lane_sp, eSectionTypeNVGPUPredicates));
  copy_class(
      m_register_data.uniform, sass::kNumURRegs, dev_or->numUniformRegsPrWarp,
      nvgpu_core::FindChildByType(*warp_sp, eSectionTypeNVGPUUniformRegisters));
  copy_class(m_register_data.uniform_predicate, sass::kNumUPRegs,
             dev_or->numUniformPredicatesPrWarp,
             nvgpu_core::FindChildByType(*warp_sp,
                                         eSectionTypeNVGPUUniformPredicates));
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

  // Every RegisterInfo from `sass::GetRegisterInfos()` has its `byte_offset`
  // computed against `sass::RegisterLayout` -- which is exactly the layout
  // of `m_register_data`. So a register read is a `byte_offset + byte_size`
  // slice of that buffer, with no per-class dispatch.
  const uint32_t offset = reg_info->byte_offset;
  if (offset + reg_info->byte_size > sizeof(m_register_data))
    return false;

  Status error;
  reg_value.SetFromMemoryData(
      *reg_info, reinterpret_cast<const uint8_t *>(&m_register_data) + offset,
      reg_info->byte_size, eByteOrderLittle, error);
  return error.Success();
}

bool RegisterContextNVGPUCore::WriteRegister(const RegisterInfo *reg_info,
                                             const RegisterValue &reg_value) {
  return false;
}

bool RegisterContextNVGPUCore::ReadAllRegisterValues(
    WritableDataBufferSP &data_sp) {
  // The canonical layout IS our internal representation, so the snapshot is
  // a single copy of the whole buffer.
  data_sp = std::make_shared<DataBufferHeap>(&m_register_data,
                                             sizeof(m_register_data));
  return true;
}

bool RegisterContextNVGPUCore::WriteAllRegisterValues(
    const DataBufferSP &data_sp) {
  return false;
}
