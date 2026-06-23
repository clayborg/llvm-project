//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVGPU.h"

#include "../Utils/Utils.h"
#include "DeviceState.h"
#include "ProcessNVGPU.h"
#include "ThreadNVGPU.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "cudadebugger.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

RegisterContextNVGPU::RegisterContextNVGPU(ThreadNVGPU &thread)
    : NativeRegisterContext(thread) {}

void RegisterContextNVGPU::InvalidateAllRegisters() { m_regs.reset(); }

ThreadNVGPU &RegisterContextNVGPU::GetGPUThread() {
  return static_cast<ThreadNVGPU &>(GetThread());
}

CUDBGAPI RegisterContextNVGPU::GetDebuggerAPI() {
  return GetGPUThread().GetGPU().GetDebuggerAPI();
}

static void ReadRegularRegistersFromDevice(CUDBGAPI api, WarpState &warp_state,
                                           const ThreadCoords &thread_coords,
                                           ThreadRegistersWithValidity &regs) {
  uint32_t num_regs_read = warp_state.GetCurrentNumRegularRegisters();
  // Always call the stable 7-arg entry point. It lives at a fixed offset in
  // every in-major driver's API table, so it works regardless of the
  // (possibly older) driver we attached to -- no runtime version check is
  // needed. CTK 13.2 (CUDBG API revision > 167) renamed it to
  // readRegisterRange60 and appended a new 8-arg readRegisterRange variant we
  // don't use; select the right name at compile time.
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(168)
  CUDBGResult res = api->readRegisterRange60(
      thread_coords.dev_id, thread_coords.sm_id, thread_coords.warp_id,
      thread_coords.thread_id, 0, num_regs_read, regs.val.regular);
#else
  CUDBGResult res = api->readRegisterRange(
      thread_coords.dev_id, thread_coords.sm_id, thread_coords.warp_id,
      thread_coords.thread_id, 0, num_regs_read, regs.val.regular);
#endif
  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVGPU::ReadAllRegsFromDevice(). "
                           "readRegisterRange failed: {}",
                           cudbgGetErrorString(res));
  for (size_t i = 0; i < num_regs_read; ++i)
    regs.is_valid.regular[i] = true;
}

static void
ReadPredicateRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                                 const ThreadCoords &thread_coords,
                                 ThreadRegistersWithValidity &regs) {
  size_t num_regs = device_info.GetNumPredicateRegisters();
  if (num_regs == 0)
    return;

  CUDBGResult res = api->readPredicates(
      thread_coords.dev_id, thread_coords.sm_id, thread_coords.warp_id,
      thread_coords.thread_id, num_regs, regs.val.predicate);

  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVGPU::ReadAllRegsFromDevice(). "
                           "readPredicates failed: {}",
                           cudbgGetErrorString(res));

  for (size_t i = 0; i < num_regs; ++i)
    regs.is_valid.predicate[i] = true;
}

const ThreadRegistersWithValidity &
RegisterContextNVGPU::ReadAllRegsFromDevice() {
  if (m_regs)
    return *m_regs;

  m_regs.emplace();
  ThreadRegistersWithValidity &regs = *m_regs;
  ThreadNVGPU &thread = GetGPUThread();
  const ThreadState *thread_state = thread.GetThreadState();

  if (!thread_state) {
    // We need to send always a PC to the client upon stop events, otherwise the
    // client will be in a borked state.
    regs.val.PC = 0;
    regs.is_valid.PC = true;
    return regs;
  }

  CUDBGAPI api = GetDebuggerAPI();
  const ThreadCoords &thread_coords = thread_state->GetCoords();

  {
    regs.val.PC = thread_state->GetPC();
    regs.is_valid.PC = true;
  }

  {
    if (const ExceptionInfo *exception = thread_state->GetException();
        exception && exception->errorPC.has_value()) {
      regs.val.errorPC = *exception->errorPC;
      regs.is_valid.errorPC = true;
    }
  }

  WarpState &warp_state = thread_state->GetWarpState();
  DeviceState &device_info =
      thread.GetGPU().GetAllDevices()[thread_coords.dev_id];

  ReadRegularRegistersFromDevice(api, warp_state, thread_coords, regs);
  ReadPredicateRegistersFromDevice(device_info, api, thread_coords, regs);

  const WarpRegistersWithValidity &warp_regs = warp_state.GetRegisters();

  std::copy(warp_regs.val.uniform, warp_regs.val.uniform + kNumURRegs,
            regs.val.uniform);
  std::copy(warp_regs.val.uniform_predicate,
            warp_regs.val.uniform_predicate + kNumUPRegs,
            regs.val.uniform_predicate);
  std::copy(warp_regs.is_valid.uniform, warp_regs.is_valid.uniform + kNumURRegs,
            regs.is_valid.uniform);
  std::copy(warp_regs.is_valid.uniform_predicate,
            warp_regs.is_valid.uniform_predicate + kNumUPRegs,
            regs.is_valid.uniform_predicate);

  {
    regs.val.regular_zero = 0;
    regs.is_valid.regular_zero = true;
  }

  {
    regs.val.uniform_zero = 0;
    regs.is_valid.uniform_zero = true;
  }

  return regs;
}

uint32_t RegisterContextNVGPU::GetRegisterSetCount() const {
  return sass::GetRegisterSets().size();
}

uint32_t RegisterContextNVGPU::GetRegisterCount() const {
  return sass::GetRegisterInfos().size();
}

uint32_t RegisterContextNVGPU::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextNVGPU::GetRegisterInfoAtIndex(uint32_t reg) const {
  llvm::ArrayRef<lldb_private::RegisterInfo> infos = sass::GetRegisterInfos();
  if (reg < infos.size())
    return &infos[reg];
  return nullptr;
}

const RegisterSet *
RegisterContextNVGPU::GetRegisterSet(uint32_t set_index) const {
  llvm::ArrayRef<lldb_private::RegisterSet> sets = sass::GetRegisterSets();
  if (set_index < sets.size())
    return &sets[set_index];
  return nullptr;
}

Status RegisterContextNVGPU::ReadRegister(const RegisterInfo *reg_info,
                                          RegisterValue &reg_value) {
  const ThreadRegistersWithValidity &regs = ReadAllRegsFromDevice();
  int reg_num = reg_info->kinds[eRegisterKindLLDB];

  if (reg_num == sass::LLDB_SP)
    reg_num = sass::LLDB_R0 + sass::SASS_SP_REG;
  if (reg_num == sass::LLDB_FP)
    reg_num = sass::LLDB_R0 + sass::SASS_FP_REG;

  if (reg_num == sass::LLDB_PC) {
    if (!regs.is_valid.PC)
      return Status("PC register is invalid");
  } else if (reg_num == sass::LLDB_ERROR_PC) {
    if (!regs.is_valid.errorPC)
      return Status("errorPC register is invalid");
  } else if (reg_num == sass::LLDB_RA) {
    if (!regs.is_valid.regular[sass::SASS_RA_REG_LO] ||
        !regs.is_valid.regular[sass::SASS_RA_REG_HI])
      return Status("RA register is invalid");
  } else if (reg_num >= static_cast<int>(sass::kNumSASSRegs)) {
    return Status::FromErrorStringWithFormatv("unknown register #{}", reg_num);
  } else if (int up_index = reg_num - sass::LLDB_UP0;
             up_index >= 0 && up_index < static_cast<int>(sass::kNumUPRegs)) {
    if (!regs.is_valid.uniform_predicate[up_index])
      return Status::FromErrorStringWithFormatv("UP{} register is invalid",
                                                up_index);
  } else if (int p_index = reg_num - sass::LLDB_P0;
             p_index >= 0 && p_index < static_cast<int>(sass::kNumPRegs)) {
    if (!regs.is_valid.predicate[p_index])
      return Status::FromErrorStringWithFormatv("P{} register is invalid",
                                                p_index);
  } else if (int ur_index = reg_num - sass::LLDB_UR0;
             ur_index >= 0 && ur_index < static_cast<int>(sass::kNumURRegs)) {
    if (!regs.is_valid.uniform[ur_index])
      return Status::FromErrorStringWithFormatv("UR{} register is invalid",
                                                ur_index);
  } else if (int r_index = reg_num - sass::LLDB_R0;
             r_index >= 0 && r_index < static_cast<int>(sass::kNumRRegs)) {
    if (!regs.is_valid.regular[r_index])
      return Status::FromErrorStringWithFormatv("R{} register is invalid",
                                                r_index);
  }

  Status error;
  reg_value.SetFromMemoryData(
      *reg_info, (const uint8_t *)&regs.val + reg_info->byte_offset,
      reg_info->byte_size, lldb::eByteOrderLittle, error);
  return error;
}

Status RegisterContextNVGPU::WriteRegister(const RegisterInfo *reg_info,
                                           const RegisterValue &reg_value) {
  return Status("WriteRegister unimplemented");
}

Status RegisterContextNVGPU::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return Status("ReadAllRegisterValues unimplemented");
}

Status RegisterContextNVGPU::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return Status("WriteAllRegisterValues unimplemented");
}

std::vector<uint32_t>
RegisterContextNVGPU::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    g_expedited_regs.push_back(sass::LLDB_PC);
    g_expedited_regs.push_back(sass::LLDB_ERROR_PC);
    g_expedited_regs.push_back(sass::LLDB_SP);
    g_expedited_regs.push_back(sass::LLDB_FP);
    g_expedited_regs.push_back(sass::LLDB_RA);
    for (uint32_t i = 0; i < sass::kNumRRegs; ++i)
      g_expedited_regs.push_back(sass::LLDB_R0 + i);
  }
  return g_expedited_regs;
}

std::optional<uint64_t> RegisterContextNVGPU::ReadErrorPC() {
  const ThreadRegistersWithValidity &regs = ReadAllRegsFromDevice();
  if (regs.is_valid.errorPC)
    return regs.val.errorPC;
  return std::nullopt;
}

ThreadRegisterValidity::ThreadRegisterValidity() {
  PC = false;
  errorPC = false;
  for (size_t i = 0; i < kNumRRegs; ++i)
    regular[i] = false;
  regular_zero = false;
  for (size_t i = 0; i < kNumPRegs; ++i)
    predicate[i] = false;
  for (size_t i = 0; i < kNumURRegs; ++i)
    uniform[i] = false;
  uniform_zero = false;
  for (size_t i = 0; i < kNumUPRegs; ++i)
    uniform_predicate[i] = false;
}

WarpRegisterValidity::WarpRegisterValidity() {
  for (size_t i = 0; i < kNumURRegs; ++i)
    uniform[i] = false;
  for (size_t i = 0; i < kNumUPRegs; ++i)
    uniform_predicate[i] = false;
}
