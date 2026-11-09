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
#include "lldb/Utility/NVGPU/CUDADebuggerAPIVersion.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/NVGPU/SASSRegisterNumbers.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/MathExtras.h"

#include "cudadebugger.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

using namespace sass::regnum;

RegisterContextNVGPU::RegisterContextNVGPU(ThreadNVGPU &thread)
    : NativeRegisterContext(thread) {}

void RegisterContextNVGPU::InvalidateAllRegisters() {
  m_regular_registers.reset();
  m_predicate_registers.reset();
}

ThreadNVGPU &RegisterContextNVGPU::GetGPUThread() {
  return static_cast<ThreadNVGPU &>(GetThread());
}

CUDBGAPI RegisterContextNVGPU::GetDebuggerAPI() {
  return GetGPUThread().GetGPU().GetDebuggerAPI();
}

static llvm::Error
ReadRegularRegistersFromDevice(CUDBGAPI api, WarpState &warp_state,
                               const nvgpu::PhysicalCoords &coords,
                               llvm::SmallVectorImpl<uint32_t> &registers) {
  if (!coords.device || !coords.sm || !coords.warp || !coords.lane)
    return llvm::Error::success();

  llvm::Expected<uint32_t> num_regs_read =
      warp_state.GetCurrentNumRegularRegisters();
  if (!num_regs_read)
    return num_regs_read.takeError();

  if (*num_regs_read > sass::kNumRRegs)
    return createStringErrorFmt(
        "The device reported {0} regular registers, but only {1} are supported",
        *num_regs_read, sass::kNumRRegs);
  // Always call the stable 7-arg entry point. It lives at a fixed offset in
  // every in-major driver's API table, so it works regardless of the
  // (possibly older) driver we attached to -- no runtime version check is
  // needed. CTK 13.2 (CUDBG API revision 168) renamed it to
  // readRegisterRange60 and appended a new 8-arg readRegisterRange we do not
  // use.
  registers.resize(*num_regs_read);
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(168)
  auto read_register_range = api->readRegisterRange60;
#else
  auto read_register_range = api->readRegisterRange;
#endif
  CUDBGResult res =
      read_register_range(*coords.device, *coords.sm, *coords.warp,
                          *coords.lane, 0, *num_regs_read, registers.data());
  if (res != CUDBG_SUCCESS) {
    registers.clear();
    return createStringErrorFmt("Failed to read the regular registers. {0}",
                                cudbgGetErrorString(res));
  }
  return llvm::Error::success();
}

static llvm::Error
ReadPredicateRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                                 const nvgpu::PhysicalCoords &coords,
                                 llvm::SmallVectorImpl<uint32_t> &registers) {
  if (!coords.device || !coords.sm || !coords.warp || !coords.lane)
    return llvm::Error::success();

  llvm::Expected<size_t> num_regs = device_info.GetNumPredicateRegisters();
  if (!num_regs)
    return num_regs.takeError();
  if (*num_regs == 0)
    return llvm::Error::success();

  if (*num_regs > sass::kNumPRegs)
    return createStringErrorFmt("The device reported {0} predicate registers, "
                                "but only {1} are supported",
                                *num_regs, sass::kNumPRegs);

  registers.resize(*num_regs);
  CUDBGResult res =
      api->readPredicates(*coords.device, *coords.sm, *coords.warp,
                          *coords.lane, *num_regs, registers.data());

  if (res != CUDBG_SUCCESS) {
    registers.clear();
    return createStringErrorFmt("Failed to read the predicate registers. {0}",
                                cudbgGetErrorString(res));
  }

  return llvm::Error::success();
}

static llvm::Error InvalidRegister(const RegisterInfo &reg_info) {
  return createStringErrorFmt("{0} register is invalid", reg_info.name);
}

/// Read little-endian register bytes.
static llvm::Error SetValue(const RegisterInfo &reg_info, const void *data,
                            RegisterValue &value) {
  Status error;
  value.SetFromMemoryData(reg_info, data, reg_info.byte_size,
                          lldb::eByteOrderLittle, error);
  return error.ToError();
}

static llvm::Error SetDim3Value(const RegisterInfo &reg_info,
                                const nvgpu::Dim3u &dim, RegisterValue &value) {
  const uint32_t components[sass::kNumXYZComponents] = {dim.x, dim.y, dim.z};
  return SetValue(reg_info, components, value);
}

/// Read from 32-bit slots: predicates require one slot, RA requires two.
static llvm::Error SetValueAt(const RegisterInfo &reg_info,
                              llvm::ArrayRef<uint32_t> registers,
                              uint32_t index, RegisterValue &value) {
  const size_t count = llvm::divideCeil(reg_info.byte_size, sizeof(uint32_t));
  if (index + count > registers.size())
    return InvalidRegister(reg_info);
  return SetValue(reg_info, &registers[index], value);
}

llvm::ArrayRef<uint32_t>
RegisterContextNVGPU::GetRegularRegisters(const ThreadState &thread_state) {
  if (m_regular_registers)
    return *m_regular_registers;
  llvm::SmallVectorImpl<uint32_t> &registers = m_regular_registers.emplace();
  if (llvm::Error err = ReadRegularRegistersFromDevice(
          GetDebuggerAPI(), thread_state.GetWarpState(),
          thread_state.GetPhysicalCoords(), registers))
    LLDB_LOG_ERROR(GetLog(GDBRLog::Plugin), std::move(err),
                   "RegisterContextNVGPU::GetRegularRegisters(). Thread {1}: "
                   "{0}",
                   GetGPUThread().GetID());
  return registers;
}

llvm::ArrayRef<uint32_t>
RegisterContextNVGPU::GetPredicateRegisters(const ThreadState &thread_state) {
  if (m_predicate_registers)
    return *m_predicate_registers;
  llvm::SmallVectorImpl<uint32_t> &registers = m_predicate_registers.emplace();
  DeviceState &device_state =
      thread_state.GetWarpState().GetSMState().GetDeviceState();
  if (llvm::Error err = ReadPredicateRegistersFromDevice(
          device_state, GetDebuggerAPI(), thread_state.GetPhysicalCoords(),
          registers))
    LLDB_LOG_ERROR(GetLog(GDBRLog::Plugin), std::move(err),
                   "RegisterContextNVGPU::GetPredicateRegisters(). Thread {1}: "
                   "{0}",
                   GetGPUThread().GetID());
  return registers;
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
  return Status::FromError(ReadRegisterValue(*reg_info, reg_value));
}

llvm::Error
RegisterContextNVGPU::ReadRegisterValue(const RegisterInfo &reg_info,
                                        RegisterValue &value) {
  const uint32_t reg_num = reg_info.kinds[eRegisterKindLLDB];
  if (reg_num >= LLDB_REG_COUNT)
    return createStringErrorFmt("unknown register #{0}", reg_num);

  if (reg_num == LLDB_RZ || reg_num == LLDB_URZ) {
    const uint32_t zero = 0;
    return SetValue(reg_info, &zero, value);
  }

  const ThreadState *thread_state = GetGPUThread().GetThreadState();
  if (reg_num == LLDB_PC) {
    // Every stop needs a PC, even without a current snapshot.
    const uint64_t pc = thread_state ? thread_state->GetPC() : 0;
    return SetValue(reg_info, &pc, value);
  }
  if (!thread_state)
    return InvalidRegister(reg_info);

  // Module-load stops leave the GPU running and the snapshot stale.
  if (!GetGPUThread().GetGPU().IsDeviceStateCurrent())
    return createStringErrorFmt(
        "{0} register is unavailable while the GPU is running", reg_info.name);

  // Read snapshot and cached grid values without fetching registers.
  WarpState &warp_state = thread_state->GetWarpState();
  DeviceState &device_state = warp_state.GetSMState().GetDeviceState();
  const nvgpu::LogicalCoords &logical = warp_state.GetLogicalCoords();
  switch (reg_num) {
  case LLDB_ERROR_PC: {
    const ExceptionInfo *exception = thread_state->GetException();
    if (!exception || !exception->errorPC)
      return InvalidRegister(reg_info);
    const uint64_t error_pc = *exception->errorPC;
    return SetValue(reg_info, &error_pc, value);
  }
  case LLDB_VREG_THREAD_IDX: {
    const std::optional<ThreadIdx> &thread_idx = thread_state->GetThreadIdx();
    if (!thread_idx)
      return InvalidRegister(reg_info);
    return SetDim3Value(reg_info, *thread_idx, value);
  }
  case LLDB_VREG_BLOCK_IDX:
    if (!logical.block_idx)
      return InvalidRegister(reg_info);
    return SetDim3Value(reg_info, *logical.block_idx, value);
  case LLDB_VREG_BLOCK_DIM:
  case LLDB_VREG_GRID_DIM: {
    if (!logical.grid_id)
      return InvalidRegister(reg_info);
    llvm::Expected<const CUDBGGridInfo &> grid_info =
        device_state.GetGridInfo(*logical.grid_id);
    if (!grid_info)
      return grid_info.takeError();
    const CuDim3 &dim = reg_num == LLDB_VREG_BLOCK_DIM ? grid_info->blockDim
                                                       : grid_info->gridDim;
    return SetDim3Value(reg_info, nvgpu::ToDim3u(dim), value);
  }
  case LLDB_VREG_WARP_SIZE: {
    const int32_t warp_size = device_state.GetNumThreadsPerWarp();
    return SetValue(reg_info, &warp_size, value);
  }
  default:
    break;
  }

  if (IsUniformLLDB(reg_num))
    return SetValueAt(reg_info, warp_state.GetUniformRegisters(),
                      reg_num - LLDB_UR_BASE, value);
  if (IsUniformPredicateLLDB(reg_num))
    return SetValueAt(reg_info, warp_state.GetUniformPredicateRegisters(),
                      reg_num - LLDB_UP_BASE, value);

  if (IsPredicateLLDB(reg_num))
    return SetValueAt(reg_info, GetPredicateRegisters(*thread_state),
                      reg_num - LLDB_P_BASE, value);

  uint32_t index;
  switch (reg_num) {
  case LLDB_SP:
    index = SASS_SP;
    break;
  case LLDB_FP:
    index = SASS_FP;
    break;
  case LLDB_RA:
    index = SASS_RA_LO;
    break;
  default:
    assert(IsRegularLLDB(reg_num) && "every other register is handled above");
    index = reg_num - LLDB_R_BASE;
    break;
  }
  return SetValueAt(reg_info, GetRegularRegisters(*thread_state), index, value);
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
  // Keep per-thread jThreadsInfo updates free of device register reads.
  // The reporting thread's full stop reply includes regular registers
  // for unwinding.
  if (expType == ExpeditedRegs::Minimal)
    return {LLDB_PC, LLDB_ERROR_PC, LLDB_VREG_THREAD_IDX, LLDB_VREG_BLOCK_IDX};

  static const std::vector<uint32_t> g_full_regs = [] {
    std::vector<uint32_t> regs = {LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP,
                                  LLDB_RA};
    for (uint32_t i = 0; i < sass::kNumRRegs; ++i)
      regs.push_back(GetRegularLLDB(i));
    return regs;
  }();
  return g_full_regs;
}
