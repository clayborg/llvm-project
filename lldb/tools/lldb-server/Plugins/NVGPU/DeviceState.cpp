//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceState.h"
#include "../Utils/Utils.h"
#include "ProcessNVGPU.h"
#include "ThreadNVGPU.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/Utility/StreamString.h"
#include <numeric>

using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::sass;

ThreadState::ThreadState(ProcessNVGPU &gpu, nvgpu::PhysicalCoords physical,
                         WarpState &warp_state)
    : m_physical_coords(physical), m_thread_nvgpu(gpu, this),
      m_warp_state(&warp_state) {}

ThreadState::ThreadState(ThreadState &&other)
    : m_physical_coords(other.GetPhysicalCoords()),
      m_thread_nvgpu(other.m_thread_nvgpu.GetGPU(), this),
      m_warp_state(other.m_warp_state) {
  logAndReportFatalError("ThreadState is not movable. Ensure that this "
                         "constructor is never called by reserving the "
                         "appropriate amount of space in parent container.");
}

nvgpu::ThreadCoords ThreadState::GetThreadCoords() const {
  nvgpu::ThreadCoords coords{m_physical_coords,
                             GetWarpState().GetLogicalCoords(), m_is_active};
  coords.logical.thread_idx = m_thread_idx;
  return coords;
}

void ThreadState::Dump(Stream &s) {
  if (m_thread_idx) {
    s.Indent();
    s.Format("x = {}, y = {}, z = {}\n", m_thread_idx->x, m_thread_idx->y,
             m_thread_idx->z);
  }
  s.Indent();
  s.Format("pc = 0x{:x}", m_pc);
  if (m_is_active)
    s.Format(", is_active = {}", *m_is_active ? "true" : "false");
  s.Format(", stop_reason = {}\n",
           StopReasonToString(m_thread_nvgpu.GetStopReason()));
  if (m_exception) {
    s.Indent();
    s.Format("exception = {}\n", m_exception->exception);
    if (m_exception->errorPC) {
      s.Indent();
      s.Format("errorPC = 0x{:x}\n", *m_exception->errorPC);
    }
  }
}

WarpState::WarpState(ProcessNVGPU &gpu, uint32_t num_threads,
                     uint32_t device_id, uint32_t sm_id, uint32_t warp_id,
                     SMState &sm_state)
    : m_physical_coords{device_id, sm_id, warp_id, std::nullopt},
      m_sm_state(&sm_state) {
  m_threads.reserve(num_threads);
  for (uint32_t lane = 0; lane < num_threads; ++lane)
    m_threads.emplace_back(
        gpu, nvgpu::PhysicalCoords{device_id, sm_id, warp_id, lane}, *this);
}

void WarpState::Dump(Stream &s) {
  s.Indent();
  s.Format("Threads (#{}):\n", m_threads.size());

  s.IndentMore();
  for (uint32_t thread_id = 0; thread_id < m_threads.size(); ++thread_id) {
    if (!m_threads[thread_id].IsValid())
      continue;

    s.Indent();
    s.Format("thread_id = {}\n", thread_id);

    s.IndentMore();
    m_threads[thread_id].Dump(s);
    s.IndentLess();
  }
  s.IndentLess();
}

static bool HasRequiredWarpCoords(const nvgpu::PhysicalCoords &physical) {
  return physical.device && physical.sm && physical.warp;
}

llvm::Expected<uint32_t> WarpState::GetCurrentNumRegularRegisters() {
  if (m_current_num_regular_registers)
    return *m_current_num_regular_registers;

  const nvgpu::PhysicalCoords &physical = GetPhysicalCoords();
  if (!HasRequiredWarpCoords(physical))
    return llvm::createStringError(
        "The physical coordinates of the warp are unknown");

  CUDBGWarpResources resources;
  CUDBGResult res = GetSMState().GetDeviceState().GetAPI()->readWarpResources(
      *physical.device, *physical.sm, *physical.warp, &resources);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("Failed to read the resources of the warp. {0}",
                                cudbgGetErrorString(res));

  m_current_num_regular_registers = resources.numRegisters;
  return resources.numRegisters;
}

static llvm::Error
ReadUniformRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                               const nvgpu::PhysicalCoords &physical,
                               WarpSharedRegisterCache &regs) {
  if (!HasRequiredWarpCoords(physical))
    return llvm::Error::success();
  llvm::Expected<size_t> num_regs = device_info.GetNumUniformRegisters();
  if (!num_regs)
    return num_regs.takeError();
  if (*num_regs == 0)
    return llvm::Error::success();

  if (*num_regs > kNumURRegs)
    return createStringErrorFmt(
        "The device reported {0} uniform registers, but only {1} are supported",
        *num_regs, kNumURRegs);

  CUDBGResult res = api->readUniformRegisterRange(
      *physical.device, *physical.sm, *physical.warp, 0, *num_regs,
      regs.val.uniform);

  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("Failed to read the uniform registers. {0}",
                                cudbgGetErrorString(res));

  for (size_t i = 0; i < *num_regs; ++i)
    regs.is_valid.uniform[i] = true;
  return llvm::Error::success();
}

static llvm::Error
ReadUniformPredicateRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                                        const nvgpu::PhysicalCoords &physical,
                                        WarpSharedRegisterCache &regs) {
  if (!HasRequiredWarpCoords(physical))
    return llvm::Error::success();
  llvm::Expected<size_t> num_regs =
      device_info.GetNumUniformPredicateRegisters();
  if (!num_regs)
    return num_regs.takeError();
  if (*num_regs == 0)
    return llvm::Error::success();

  if (*num_regs > kNumUPRegs)
    return createStringErrorFmt("The device reported {0} uniform predicate "
                                "registers, but only {1} are supported",
                                *num_regs, kNumUPRegs);

  CUDBGResult res =
      api->readUniformPredicates(*physical.device, *physical.sm, *physical.warp,
                                 *num_regs, regs.val.uniform_predicate);

  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to read the uniform predicate registers. {0}",
        cudbgGetErrorString(res));

  for (size_t i = 0; i < *num_regs; ++i) {
    regs.val.uniform_predicate[i] = regs.val.uniform_predicate[i] & 0x1;
    regs.is_valid.uniform_predicate[i] = true;
  }
  return llvm::Error::success();
}

const WarpSharedRegisterCache &WarpState::GetRegisters() {
  if (m_regs_calculated)
    return m_shared_registers;

  // Registers that fail to read stay invalid, instead of keeping the values of
  // an earlier stop.
  m_shared_registers = {};
  m_regs_calculated = true;

  const nvgpu::PhysicalCoords &physical = GetPhysicalCoords();
  if (!HasRequiredWarpCoords(physical))
    return m_shared_registers;

  DeviceState &device_info = GetSMState().GetDeviceState();
  CUDBGAPI api = device_info.GetAPI();
  auto log_error = [&physical](llvm::Error error) {
    LLDB_LOG_ERROR(GetLog(GDBRLog::Plugin), std::move(error),
                   "WarpState::GetRegisters(). Device {1}, SM {2}, warp {3}: "
                   "{0}",
                   *physical.device, *physical.sm, *physical.warp);
  };

  if (llvm::Error err = ReadUniformRegistersFromDevice(
          device_info, api, physical, m_shared_registers))
    log_error(std::move(err));
  if (llvm::Error err = ReadUniformPredicateRegistersFromDevice(
          device_info, api, physical, m_shared_registers))
    log_error(std::move(err));

  return m_shared_registers;
}

SMState::SMState(ProcessNVGPU &gpu, uint32_t num_warps,
                 uint32_t num_threads_per_warp, uint32_t device_id,
                 uint32_t sm_id, DeviceState &device_state)
    : m_is_active(false), m_warps(), m_device_state(&device_state) {
  m_warps.reserve(num_warps);
  for (uint32_t warp_id = 0; warp_id < num_warps; ++warp_id)
    m_warps.emplace_back(gpu, num_threads_per_warp, device_id, sm_id, warp_id,
                         *this);
}

void SMState::SetIsActive(bool is_active) { m_is_active = is_active; }

void SMState::Dump(Stream &s) {
  s.Indent();
  s.Format("Warps (#{}):\n", m_warps.size());

  s.IndentMore();

  for (uint32_t warp_id = 0; warp_id < m_warps.size(); ++warp_id) {
    if (!m_warps[warp_id].IsValid())
      continue;

    s.Indent();
    s.Format("warp_id = {}\n", warp_id);

    s.IndentMore();
    m_warps[warp_id].Dump(s);
    s.IndentLess();
  }
  s.IndentLess();
}

DeviceState::DeviceState(ProcessNVGPU &gpu, uint32_t device_id)
    : m_api(gpu.GetDebuggerAPI()), m_device_id(device_id) {}

llvm::Error DeviceState::Initialize(ProcessNVGPU &gpu) {
  CUDBGResult res =
      m_api->getDeviceInfoSizes(m_device_id, &m_device_info_sizes);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the device info sizes of CUDA device {0}. {1}",
        m_device_id, cudbgGetErrorString(res));
  m_device_info_buffer.resize(m_device_info_sizes.requiredBufferSize);

  res = m_api->getNumSMs(m_device_id, &m_num_sms);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of SMs of CUDA device {0}. {1}", m_device_id,
        cudbgGetErrorString(res));

  res = m_api->getNumWarps(m_device_id, &m_num_warps_per_sm);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of warps per SM of CUDA device {0}. {1}",
        m_device_id, cudbgGetErrorString(res));

  res = m_api->getNumLanes(m_device_id, &m_num_threads_per_warp);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of lanes per warp of CUDA device {0}. {1}",
        m_device_id, cudbgGetErrorString(res));

  m_sms.reserve(m_num_sms);
  for (uint32_t sm_id = 0; sm_id < m_num_sms; ++sm_id)
    m_sms.emplace_back(gpu, m_num_warps_per_sm, m_num_threads_per_warp,
                       m_device_id, sm_id, *this);
  return llvm::Error::success();
}

llvm::Expected<size_t> DeviceState::GetNumPredicateRegisters() {
  if (m_num_predicate_registers)
    return *m_num_predicate_registers;

  uint32_t num_predicate_registers = 0;
  CUDBGResult res =
      m_api->getNumPredicates(m_device_id, &num_predicate_registers);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of predicate registers. {0}",
        cudbgGetErrorString(res));
  m_num_predicate_registers = static_cast<size_t>(num_predicate_registers);
  return num_predicate_registers;
}

llvm::Expected<size_t> DeviceState::GetNumUniformPredicateRegisters() {
  if (m_num_uniform_predicate_registers)
    return *m_num_uniform_predicate_registers;

  uint32_t num_uniform_predicate_registers = 0;
  CUDBGResult res = m_api->getNumUniformPredicates(
      m_device_id, &num_uniform_predicate_registers);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of uniform predicate registers. {0}",
        cudbgGetErrorString(res));
  m_num_uniform_predicate_registers =
      static_cast<size_t>(num_uniform_predicate_registers);
  return num_uniform_predicate_registers;
}

llvm::Expected<size_t> DeviceState::GetNumUniformRegisters() {
  if (m_num_uniform_registers)
    return *m_num_uniform_registers;

  uint32_t num_uniform_registers = 0;
  CUDBGResult res =
      m_api->getNumUniformRegisters(m_device_id, &num_uniform_registers);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of uniform registers. {0}",
        cudbgGetErrorString(res));
  m_num_uniform_registers = static_cast<size_t>(num_uniform_registers);
  return num_uniform_registers;
}

llvm::Expected<size_t> DeviceState::GetMaxNumSupportedRegularRegister() {
  if (m_num_r_registers)
    return *m_num_r_registers;

  uint32_t num_r_registers = 0;
  CUDBGResult res = m_api->getNumRegisters(m_device_id, &num_r_registers);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the number of regular registers. {0}",
        cudbgGetErrorString(res));
  m_num_r_registers = static_cast<size_t>(num_r_registers);
  return num_r_registers;
}

void DeviceState::Dump(Stream &s) {
  s.Format("Device id: {0}\n", m_device_id);
  s.Format("SMs (#{}):\n", m_sms.size());
  s.IndentMore();
  for (uint32_t sm_id = 0; sm_id < m_sms.size(); ++sm_id) {
    if (!m_sms[sm_id].IsActive())
      continue;

    s.Indent();
    s.Format("sm_id = {}\n", sm_id);

    s.IndentMore();
    m_sms[sm_id].Dump(s);
    s.IndentLess();
  }
  s.IndentLess();
}

llvm::Expected<const CUDBGGridInfo &>
DeviceState::GetGridInfo(uint64_t grid_id) {
  auto it = m_grid_info.find(grid_id);
  if (it != m_grid_info.end())
    return it->second;

  CUDBGGridStatus grid_status = CUDBG_GRID_STATUS_INVALID;
  CUDBGResult res = m_api->getGridStatus(m_device_id, grid_id, &grid_status);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("Failed to get the status of grid {0}. {1}",
                                grid_id, cudbgGetErrorString(res));
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    return createStringErrorFmt("Grid {0} is invalid", grid_id);

  CUDBGGridInfo grid_info;
  memset(&grid_info, 0, sizeof(CUDBGGridInfo));

  res = m_api->getGridInfo(m_device_id, grid_id, &grid_info);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("Failed to get the info of grid {0}. {1}",
                                grid_id, cudbgGetErrorString(res));

  return m_grid_info.insert({grid_id, grid_info}).first->second;
}

size_t DeviceState::GetMaxNumSupportedThreads() const {
  return m_num_threads_per_warp * m_num_warps_per_sm * m_num_sms;
}

llvm::Expected<DeviceStateRegistry>
DeviceStateRegistry::Create(ProcessNVGPU &gpu) {
  uint32_t num_devices = 0;
  CUDBGResult res = gpu.GetDebuggerAPI()->getNumDevices(&num_devices);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("Failed to get the number of CUDA devices. {0}",
                                cudbgGetErrorString(res));

  DeviceStateRegistry registry;
  // Reserve up front so that no device moves after it is initialized.
  registry.m_devices.reserve(num_devices);
  for (uint32_t device_id = 0; device_id < num_devices; ++device_id) {
    DeviceState &device = registry.m_devices.emplace_back(gpu, device_id);
    if (llvm::Error err = device.Initialize(gpu))
      return err;
  }
  return registry;
}

llvm::Error DeviceStateRegistry::BatchUpdate(uint32_t stop_id) {
  m_snapshot_stop_id.reset();
  for (DeviceState &device : m_devices)
    if (llvm::Error err = device.BatchUpdate())
      return err;
  m_snapshot_stop_id = stop_id;
  return llvm::Error::success();
}

void DeviceStateRegistry::Dump(Stream &s) {
  for (DeviceState &device : m_devices)
    device.Dump(s);
}

std::string DeviceStateRegistry::Dump() {
  StreamString s;
  Dump(s);
  return s.GetData();
}

size_t DeviceStateRegistry::GetMaxNumSupportedThreads() const {
  return std::accumulate(m_devices.begin(), m_devices.end(), 0,
                         [](size_t acc, const DeviceState &device) {
                           return acc + device.GetMaxNumSupportedThreads();
                         });
}
