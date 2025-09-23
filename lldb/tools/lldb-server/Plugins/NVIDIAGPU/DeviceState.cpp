//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceState.h"
#include "../Utils/Utils.h"
#include "NVIDIAGPU.h"
#include "lldb/Utility/StreamString.h"
#include <numeric>

using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

std::string PhysicalCoords::Dump() const {
  return llvm::formatv("dev_id = {} sm_id = {} warp_id = {} thread_id = {}",
                       dev_id, sm_id, warp_id, thread_id);
}

llvm::StringRef static ExceptionToString(CUDBGException_t exception) {
  switch (exception) {
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
    return "Warp - Illegal instruction";
  case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
    return "Warp - Out of range address";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
    return "Warp - Misaligned address";
  case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
    return "Warp - Invalid address space";
  case CUDBG_EXCEPTION_WARP_INVALID_PC:
    return "Warp - invalid PC";
  case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
    return "Warp - Hardware stack overflow";
  case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
    return "Device - Illegal address";
  case CUDBG_EXCEPTION_WARP_ASSERT:
    return "Warp - Assert";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS:
    return "Warp - Illegal address";
  case CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT:
    return "Cluster - Block not present";
  case CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS:
    return "Cluster - Out of range address";
  case CUDBG_EXCEPTION_WARP_STACK_CANARY:
    return "Warp - Stack canary";
#if CUDBG_API_VERSION_MAJOR >= 13
  case CUDBG_EXCEPTION_WARP_TMEM_ACCESS_CHECK:
    return "Warp - TMEM access check";
  case CUDBG_EXCEPTION_WARP_TMEM_LEAK:
    return "Warp - TMEM leak";
  case CUDBG_EXCEPTION_WARP_CALL_REQUIRES_NEWER_DRIVER:
    return "Warp - Call requires newer driver";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_PC:
    return "Warp - Misaligned PC";
  case CUDBG_EXCEPTION_WARP_PC_OVERFLOW:
    return "Warp - PC overflow";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_REG:
    return "Warp - Misaligned register";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_ENCODING:
    return "Warp - Illegal instruction encoding";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_PARAM:
    return "Warp - Illegal instruction parameter";
  case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_REGISTER:
    return "Warp - Out of range register";
  case CUDBG_EXCEPTION_WARP_INVALID_CONST_ADDR_LDC:
    return "Warp - Invalid constant address LDC";
  case CUDBG_EXCEPTION_WARP_MMU_FAULT:
    return "Warp - MMU fault";
  case CUDBG_EXCEPTION_WARP_ARRIVE:
    return "Warp - Arrive";
  case CUDBG_EXCEPTION_CLUSTER_POISON:
    return "Cluster - Poison";
  case CUDBG_EXCEPTION_WARP_API_STACK_ERROR:
    return "Warp - API stack error";
  case CUDBG_EXCEPTION_WARP_BLOCK_NOT_PRESENT:
    return "Warp - Block not present";
  case CUDBG_EXCEPTION_WARP_USER_STACK_OVERFLOW:
    return "Warp - User stack overflow";
#endif
  default:
    return "Device Unknown Exception";
  }
}

ExceptionInfo::ExceptionInfo(CUDBGException_t exception,
                             std::optional<uint64_t> errorPC)
    : exception(exception), errorPC(errorPC) {
  if (exception == CUDBG_EXCEPTION_NONE)
    logAndReportFatalError(
        "ExceptionInfo: exception shouldn't be CUDBG_EXCEPTION_NONE");
}

std::string ExceptionInfo::ToString() const {
  std::string result = ExceptionToString(exception).str();
  if (std::optional<uint64_t> error_pc = errorPC)
    result += llvm::formatv(" at {0:x}", *error_pc);

  return result;
}

static lldb::tid_t g_thread_id = 1;

ThreadState::ThreadState(NVIDIAGPU &gpu, const PhysicalCoords &physical_coords)
    : m_physical_coords(physical_coords), m_thread_id(g_thread_id++),
      m_thread_nvidiagpu(gpu, m_thread_id, this) {}

ThreadState::ThreadState(ThreadState &&other)
    : m_physical_coords(other.GetPhysicalCoords()),
      m_thread_id(other.m_thread_id),
      m_thread_nvidiagpu(other.m_thread_nvidiagpu.GetGPU(), other.m_thread_id,
                         this) {
  logAndReportFatalError("ThreadState is not movable. Ensure that this "
                         "constructor is never called by reserving the "
                         "appropriate amount of space in parent container.");
}

void ThreadState::Dump(Stream &s) {
  s.Indent();
  s.Format("x = {}, y = {}, z = {}\n", m_thread_idx.x, m_thread_idx.y,
           m_thread_idx.z);
  s.Indent();
  s.Format("pc = 0x{:x}\n", m_pc);
  if (m_exception) {
    s.Indent();
    s.Format("exception = {}\n", m_exception->exception);
    if (m_exception->errorPC) {
      s.Indent();
      s.Format("errorPC = 0x{:x}\n", *m_exception->errorPC);
    }
  }
}

WarpState::WarpState(NVIDIAGPU &gpu, uint32_t num_threads, uint32_t device_id,
                     uint32_t sm_id, uint32_t warp_id) {
  m_threads.reserve(num_threads);
  for (uint32_t thread_id = 0; thread_id < num_threads; ++thread_id)
    m_threads.emplace_back(
        gpu, PhysicalCoords(device_id, sm_id, warp_id, thread_id));
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

const ThreadState *WarpState::FindSomeThreadWithException() const {
  for (const ThreadState &thread : m_threads)
    if (thread.IsValid() && thread.HasException())
      return &thread;

  return nullptr;
}

size_t WarpState::GetMaxNumSupportedThreads() const { return m_threads.size(); }

size_t WarpState::GetCurrentNumRegularRegisters() {
  if (m_current_num_regular_registers)
    return *m_current_num_regular_registers;

  CUDBGWarpResources resources;
  // We get the coordinates and a handle to the API from the first thread in
  // the warp. We do this to avoid storing additional copies at the warp level.
  const PhysicalCoords &physical_coords = m_threads[0].GetPhysicalCoords();
  NVIDIAGPU &gpu = m_threads[0].GetThreadNVIDIAGPU().GetGPU();

  CUDBGResult res = gpu.GetDebuggerAPI()->readWarpResources(
      physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
      &resources);
  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("WarpState::GetCurrentNumRegularRegisters(). "
                           "readWarpResources failed: {}",
                           cudbgGetErrorString(res));

  m_current_num_regular_registers = resources.numRegisters;
  return *m_current_num_regular_registers;
}

SMState::SMState(NVIDIAGPU &gpu, uint32_t num_warps,
                 uint32_t num_threads_per_warp, uint32_t device_id,
                 uint32_t sm_id)
    : m_is_active(false), m_warps() {
  m_warps.reserve(num_warps);
  for (uint32_t warp_id = 0; warp_id < num_warps; ++warp_id)
    m_warps.emplace_back(gpu, num_threads_per_warp, device_id, sm_id, warp_id);
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

const ThreadState *SMState::FindSomeThreadWithException() const {
  for (const WarpState &warp : m_warps)
    if (warp.IsValid() && warp.HasException())
      return warp.FindSomeThreadWithException();

  return nullptr;
}

size_t SMState::GetMaxNumSupportedThreads() const {
  return std::accumulate(m_warps.begin(), m_warps.end(), 0,
                         [](size_t acc, const WarpState &warp) {
                           return acc + warp.GetMaxNumSupportedThreads();
                         });
}

DeviceState::DeviceState(NVIDIAGPU &gpu, uint32_t device_id)
    : m_api(gpu.GetDebuggerAPI()), m_device_id(device_id) {
  CUDBGResult res =
      m_api->getDeviceInfoSizes(m_device_id, &m_device_info_sizes);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::DeviceInformation(). "
                           "getDeviceInfoSizes failed: {0}",
                           cudbgGetErrorString(res));
  }
  m_device_info_buffer.resize(m_device_info_sizes.requiredBufferSize);

  res = m_api->getNumSMs(m_device_id, &m_num_sms);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::DeviceInformation(). "
                           "getNumSMs failed: {0}",
                           cudbgGetErrorString(res));
  }

  res = m_api->getNumWarps(m_device_id, &m_num_warps_per_sm);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::DeviceInformation(). "
                           "getNumWarps failed: {0}",
                           cudbgGetErrorString(res));
  }

  res = m_api->getNumLanes(m_device_id, &m_num_threads_per_warp);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::DeviceInformation(). "
                           "getNumLanes failed: {0}",
                           cudbgGetErrorString(res));
  }

  m_sms.reserve(m_num_sms);
  for (uint32_t sm_id = 0; sm_id < m_num_sms; ++sm_id)
    m_sms.emplace_back(gpu, m_num_warps_per_sm, m_num_threads_per_warp,
                       m_device_id, sm_id);
}

size_t DeviceState::GetNumPredicateRegisters() {
  if (m_num_predicate_registers)
    return *m_num_predicate_registers;

  uint32_t num_predicate_registers = 0;
  CUDBGResult res =
      m_api->getNumPredicates(m_device_id, &num_predicate_registers);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::GetNumPredicateRegisters(). "
                           "getNumPredicates failed: {}",
                           cudbgGetErrorString(res));
  }
  m_num_predicate_registers = static_cast<size_t>(num_predicate_registers);
  return *m_num_predicate_registers;
}

size_t DeviceState::GetNumUniformPredicateRegisters() {
  if (m_num_uniform_predicate_registers)
    return *m_num_uniform_predicate_registers;

  uint32_t num_uniform_predicate_registers = 0;
  CUDBGResult res = m_api->getNumUniformPredicates(
      m_device_id, &num_uniform_predicate_registers);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "DeviceInformation::GetNumUniformPredicateRegisters(). "
        "getNumUniformPredicates failed: {}",
        cudbgGetErrorString(res));
  }
  m_num_uniform_predicate_registers =
      static_cast<size_t>(num_uniform_predicate_registers);
  return *m_num_uniform_predicate_registers;
}

size_t DeviceState::GetNumUniformRegisters() {
  if (m_num_uniform_registers)
    return *m_num_uniform_registers;

  uint32_t num_uniform_registers = 0;
  CUDBGResult res =
      m_api->getNumUniformRegisters(m_device_id, &num_uniform_registers);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::GetNumUniformRegisters(). "
                           "getNumUniformRegisters failed: {}",
                           cudbgGetErrorString(res));
  }
  m_num_uniform_registers = static_cast<size_t>(num_uniform_registers);
  return *m_num_uniform_registers;
}

size_t DeviceState::GetMaxNumSupportedRegularRegister() {
  if (m_num_r_registers)
    return *m_num_r_registers;

  uint32_t num_r_registers = 0;
  CUDBGResult res = m_api->getNumRegisters(m_device_id, &num_r_registers);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::GetNumRRegisters(). "
                           "getNumRegisters failed: {}",
                           cudbgGetErrorString(res));
  }
  m_num_r_registers = static_cast<size_t>(num_r_registers);
  return *m_num_r_registers;
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

const CUDBGGridInfo &DeviceState::GetGridInfo(uint64_t grid_id) {
  auto it = m_grid_info.find(grid_id);
  if (it != m_grid_info.end())
    return it->second;

  CUDBGGridStatus grid_status = CUDBG_GRID_STATUS_INVALID;
  m_api->getGridStatus(m_device_id, grid_id, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    logAndReportFatalError("DeviceInformation::GetGridInfo(). "
                           "getGridStatus returned invalid grid status: {}",
                           grid_status);

  CUDBGGridInfo grid_info;
  memset(&grid_info, 0, sizeof(CUDBGGridInfo));

  CUDBGResult res = m_api->getGridInfo(m_device_id, grid_id, &grid_info);
  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("DeviceInformation::GetGridInfo(). "
                           "getGridInfo failed: {}",
                           cudbgGetErrorString(res));

  return m_grid_info.insert({grid_id, grid_info}).first->second;
}

const ThreadState *DeviceState::FindSomeThreadWithException() const {
  for (uint32_t sm_id = 0; sm_id < m_sms.size(); ++sm_id)
    if (m_sm_active_mask.Get(sm_id) && m_sm_exception_mask.Get(sm_id))
      return m_sms[sm_id].FindSomeThreadWithException();

  return nullptr;
}

size_t DeviceState::GetMaxNumSupportedThreads() const {
  return std::accumulate(m_sms.begin(), m_sms.end(), 0,
                         [](size_t acc, const SMState &sm) {
                           return acc + sm.GetMaxNumSupportedThreads();
                         });
}

DeviceStateRegistry::DeviceStateRegistry(NVIDIAGPU &gpu) {
  uint32_t num_devices;
  CUDBGResult res = gpu.GetDebuggerAPI()->getNumDevices(&num_devices);
  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("AllDevices::AllDevices(). "
                           "getNumDevices failed: {0}",
                           cudbgGetErrorString(res));

  m_devices.reserve(num_devices);
  for (uint32_t device_id = 0; device_id < num_devices; ++device_id)
    m_devices.emplace_back(gpu, device_id);
}

void DeviceStateRegistry::BatchUpdate() {
  for (DeviceState &device : m_devices)
    device.BatchUpdate();
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

const ThreadState *DeviceStateRegistry::FindSomeThreadWithException() const {
  for (const DeviceState &device : m_devices)
    if (const ThreadState *thread = device.FindSomeThreadWithException())
      return thread;

  return nullptr;
}

size_t DeviceStateRegistry::GetMaxNumSupportedThreads() const {
  return std::accumulate(m_devices.begin(), m_devices.end(), 0,
                         [](size_t acc, const DeviceState &device) {
                           return acc + device.GetMaxNumSupportedThreads();
                         });
}
