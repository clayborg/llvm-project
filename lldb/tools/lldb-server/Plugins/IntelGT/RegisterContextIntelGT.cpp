//===-- RegisterContextIntelGT.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextIntelGT.h"
#include "EUThreadIntelGT.h"
#include "LevelZeroHelpers.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessIntelGT.h"
#include "ThreadIntelGT.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-defines.h"

#include <cinttypes>
#include <cstring>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

// ---------------------------------------------------------------------------
// ZET_DEBUG_REGSET_TYPE_* values used when interpreting register sets.
// ---------------------------------------------------------------------------

static constexpr uint32_t kRegsetFlagWriteable = 2;
static constexpr uint32_t kCR0_DWORD2_OFFSET = 8; // bytes 8-11

RegisterContextIntelGT::RegisterContextIntelGT(
    NativeThreadProtocol &native_thread, const ProcessIntelGT *process,
    std::shared_ptr<EUThreadIntelGT> eu_thread, uint32_t device_index,
    uint32_t lane_id, uint32_t simd_width, bool is_shadow_thread)
    : NativeRegisterContext(native_thread), m_process(process),
      m_eu_thread(std::move(eu_thread)), m_device_index(device_index),
      m_lane_id(lane_id), m_simd_width(simd_width > 0 ? simd_width : 1),
      m_is_shadow_thread(is_shadow_thread) {}

// ---------------------------------------------------------------------------
// EnsureLazyDiscovery
//
// Trigger device-level register discovery on first metadata access so the
// target XML is populated before LLDB requests it.
// ---------------------------------------------------------------------------

void RegisterContextIntelGT::EnsureLazyDiscovery() const {
  if (!m_process)
    return;
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (session && session->registers_discovered)
    return;
  const_cast<ProcessIntelGT *>(m_process)->EnsureDeviceRegistersDiscovered(
      m_device_index);
}

// ---------------------------------------------------------------------------
// ReadRegisterSet
// ---------------------------------------------------------------------------

Status RegisterContextIntelGT::ReadRegisterSet(uint32_t regset_index) {
  if (m_is_shadow_thread)
    return Status::FromErrorString("No registers for shadow thread");

  if (!m_eu_thread)
    return Status::FromErrorString("No EU thread for register access");

  if (!m_process)
    return Status::FromErrorString("No process for register metadata");

  EnsureLazyDiscovery();

  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return Status::FromErrorString("No device session");

  if (regset_index >= session->regset_props.size())
    return Status::FromErrorString("Register set index out of range");

  // Allocate the EU thread's shared register cache if needed; device-level
  // discovery populates DeviceSession but not the per-EU-thread cache.
  if (!m_eu_thread->IsRegsetCacheAllocated())
    m_eu_thread->AllocateRegsetCache(session->regset_props);

  if (m_eu_thread->IsRegsetValid(regset_index))
    return Status();

  return m_eu_thread->ReadRegisterSet(regset_index,
                                      session->regset_props[regset_index]);
}

// ---------------------------------------------------------------------------
// NativeRegisterContext interface
// ---------------------------------------------------------------------------

uint32_t RegisterContextIntelGT::GetRegisterCount() const {
  if (!m_process)
    return 0;
  EnsureLazyDiscovery();
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return 0;
  uint32_t count = static_cast<uint32_t>(session->reg_infos.size());
  return count;
}

uint32_t RegisterContextIntelGT::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextIntelGT::GetRegisterInfoAtIndex(uint32_t reg) const {
  EnsureLazyDiscovery();
  if (!m_process)
    return nullptr;
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session || reg >= session->reg_infos.size())
    return nullptr;
  return &session->reg_infos[reg];
}

uint32_t RegisterContextIntelGT::GetRegisterSetCount() const {
  EnsureLazyDiscovery();
  if (!m_process)
    return 0;
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return 0;
  return static_cast<uint32_t>(session->reg_sets.size());
}

const RegisterSet *
RegisterContextIntelGT::GetRegisterSet(uint32_t set_index) const {
  EnsureLazyDiscovery();
  if (!m_process)
    return nullptr;
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session || set_index >= session->reg_sets.size())
    return nullptr;
  return &session->reg_sets[set_index];
}

Status RegisterContextIntelGT::ReadRegister(const RegisterInfo *reg_info,
                                            RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("null RegisterInfo");

  if (m_is_shadow_thread) {
    // Return zeros so the remote 'p' packet probe succeeds.
    std::vector<uint8_t> zeros(reg_info->byte_size, 0);
    Status error;
    reg_value.SetFromMemoryData(*reg_info, zeros.data(), reg_info->byte_size,
                                eByteOrderLittle, error);
    return error;
  }

  if (!m_eu_thread)
    return Status::FromErrorString("No EU thread for register access");

  if (!m_process)
    return Status::FromErrorString("No process for register metadata");

  EnsureLazyDiscovery();

  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return Status::FromErrorString("No device session");

  const uint32_t lldb_reg = reg_info->kinds[eRegisterKindLLDB];
  if (lldb_reg >= session->reg_locations.size())
    return Status::FromErrorString("Invalid register number");

  const DeviceSession::RegLocation &loc = session->reg_locations[lldb_reg];

  // Read the register set from hardware if not yet cached.
  Status error = ReadRegisterSet(loc.regset_index);
  if (error.Fail())
    return error;

  const std::vector<uint8_t> &buf =
      m_eu_thread->GetRegsetData(loc.regset_index);

  // "ip" pseudo-register: raw 32-bit instruction pointer from CR0.dword2.
  if (lldb_reg == session->ip_reg_num && loc.full_byte_size >= 12) {
    uint32_t ip = 0;
    const size_t read_offset =
        static_cast<size_t>(loc.byte_offset) + kCR0_DWORD2_OFFSET;
    if (read_offset + sizeof(ip) > buf.size())
      return Status("register set buffer too small to read ip");
    memcpy(&ip, buf.data() + read_offset, sizeof(ip));
    reg_value.SetUInt32(ip);
    return Status();
  }

  // PC pseudo-register: 64-bit PC = isabase + 32-bit IP from CR0 dword 2.
  // TODO: missing correct XE3P+ support.
  if (lldb_reg == session->pc_reg_num && loc.full_byte_size >= 12) {
    uint32_t ip = 0;
    const size_t read_offset =
        static_cast<size_t>(loc.byte_offset) + kCR0_DWORD2_OFFSET;
    if (read_offset + sizeof(ip) > buf.size())
      return Status("register set buffer too small to read PC");
    memcpy(&ip, buf.data() + read_offset, sizeof(ip));

    uint64_t isabase = 0;
    if (session->isabase_reg_num != UINT32_MAX) {
      const DeviceSession::RegLocation &isa_loc =
          session->reg_locations[session->isabase_reg_num];
      Status error = ReadRegisterSet(isa_loc.regset_index);
      if (error.Success()) {
        const auto &sba_buf = m_eu_thread->GetRegsetData(isa_loc.regset_index);
        if (isa_loc.byte_offset + sizeof(isabase) <= sba_buf.size())
          memcpy(&isabase, sba_buf.data() + isa_loc.byte_offset,
                 sizeof(isabase));
      }
    }

    uint64_t pc = isabase + (uint64_t)ip;
    reg_value.SetUInt64(pc);
    return Status();
  }

  uint32_t read_offset = loc.byte_offset;

  if (read_offset + loc.byte_size > buf.size()) {
    return Status::FromErrorString("Register offset out of bounds");
  }
  reg_value.SetFromMemoryData(*reg_info, buf.data() + read_offset,
                              static_cast<uint32_t>(loc.byte_size),
                              eByteOrderLittle, error);
  return error;
}

Status RegisterContextIntelGT::WriteRegister(const RegisterInfo *reg_info,
                                             const RegisterValue &reg_value) {
  if (m_is_shadow_thread)
    return Status::FromErrorString("No registers for shadow thread");

  if (!m_eu_thread)
    return Status::FromErrorString("No EU thread for register access");

  if (!m_process)
    return Status::FromErrorString("No process for register metadata");

  EnsureLazyDiscovery();

  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return Status::FromErrorString("No device session");

  const uint32_t lldb_reg = reg_info->kinds[eRegisterKindLLDB];
  if (lldb_reg >= session->reg_locations.size())
    return Status::FromErrorString("Invalid register number");

  const DeviceSession::RegLocation &loc = session->reg_locations[lldb_reg];
  const zet_debug_regset_properties_t &props =
      session->regset_props[loc.regset_index];
  Status error;

  if (!(props.generalFlags & kRegsetFlagWriteable))
    return Status::FromErrorString("Register set is not writeable");

  // Ensure the regset is in cache before patching.
  error = ReadRegisterSet(loc.regset_index);
  if (error.Fail())
    return error;

  std::vector<uint8_t> write_buf = m_eu_thread->GetRegsetData(loc.regset_index);

  uint32_t write_offset = loc.byte_offset;
  uint32_t write_size = loc.byte_size;

  reg_value.GetAsMemoryData(*reg_info, write_buf.data() + write_offset,
                            write_size, eByteOrderLittle, error);
  if (error.Fail())
    return error;

  uint32_t reg_in_set = (loc.byte_offset / props.byteSize);
  ze_result_t result = zetDebugWriteRegisters(
      m_eu_thread->GetSession(), m_eu_thread->GetZeThread(), props.type,
      reg_in_set, 1, write_buf.data() + loc.byte_offset);
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat(
        "zetDebugWriteRegisters(type=%u) failed: %s", props.type,
        ZeResultToString(result).data());
  }

  m_eu_thread->InvalidateRegsetCache();

  return Status();
}

Status
RegisterContextIntelGT::ReadAllRegisterValues(WritableDataBufferSP &data_sp) {
  if (m_is_shadow_thread)
    return Status::FromErrorString("No registers for shadow thread");

  if (!m_eu_thread)
    return Status::FromErrorString("No EU thread for register access");

  if (!m_process)
    return Status::FromErrorString("No process for register metadata");

  EnsureLazyDiscovery();

  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return Status::FromErrorString("No device session");

  size_t total = 0;
  for (const auto &loc : session->reg_locations)
    total += loc.byte_size;

  data_sp.reset(new DataBufferHeap(total, 0));
  uint8_t *dst = data_sp->GetBytes();
  size_t offset = 0;

  for (size_t i = 0; i < session->reg_locations.size(); ++i) {
    const DeviceSession::RegLocation &loc = session->reg_locations[i];
    Status error = ReadRegisterSet(loc.regset_index);
    if (error.Fail())
      return error;

    const std::vector<uint8_t> &buf =
        m_eu_thread->GetRegsetData(loc.regset_index);

    if (loc.byte_offset + loc.byte_size <= buf.size())
      memcpy(dst + offset, buf.data() + loc.byte_offset, loc.byte_size);
    offset += loc.byte_size;
  }
  return Status();
}

Status
RegisterContextIntelGT::WriteAllRegisterValues(const DataBufferSP &data_sp) {
  return Status::FromErrorString("WriteAllRegisterValues not supported");
}

std::vector<uint32_t> RegisterContextIntelGT::GetExpeditedRegisters(
    ExpeditedRegs /*exp_type*/) const {
  std::vector<uint32_t> expedited;
  if (m_is_shadow_thread || !m_process)
    return expedited;
  EnsureLazyDiscovery();
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (session && session->pc_reg_num != UINT32_MAX)
    expedited.push_back(session->pc_reg_num);
  return expedited;
}

void RegisterContextIntelGT::InvalidateAllRegisters() {
  if (m_eu_thread)
    m_eu_thread->InvalidateRegsetCache();
}

uint32_t RegisterContextIntelGT::GetPCRegisterNumber() const {
  if (m_is_shadow_thread || !m_process)
    return UINT32_MAX;
  EnsureLazyDiscovery();
  const DeviceSession *session = m_process->GetDeviceSession(m_device_index);
  if (!session)
    return UINT32_MAX;
  return session->pc_reg_num;
}
