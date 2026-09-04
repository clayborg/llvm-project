//===-- EUThreadIntelGT.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EUThreadIntelGT.h"
#include "IntelGTArch.h"
#include "LevelZeroHelpers.h"
#include "ProcessIntelGT.h"
#include "ThreadIntelGT.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <signal.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

// ZET_DEBUG_REGSET_TYPE_* values for reading execution mask and control
// registers.
static constexpr uint32_t kRegsetCE = 4;
static constexpr uint32_t kRegsetSR = 5;
static constexpr uint32_t kRegsetCR = 6;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

EUThreadIntelGT::EUThreadIntelGT(ze_device_thread_t ze_thread,
                                 uint32_t device_index,
                                 zet_debug_session_handle_t session,
                                 uint32_t simd_width, uint32_t device_id)
    : m_ze_thread(ze_thread), m_device_index(device_index), m_session(session),
      m_simd_width(simd_width), m_device_id(device_id),
      m_active_lanes((1u << simd_width) - 1) {
  // Default: all lanes active.  ReadExecutionMask() refines this.
  // Default stop reason: SIGTRAP, matches AMD shadow-thread convention.
  SetStopReason(eStopReasonSignal, SIGTRAP);
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

// SBA register set type and isabase index within it.
static constexpr uint32_t kRegsetSBA = 11;
static constexpr uint32_t kSBA_IsabaseIndex = 4;

void EUThreadIntelGT::ReadStopReason() {
  uint32_t cr0[4] = {};
  ze_result_t result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    SetStopReason(eStopReasonSignal, SIGTRAP);
    return;
  }

  // Cache the PC: isabase (SBA[4], 64-bit) + IP (CR0 DWORD2, 32-bit).
  uint32_t ip = cr0[2]; // CR0 DWORD2 = instruction pointer
  uint64_t sba_regs[9] = {};
  ze_result_t sba_result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetSBA, 0, 9, sba_regs);
  if (sba_result == ZE_RESULT_SUCCESS) {
    m_isa_base = sba_regs[kSBA_IsabaseIndex];
    m_pc = m_isa_base + ip;
  } else {
    m_pc = ip; // Best effort without isabase.
  }

  SetStopReasonFromCR0(cr0[1]);
}

void EUThreadIntelGT::SetStopReasonFromCR0(uint32_t cr0_dword1) {
  // Decode stop reason from CR0[1] bits.

  if (cr0_dword1 & (1u << intelgt::cr0_1_shared_function_exception_status)) {
    ClearExceptionBit(intelgt::cr0_1_shared_function_exception_status);
    SetStopReason(eStopReasonException, SIGSEGV);
    m_stop_description = "Shared function exception";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_oob_status)) {
    ClearExceptionBit(intelgt::cr0_1_oob_status);
    SetStopReason(eStopReasonException, SIGILL);
    if (intelgt::is_xe2_or_later(m_device_id)) {
      m_stop_description = "Systolic exception";
    } else {
      m_stop_description = "Out of bounds";
    }
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_illegal_opcode_status)) {
    ClearExceptionBit(intelgt::cr0_1_illegal_opcode_status);
    SetStopReason(eStopReasonException, SIGILL);
    m_stop_description = "Illegal opcode";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_pagefault_status)) {
    ClearExceptionBit(intelgt::cr0_1_pagefault_status);
    SetStopReason(eStopReasonException, SIGSEGV);
    m_stop_description = "Page fault";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_software_exception_control)) {
    ClearExceptionBit(intelgt::cr0_1_software_exception_control);
    SetStopReason(eStopReasonException, SIGTRAP);
    m_stop_description = "Software exception";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_force_exception_status)) {
    ClearExceptionBit(intelgt::cr0_1_force_exception_status);
    SetStopReason(eStopReasonException, SIGINT);
    m_stop_description = "Force exception";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_external_halt_status)) {
    SetStopReason(eStopReasonException, SIGINT);
    m_stop_description = "External halt";
    return;
  }

  if (cr0_dword1 & (1u << intelgt::cr0_1_breakpoint_status)) {
    if (m_resume_state == ResumeState::Step) {
      SetStopReason(eStopReasonTrace, SIGTRAP);
    } else {
      SetStopReason(eStopReasonBreakpoint, SIGTRAP);
    }
    return;
  }
  // Fallback: treat as SIGTRAP.
  SetStopReason(eStopReasonSignal, SIGTRAP);
}

void EUThreadIntelGT::SetStopReason(StopReason reason, uint32_t signo) {
  m_stop_info = ThreadStopInfo{};
  m_stop_info.reason = reason;
  m_stop_info.signo = signo;
  m_stop_description.clear();
}

void EUThreadIntelGT::ClearExceptionBit(uint32_t bit_position) {
  uint32_t cr0[4] = {};
  ze_result_t result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  cr0[1] &= ~(1u << bit_position);

  result = zetDebugWriteRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
  } else {
  }
}

// ---------------------------------------------------------------------------
// SuppressCurrentBreakpoint
// ---------------------------------------------------------------------------
// Set CR0.0 bit 15 so the EU can execute past the current breakpoint without
// re-triggering. Must be called before zetDebugResume() at every BP.

void EUThreadIntelGT::SuppressCurrentBreakpoint() {
  uint32_t cr0[4] = {};
  ze_result_t result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  cr0[0] |= (1u << intelgt::cr0_0_breakpoint_suppress);

  // Clear CR0.1 breakpoint_status; otherwise the next resume becomes a
  // single-step (suppress current + break on next) instead of a continue.
  cr0[1] &= ~(1u << intelgt::cr0_1_breakpoint_status);

  result = zetDebugWriteRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
  } else {
  }
}

// ---------------------------------------------------------------------------
// PrepareStep
// ---------------------------------------------------------------------------
// Arm a single-step by setting CR0.0 bit 15 (suppress current) and CR0.1
// bit 31 (break on next). zetDebugResume has no step flag; CR0 drives it.

void EUThreadIntelGT::PrepareStep() {
  // CR0 is 4 DWORDs (16 bytes).  Read, modify, write.
  uint32_t cr0[4] = {};
  ze_result_t result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  cr0[0] |= (1u << intelgt::cr0_0_breakpoint_suppress);
  cr0[1] |= (1u << intelgt::cr0_1_breakpoint_status);

  result = zetDebugWriteRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  m_resume_state = ResumeState::Step;
}

// ---------------------------------------------------------------------------
// ClearStepBits
// ---------------------------------------------------------------------------
// Clear CR0.1 bit 31 (breakpoint_status) after a single-step completes.

void EUThreadIntelGT::ClearStepBits() {
  uint32_t cr0[4] = {};
  ze_result_t result =
      zetDebugReadRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  cr0[1] &= ~(1u << intelgt::cr0_1_breakpoint_status);

  result = zetDebugWriteRegisters(m_session, m_ze_thread, kRegsetCR, 0, 1, cr0);
  if (result != ZE_RESULT_SUCCESS) {
    return;
  }

  m_resume_state = ResumeState::Run;
}

// ---------------------------------------------------------------------------
// ReadExecutionMask
// ---------------------------------------------------------------------------
// Active lanes = CE (Channel Enable) & SR0 dispatch mask. Falls back to
// all-lanes-active on register read failure.

void EUThreadIntelGT::ReadExecutionMask() {
  // Try to read CE register (regset type kRegsetCE).
  // CE register set typically has 1 register of 4 bytes (DWORD).
  uint32_t ce_value = 0;
  {
    uint32_t count = 1;
    ze_result_t result = zetDebugReadRegisters(m_session, m_ze_thread,
                                               kRegsetCE, 0, count, &ce_value);
    if (result != ZE_RESULT_SUCCESS) {
      m_active_lanes = (1u << m_simd_width) - 1;
      return;
    }
  }

  // Try to read SR0 (regset type kRegsetSR, DWORD 2 = dispatch mask).
  uint32_t dispatch_mask = 0xFFFFFFFF;
  {
    // SR registers may be larger; read all and extract DWORD 2.
    uint32_t sr_buf[16] = {};
    uint32_t count = 1;
    ze_result_t result = zetDebugReadRegisters(m_session, m_ze_thread,
                                               kRegsetSR, 0, count, sr_buf);
    if (result == ZE_RESULT_SUCCESS) {
      dispatch_mask = sr_buf[2];
    } else {
    }
  }

  m_active_lanes = ce_value & dispatch_mask;

  // Clamp to simd_width bits.
  if (m_simd_width < 32)
    m_active_lanes &= (1u << m_simd_width) - 1;

  // If no lanes are active, keep lane 0 so LLDB has a thread for the EU.
  if (m_active_lanes == 0)
    m_active_lanes = 0x1; // Lane 0 only
}

// ---------------------------------------------------------------------------
// AddLaneThreads
// ---------------------------------------------------------------------------
// Create one ThreadIntelGT per active SIMD lane and append to the list.
// Returns the TID of the lane that receives the stop reason.

lldb::tid_t EUThreadIntelGT::AddLaneThreads(
    ProcessIntelGT &process,
    std::vector<std::unique_ptr<NativeThreadProtocol>> &threads,
    lldb::tid_t tid_base, lldb::tid_t focus_tid) {

  m_tid_base = tid_base;

  // Only active lanes become threads; inactive lanes get their TID back on
  // re-activation because TIDs are derived from EU topology + lane.
  // The focus lane reports the real stop reason; siblings report
  // eStopReasonBreakpoint so Thread::ShouldStop engages the step plan.
  lldb::tid_t first_active_tid = LLDB_INVALID_THREAD_ID;

  // Stable per-lane TID: [slice:16][subslice:16][eu:16][thread+1:8][lane:8].
  // The +1 keeps TIDs >= 0x100 (avoiding LLDB_INVALID_THREAD_ID=0 and the
  // shadow-thread TID=1) and prevents bit-8 aliasing between different
  // hardware thread indices.
  auto lane_tid = [this](uint32_t lane) -> lldb::tid_t {
    return ((uint64_t)m_ze_thread.slice << 48) |
           ((uint64_t)m_ze_thread.subslice << 32) |
           ((uint64_t)m_ze_thread.eu << 16) |
           ((uint64_t)(m_ze_thread.thread + 1) << 8) | lane;
  };

  // Decide the focus lane up front: honour focus_tid if it names an active
  // lane of this EU thread, else fall back to the first active lane.
  lldb::tid_t focus_lane_tid = LLDB_INVALID_THREAD_ID;
  for (uint32_t lane = 0; lane < m_simd_width; ++lane) {
    if (!(m_active_lanes & (1u << lane)))
      continue;
    lldb::tid_t tid = lane_tid(lane);
    if (first_active_tid == LLDB_INVALID_THREAD_ID)
      first_active_tid = tid;
    if (tid == focus_tid)
      focus_lane_tid = tid; // requested lane is still active
  }
  if (focus_lane_tid == LLDB_INVALID_THREAD_ID)
    focus_lane_tid = first_active_tid; // requested lane gone (or none) -> first

  for (uint32_t lane = 0; lane < m_simd_width; ++lane) {
    if (!(m_active_lanes & (1u << lane)))
      continue;

    lldb::tid_t tid = lane_tid(lane);
    auto t =
        std::make_unique<ThreadIntelGT>(process, tid, shared_from_this(), lane);
    t->SetState(eStateStopped);

    if (tid == focus_lane_tid) {
      t->SetStopReason(m_stop_info.reason, m_stop_info.signo);
      t->SetStopDescription(m_stop_description);
    } else {
      // Siblings share the EU's hardware stop; report eStopReasonBreakpoint
      // so Thread::ShouldStop consults the step plan on sibling-lane steps.
      t->SetStopReason(eStopReasonBreakpoint, SIGTRAP);
    }

    threads.push_back(std::move(t));
  }

  return focus_lane_tid != LLDB_INVALID_THREAD_ID ? focus_lane_tid : m_tid_base;
}

// ---------------------------------------------------------------------------
// Shared register data cache
// ---------------------------------------------------------------------------

void EUThreadIntelGT::AllocateRegsetCache(
    const std::vector<zet_debug_regset_properties_t> &props) {
  if (m_regsets_cache_allocated)
    return;
  m_regset_data.resize(props.size());
  m_regset_valid.assign(props.size(), false);
  for (size_t i = 0; i < props.size(); ++i) {
    uint32_t total = props[i].count * props[i].byteSize;
    m_regset_data[i].assign(total, 0);
  }
  m_regsets_cache_allocated = true;
}

Status
EUThreadIntelGT::ReadRegisterSet(uint32_t regset_index,
                                 const zet_debug_regset_properties_t &props) {
  if (regset_index >= m_regset_data.size())
    return Status::FromErrorString("Register set index out of range");

  if (m_regset_valid[regset_index])
    return Status();

  std::vector<uint8_t> &buf = m_regset_data[regset_index];
  if (props.count == 0 || buf.empty())
    return Status::FromErrorString(
        "Register set has zero count or empty buffer");

  ze_result_t result = zetDebugReadRegisters(m_session, m_ze_thread, props.type,
                                             0, props.count, buf.data());
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat(
        "zetDebugReadRegisters(type=%u) failed: %s", props.type,
        ZeResultToString(result).data());
  }

  m_regset_valid[regset_index] = true;
  return Status();
}

bool EUThreadIntelGT::GetCoordinates(ProcessIntelGT &process,
                                     ze_thread_coord &coord) {
  // R0 carries the implicit-args pointer, workgroup coords, and thread id.
  // FIXME: size 16 is hardcoded.
  std::vector<uint32_t> r0(16);
  ze_result_t status = zetDebugReadRegisters(
      m_session, m_ze_thread, 1 /* REGSET_GRF */, 0, 1, r0.data());
  if (status != ZE_RESULT_SUCCESS)
    return false;

  // R0.0[31:6] has the implicit args pointer.
  uint32_t implicit_args = r0[0] >> 6;

  // Get or assign kernel instance ID via ProcessIntelGT (thread-safe).
  coord.kernel_instance = process.GetOrAssignKernelInstanceID(implicit_args);

  // Workgroup ID is passed in {R0.1, R0.6, R0.7}.
  coord.x = r0[1];
  coord.y = r0[6];
  coord.z = r0[7];

  // Thread ID in thread group is in R0.2[7:0]. Number of threads in
  // thread group is in R0.2[31:24].
  coord.id_in_wg = r0[2] & 0x000000FF;
  coord.num_threads_in_wg = r0[2] >> 24;

  return true;
}
bool EUThreadIntelGT::IsRegsetValid(uint32_t regset_index) const {
  if (regset_index >= m_regset_valid.size())
    return false;
  return m_regset_valid[regset_index];
}

const std::vector<uint8_t> &
EUThreadIntelGT::GetRegsetData(uint32_t regset_index) const {
  return m_regset_data[regset_index];
}

void EUThreadIntelGT::InvalidateRegsetCache() {
  std::fill(m_regset_valid.begin(), m_regset_valid.end(), false);
}
