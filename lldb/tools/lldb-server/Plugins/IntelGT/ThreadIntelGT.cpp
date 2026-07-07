//===-- ThreadIntelGT.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadIntelGT.h"
#include "ProcessIntelGT.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"
#include <signal.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

// ---------------------------------------------------------------------------
// Lane thread constructor
// ---------------------------------------------------------------------------

ThreadIntelGT::ThreadIntelGT(ProcessIntelGT &process, lldb::tid_t tid,
                             std::shared_ptr<EUThreadIntelGT> eu_thread,
                             uint32_t lane_id)
    : NativeThreadProtocol(process, tid, lane_id),
      m_eu_thread(std::move(eu_thread)), m_lane_id(lane_id),
      m_reg_context(*this, static_cast<const ProcessIntelGT *>(&process),
                    m_eu_thread,
                    m_eu_thread ? m_eu_thread->GetDeviceIndex() : 0, lane_id,
                    m_eu_thread ? m_eu_thread->GetSimdWidth() : 1,
                    /*is_shadow_thread=*/false) {
  // Default stop reason: SIGTRAP.
  m_stop_info.reason = eStopReasonSignal;
  m_stop_info.signo = SIGTRAP;
}

// ---------------------------------------------------------------------------
// Shadow thread constructor (private)
// ---------------------------------------------------------------------------

ThreadIntelGT::ThreadIntelGT(ProcessIntelGT &process)
    : NativeThreadProtocol(process, INTELGT_SHADOW_THREAD_ID),
      m_eu_thread(nullptr), m_lane_id(0),
      m_reg_context(*this, static_cast<const ProcessIntelGT *>(&process),
                    nullptr, 0, 0, 1,
                    /*is_shadow_thread=*/true) {
  m_stop_info.reason = eStopReasonNone;
  m_stop_info.signo = 0;
}

// ---------------------------------------------------------------------------
// Factory: shadow thread
// ---------------------------------------------------------------------------

std::unique_ptr<ThreadIntelGT>
ThreadIntelGT::CreateShadowThread(ProcessIntelGT &process) {
  auto t = std::unique_ptr<ThreadIntelGT>(new ThreadIntelGT(process));
  // No stop reason: LLDB auto-continues MODULE_LOAD stops when only the shadow
  // thread exists, which triggers ProcessIntelGT::Resume() -> ACK.
  return t;
}

// ---------------------------------------------------------------------------
// NativeThreadProtocol interface
// ---------------------------------------------------------------------------

std::string ThreadIntelGT::GetName() {
  if (IsShadowThread())
    return "IntelGT Shadow Thread";

  if (!m_eu_thread)
    return "IntelGT Unknown Thread";

  ze_device_thread_t t = m_eu_thread->GetZeThread();
  ze_thread_coord coordinates = {};
  m_eu_thread->GetCoordinates(GetProcess(), coordinates);

  char buf[80];
  snprintf(buf, sizeof(buf),
           "IntelGT EU {%u,%u,%u,%u} lane %u WGID {%u,%u,%u,%u} ", t.slice,
           t.subslice, t.eu, t.thread, m_lane_id, coordinates.x, coordinates.y,
           coordinates.z, coordinates.id_in_wg);
  return buf;
}

StateType ThreadIntelGT::GetState() { return m_state; }

bool ThreadIntelGT::GetStopReason(ThreadStopInfo &stop_info,
                                  std::string &description) {
  // Each lane thread has its own stop reason so Thread::ShouldStop can engage
  // the step plan when the user selects and steps a sibling lane.
  stop_info = m_stop_info;
  description = m_stop_description;
  return true;
}

// ---------------------------------------------------------------------------
// IntelGT-specific helpers
// ---------------------------------------------------------------------------

ze_device_thread_t ThreadIntelGT::GetZeThread() const {
  if (m_eu_thread)
    return m_eu_thread->GetZeThread();
  return {0, 0, 0, 0};
}

uint32_t ThreadIntelGT::GetDeviceIndex() const {
  if (m_eu_thread)
    return m_eu_thread->GetDeviceIndex();
  return 0;
}

void ThreadIntelGT::SetStopReason(StopReason reason, uint32_t signo) {
  // Per-lane stop reason; does not propagate to the EU thread. The shadow
  // thread must stay at eStopReasonNone or LLDB will select it as the stopped
  // thread instead of the real GPU thread that hit the breakpoint.
  if (IsShadowThread() && reason != eStopReasonNone) {
    return;
  }

  m_stop_info = ThreadStopInfo{};
  m_stop_info.reason = reason;
  m_stop_info.signo = signo;
  m_stop_description.clear();
}

ProcessIntelGT &ThreadIntelGT::GetProcess() {
  return static_cast<ProcessIntelGT &>(m_process);
}
