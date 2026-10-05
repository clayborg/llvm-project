//===-- ThreadAMDGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadAMDGPU.h"
#include "ProcessAMDGPU.h"
#include "lldb/Utility/AmdDbgApiUtils.h"
#include "lldb/lldb-enumerations.h"
#include <limits>
#include <memory>

using namespace lldb_private;
using namespace lldb_server;

ThreadAMDGPU::ThreadAMDGPU(ProcessAMDGPU &process, lldb::tid_t tid,
                           std::shared_ptr<WaveAMDGPU> wave,
                           amd_dbgapi_lane_id_t lane_id)
    : NativeThreadProtocol(process, tid, AmdDbgApiLaneIdToTid(lane_id),
                           AmdDbgApiWaveIdToTid(wave->GetWaveID())),
      m_reg_context(*this), m_wave(wave), m_lane_id(lane_id) {}

std::unique_ptr<ThreadAMDGPU>
ThreadAMDGPU::CreateGPUShadowThread(ProcessAMDGPU &process) {
  auto shadow_thread = std::make_unique<ThreadAMDGPU>(
      process, AMDGPU_SHADOW_THREAD_ID,
      std::make_shared<WaveAMDGPU>(AMD_DBGAPI_WAVE_NONE), AMD_DBGAPI_LANE_NONE);
  shadow_thread->SetStopReason(lldb::eStopReasonSignal, SIGTRAP);
  return shadow_thread;
}

// NativeThreadProtocol Interface
std::string ThreadAMDGPU::GetName() {
  if (IsShadowThread())
    return "AMD Native Shadow Thread";
  else
    return std::string("AMD GPU Thread ") + std::to_string(m_tid);
}

lldb::StateType ThreadAMDGPU::GetState() { return lldb::eStateStopped; }

bool ThreadAMDGPU::GetStopReason(ThreadStopInfo &stop_info,
                                 std::string &description) {
  if (!m_wave->GetStopReason(stop_info, description))
    return false;

  ProcessAMDGPU &process = GetProcess();
  // amd_dbgapi reports no stop reason for waves stopped by a debugger request.
  // During an interrupt, we synthesize SIGSTOP for only the selected
  // representative thread, while preserving real stop reasons and leaving
  // collateral stopped threads without a reason. This matches native
  // debugging behavior.
  if (stop_info.reason == lldb::eStopReasonNone &&
      process.IsInterruptPending() && process.GetCurrentThreadID() == GetID()) {
    stop_info.reason = lldb::eStopReasonSignal;
    stop_info.signo = SIGSTOP;
  }
  return true;
}

Status ThreadAMDGPU::SetWatchpoint(lldb::addr_t addr, size_t size,
                                   uint32_t watch_flags, bool hardware) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::RemoveWatchpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

ProcessAMDGPU &ThreadAMDGPU::GetProcess() {
  return static_cast<ProcessAMDGPU &>(m_process);
}

const ProcessAMDGPU &ThreadAMDGPU::GetProcess() const {
  return static_cast<const ProcessAMDGPU &>(m_process);
}

bool ThreadAMDGPU::GetIsActive() const {
  if (IsShadowThread())
    return true;

  return m_wave->GetExecMask() & (1ULL << m_lane_id);
}
