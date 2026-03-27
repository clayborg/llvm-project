//===-- ThreadAMDGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadAMDGPU.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Utility/AmdGpuStopReason.h"
#include <csignal>

using namespace lldb_private;

void ThreadAMDGPU::RefreshStateAfterStop() {
  GetRegisterContext()->InvalidateIfNeeded(false);
}

lldb::RegisterContextSP ThreadAMDGPU::GetRegisterContext() {
  if (!m_reg_context_sp) {
    m_reg_context_sp = std::make_shared<RegisterContextAmdGpu>(*this);
  }
  return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadAMDGPU::CreateRegisterContextForFrame(StackFrame *frame) {
  // For frame 0 (leaf frame), return the live register context.
  // For caller frames (frame > 0), use the unwinder to get the register
  // context which computes caller register values using DWARF unwind info.
  if (frame == nullptr || frame->GetConcreteFrameIndex() == 0)
    return GetRegisterContext();

  return GetUnwinder().CreateRegisterContextForFrame(frame);
}

// NativeThreadProtocol Interface
const char *ThreadAMDGPU::GetName() {
  if (!m_thread_name.empty())
    return m_thread_name.c_str();

  if (!m_wave_id)
    return "AMD Native Shadow Thread";
  
  return "AMD GPU Thread";
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
ThreadAMDGPU::GetSiginfo(size_t max_size) const {
  // TODO: how to implement this?
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no siginfo note");
}

bool ThreadAMDGPU::CalculateStopInfo() {
  lldb::ProcessSP process_sp(GetProcess());
  if (!process_sp)
    return false;

  // The shadow thread (no wave_id) doesn't have GPU-specific stop info.
  if (!m_wave_id) {
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGTRAP));
    return true;
  }

  // Query the wave's stop reason from the AMD debug API.
  amd_dbgapi_wave_stop_reasons_t stop_reason = AMD_DBGAPI_WAVE_STOP_REASON_NONE;
  amd_dbgapi_status_t status =
      amd_dbgapi_wave_get_info(*m_wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                               sizeof(stop_reason), &stop_reason);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    // Matching rocgdb behavior: treat API failure as an error.
    SetStopInfo(StopInfo::CreateStopReasonWithException(
        *this, "failed to query wave stop reason"));
    return true;
  }

  std::string description;
  lldb::StopReason reason =
      GetLldbStopReasonForDbgApiStopReason(stop_reason, &description);

  switch (reason) {
  case lldb::eStopReasonException:
    SetStopInfo(
        StopInfo::CreateStopReasonWithException(*this, description.c_str()));
    break;
  case lldb::eStopReasonBreakpoint:
    SetStopInfo(StopInfo::CreateStopReasonWithBreakpointSiteID(*this, 0));
    break;
  case lldb::eStopReasonWatchpoint:
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGTRAP));
    break;
  case lldb::eStopReasonTrace:
    SetStopInfo(StopInfo::CreateStopReasonToTrace(*this));
    break;
  case lldb::eStopReasonInterrupt:
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGSTOP));
    break;
  default:
    // No recognized stop reason — match rocgdb's GDB_SIGNAL_0 fallback.
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, 0));
    break;
  }
  return true;
}
