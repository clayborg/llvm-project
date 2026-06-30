//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadNVGPUCore.h"
#include "CudbgEntryParser.h"
#include "ProcessNVGPUCore.h"
#include "UnwindNVGPUCore.h"

#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/CUDAException.h"
#include "lldb/Utility/NVGPU/ThreadName.h"

#include <csignal>

using namespace lldb;
using namespace lldb_private;

ThreadNVGPUCore::ThreadNVGPUCore(Process &process, tid_t tid,
                                 SectionSP lane_section_sp, uint32_t lane_idx)
    : Thread(process, tid), m_lane_section_sp(std::move(lane_section_sp)),
      m_lane_idx(lane_idx) {
  // Decode the CTA and lane rows once so the thread name and stop
  // attribution are cached, instead of re-decoding on every query.
  auto &nvgpu_process = static_cast<ProcessNVGPUCore &>(process);
  ObjectFile *core = nvgpu_process.GetCoreObjectFile();
  auto cta_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::CTAEntry>(GetCTASection(), core);
  auto lane_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::LaneEntry>(m_lane_section_sp, core);

  if (cta_or && lane_or)
    m_name = nvgpu_core::FormatThreadName(*cta_or, *lane_or);
  if (m_name.empty())
    m_name = "NVIDIA GPU Thread";

  if (lane_or)
    m_stop_attribution = nvgpu_core::ComputeStopAttribution(
        *lane_or, GetLaneIndex(), GetWarpSection(), GetSMSection(), core);

  Log *log = GetLog(LLDBLog::Process);
  if (!cta_or)
    LLDB_LOG_ERROR(log, cta_or.takeError(),
                   "Failed to decode GPU CTA data for thread {1}: {0}", tid);
  if (!lane_or)
    LLDB_LOG_ERROR(log, lane_or.takeError(),
                   "Failed to decode GPU lane data for thread {1}: {0}", tid);
}

ThreadNVGPUCore::~ThreadNVGPUCore() { DestroyThread(); }

// The 5-deep parent chain (lane -> warp -> cta -> sm -> device -> nvgpucore
// root) is guaranteed intact by `ObjectFileELF::BuildNVGPUSectionList`: a
// ThreadNVGPUCore can only be constructed from a lane container that the
// builder produced, and every container the builder produces has all of its
// ancestors.
SectionSP ThreadNVGPUCore::GetWarpSection() const {
  return m_lane_section_sp->GetParent();
}

SectionSP ThreadNVGPUCore::GetCTASection() const {
  return GetWarpSection()->GetParent();
}

SectionSP ThreadNVGPUCore::GetSMSection() const {
  return GetCTASection()->GetParent();
}

SectionSP ThreadNVGPUCore::GetDeviceSection() const {
  return GetSMSection()->GetParent();
}

RegisterContextSP ThreadNVGPUCore::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  return m_reg_context_sp;
}

Unwind &ThreadNVGPUCore::GetUnwinder() {
  if (!m_unwinder_up)
    m_unwinder_up = std::make_unique<UnwindNVGPUCore>(*this);
  return *m_unwinder_up;
}

RegisterContextSP
ThreadNVGPUCore::CreateRegisterContextForFrame(StackFrame *frame) {
  return GetUnwinder().CreateRegisterContextForFrame(frame);
}

const char *ThreadNVGPUCore::GetName() { return m_name.c_str(); }

bool ThreadNVGPUCore::CalculateStopInfo() {
  // A lane gets a stop reason if it faulted, trap'd, or had its row
  // data fail to decode (surfaced so corrupt corefiles don't silently
  // look healthy). Otherwise it's left with no stop info so suspended
  // lanes don't appear to have hit SIGTRAP.
  if (!m_stop_attribution)
    return true;

  CUDBGException_t exc =
      static_cast<CUDBGException_t>(m_stop_attribution->attributed_exception);
  if (exc != CUDBG_EXCEPTION_NONE) {
    std::string desc = ("CUDA Exception: " + CUDAExceptionToString(exc)).str();
    SetStopInfo(StopInfo::CreateStopReasonWithException(*this, desc.c_str()));
  } else if (m_stop_attribution->at_trap) {
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGTRAP, "trap"));
  } else if (!m_stop_attribution->decode_error.empty()) {
    std::string desc = "error: " + m_stop_attribution->decode_error;
    SetStopInfo(StopInfo::CreateStopReasonWithException(*this, desc.c_str()));
  }
  return true;
}
