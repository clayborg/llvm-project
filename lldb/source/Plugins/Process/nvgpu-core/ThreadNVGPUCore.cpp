//===-- ThreadNVGPUCore.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadNVGPUCore.h"
#include "ProcessNVGPUCore.h"
#include "RegisterContextNVGPUCore.h"

#include "cudadebugger.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/CUDAException.h"
#include "lldb/Utility/NVGPU/ThreadName.h"

#include <csignal>

using namespace lldb;
using namespace lldb_private;

ThreadNVGPUCore::ThreadNVGPUCore(Process &process, tid_t tid,
                                 const NVGPULaneCoords &coords)
    : Thread(process, tid), m_coords(coords) {}

ThreadNVGPUCore::~ThreadNVGPUCore() { DestroyThread(); }

RegisterContextSP ThreadNVGPUCore::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  return m_reg_context_sp;
}

RegisterContextSP
ThreadNVGPUCore::CreateRegisterContextForFrame(StackFrame *frame) {
  ProcessSP process_sp = GetProcess();
  if (!process_sp) {
    LLDB_LOG(GetLog(LLDBLog::Process),
             "ThreadNVGPUCore: failed to get process for register context");
    return nullptr;
  }
  auto *nvgpu_process = static_cast<ProcessNVGPUCore *>(process_sp.get());
  return std::make_shared<RegisterContextNVGPUCore>(
      *this, nvgpu_process->GetCoreData(), m_coords,
      nvgpu_process->GetCoreObjectFile());
}

/// Resolved hierarchy state for a single GPU lane within the corefile.
struct ResolvedLaneInfo {
  const SMData *sm = nullptr;
  const CTAData *cta = nullptr;
  const WarpData *warp = nullptr;
  const LaneData *lane = nullptr;
};

/// Walk the device -> SM -> CTA -> warp -> lane hierarchy for a given set
/// of coordinates. Returns partially filled results if any level is out
/// of range.
static ResolvedLaneInfo
ResolveLane(NVGPUCoreData &core_data, ObjectFileELF *core,
            const NVGPULaneCoords &coords) {
  ResolvedLaneInfo info;
  if (!core || coords.dev_idx >= core_data.devices.size())
    return info;

  DeviceData &dev = core_data.devices[coords.dev_idx];
  llvm::ArrayRef<SMData> sms = dev.GetSMs(core);
  if (coords.sm_idx >= sms.size())
    return info;
  info.sm = &sms[coords.sm_idx];

  llvm::ArrayRef<CTAData> ctas = info.sm->GetCTAs(core);
  if (coords.cta_idx >= ctas.size())
    return info;
  info.cta = &ctas[coords.cta_idx];

  llvm::ArrayRef<WarpData> warps = info.cta->GetWarps(core);
  if (coords.warp_idx >= warps.size())
    return info;
  info.warp = &warps[coords.warp_idx];

  llvm::ArrayRef<LaneData> lanes = info.warp->GetLanes(core);
  if (coords.lane_idx >= lanes.size())
    return info;
  info.lane = &lanes[coords.lane_idx];

  return info;
}

const char *ThreadNVGPUCore::GetName() {
  if (m_name.empty()) {
    ProcessSP process_sp = GetProcess();
    if (!process_sp)
      return "NVIDIA GPU Thread";
    auto *nvgpu_process = static_cast<ProcessNVGPUCore *>(process_sp.get());
    ResolvedLaneInfo info = ResolveLane(
        nvgpu_process->GetCoreData(), nvgpu_process->GetCoreObjectFile(),
        m_coords);
    if (info.cta && info.lane)
      m_name = nvgpu::FormatThreadName(
          info.cta->entry.blockIdxX, info.cta->entry.blockIdxY,
          info.cta->entry.blockIdxZ, info.lane->entry.threadIdxX,
          info.lane->entry.threadIdxY, info.lane->entry.threadIdxZ);
    if (m_name.empty())
      m_name = "NVIDIA GPU Thread";
  }
  return m_name.c_str();
}

bool ThreadNVGPUCore::CalculateStopInfo() {
  ProcessSP process_sp = GetProcess();
  if (!process_sp)
    return false;

  auto *nvgpu_process = static_cast<ProcessNVGPUCore *>(process_sp.get());
  ResolvedLaneInfo info = ResolveLane(
      nvgpu_process->GetCoreData(), nvgpu_process->GetCoreObjectFile(),
      m_coords);

  // Check for exceptions at lane, then warp level. Only attribute an
  // exception to this thread if it occurred at the lane or warp level.
  // The SM-level exception indicates which SM had a fault but doesn't mean
  // every thread on that SM faulted.
  uint32_t exception = 0;
  if (info.lane)
    exception = info.lane->entry.exception;
  if (exception == 0 && info.warp && info.warp->entry.errorPCValid && info.sm)
    exception = info.sm->entry.exception;

  CUDBGException_t exc = static_cast<CUDBGException_t>(exception);
  if (exc != CUDBG_EXCEPTION_NONE) {
    std::string desc =
        ("CUDA Exception: " + CUDAExceptionToString(exc)).str();
    SetStopInfo(StopInfo::CreateStopReasonWithException(*this, desc.c_str()));
  } else {
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGTRAP));
  }

  return true;
}
