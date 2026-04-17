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
    : Thread(process, tid), m_coords(coords) {
  // Walk device -> SM -> CTA -> warp -> lane. Bail on any out-of-range
  // coord; later pointers stay null. This establishes the invariant that
  // a non-null later pointer implies all earlier ones are non-null.
  auto &nvgpu_process = static_cast<ProcessNVGPUCore &>(process);
  NVGPUCoreData &core_data = nvgpu_process.GetCoreData();
  ObjectFileELF *core = nvgpu_process.GetCoreObjectFile();

  if (!core || m_coords.dev_idx >= core_data.GetNumDevices())
    return;
  m_dev = &core_data.GetDevice(m_coords.dev_idx);

  llvm::ArrayRef<SMData> sms = m_dev->GetSMs(core);
  if (m_coords.sm_idx >= sms.size())
    return;
  m_sm = &sms[m_coords.sm_idx];

  llvm::ArrayRef<CTAData> ctas = m_sm->GetCTAs(core);
  if (m_coords.cta_idx >= ctas.size())
    return;
  m_cta = &ctas[m_coords.cta_idx];

  llvm::ArrayRef<WarpData> warps = m_cta->GetWarps(core);
  if (m_coords.warp_idx >= warps.size())
    return;
  m_warp = &warps[m_coords.warp_idx];

  llvm::ArrayRef<LaneData> lanes = m_warp->GetLanes(core);
  if (m_coords.lane_idx >= lanes.size())
    return;
  m_lane = &lanes[m_coords.lane_idx];
}

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
      *this, nvgpu_process->GetCoreObjectFile());
}

const char *ThreadNVGPUCore::GetName() {
  if (m_name.empty()) {
    if (m_lane) // m_lane non-null implies m_cta non-null
      m_name = nvgpu::FormatThreadName(
          m_cta->GetEntry().blockIdxX, m_cta->GetEntry().blockIdxY,
          m_cta->GetEntry().blockIdxZ, m_lane->GetEntry().threadIdxX,
          m_lane->GetEntry().threadIdxY, m_lane->GetEntry().threadIdxZ);
    if (m_name.empty())
      m_name = "NVIDIA GPU Thread";
  }
  return m_name.c_str();
}

uint32_t ThreadNVGPUCore::GetAttributedException() const {
  // Policy documented on the declaration in ThreadNVGPUCore.h.
  if (m_lane && m_lane->GetEntry().exception != 0)
    return m_lane->GetEntry().exception;
  if (m_warp && m_warp->GetEntry().errorPCValid &&
      (m_warp->GetEntry().activeLanesMask & (1u << m_coords.lane_idx)))
    return m_sm->GetEntry().exception; // m_warp non-null implies m_sm
  return 0;
}

bool ThreadNVGPUCore::CalculateStopInfo() {
  CUDBGException_t exc =
      static_cast<CUDBGException_t>(GetAttributedException());
  if (exc != CUDBG_EXCEPTION_NONE) {
    std::string desc =
        ("CUDA Exception: " + CUDAExceptionToString(exc)).str();
    SetStopInfo(StopInfo::CreateStopReasonWithException(*this, desc.c_str()));
  } else {
    SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, SIGTRAP));
  }

  return true;
}
