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
#include "lldb/Utility/NVGPU/NVGPUSectionID.h"
#include "lldb/Utility/NVGPU/ThreadCoords.h"
#include "lldb/Utility/NVGPU/ThreadName.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <csignal>

using namespace lldb;
using namespace lldb_private;

namespace {

// CUDA driver r525 added CTA cluster indices and grid cluster dimensions;
// r565 added per-CTA dimensions. The writer records the actual dimensions in
// every current CTA row, including an all-zero triple for a nonclustered CTA.
// Older CTA rows fall back to the grid's ordinary (not preferred) dimensions.
void PopulateClusterCoordinates(
    nvgpu::ThreadCoords &coords, const nvgpu_core::CTAEntry &cta,
    const std::optional<nvgpu_core::GridEntry> &grid) {
  if (!cta.HasClusterIndex())
    return;
  nvgpu::Dim3u idx{cta.clusterIdxX, cta.clusterIdxY, cta.clusterIdxZ};
  if (cta.HasClusterDimensions())
    coords.logical.cluster = nvgpu::ClusterInfo{
        idx, {cta.clusterDimX, cta.clusterDimY, cta.clusterDimZ}};
  else if (grid && grid->HasClusterDimensions())
    coords.logical.cluster = nvgpu::ClusterInfo{
        idx, {grid->clusterDimX, grid->clusterDimY, grid->clusterDimZ}};
}

} // namespace

ThreadNVGPUCore::ThreadNVGPUCore(Process &process, tid_t tid,
                                 SectionSP lane_section_sp, uint32_t lane_idx)
    : Thread(process, tid), m_lane_section_sp(std::move(lane_section_sp)),
      m_lane_idx(lane_idx) {
  m_sm_section_sp = GetCTASection()->GetParent();

  // Decode the hierarchy rows.
  auto &nvgpu_process = static_cast<ProcessNVGPUCore &>(process);
  ObjectFile *core = nvgpu_process.GetCoreObjectFile();
  Log *log = GetLog(LLDBLog::Process);
  SectionSP cta_section_sp = GetCTASection();
  SectionSP warp_section_sp = GetWarpSection();
  SectionSP sm_section_sp = GetSMSection();
  SectionSP device_section_sp = GetDeviceSection();

  std::optional<nvgpu_core::CTAEntry> cta_entry;
  std::optional<nvgpu_core::LaneEntry> lane_entry;
  std::optional<nvgpu_core::WarpEntry> warp_entry;
  std::optional<nvgpu_core::SMEntry> sm_entry;
  std::optional<nvgpu_core::DeviceEntry> device_entry;
  std::string warp_error;
  std::string sm_error;

  auto decode_row = [&](auto row_or, auto &row, const char *row_name,
                        std::string *decode_error = nullptr) {
    if (row_or) {
      row = std::move(*row_or);
      return;
    }

    std::string error = llvm::toString(row_or.takeError());
    LLDB_LOG(log, "Failed to decode GPU {0} data for thread {1}: {2}", row_name,
             tid, error);
    if (decode_error)
      *decode_error = std::move(error);
  };

  decode_row(
      nvgpu_core::ReadAndDecode<nvgpu_core::CTAEntry>(cta_section_sp, core),
      cta_entry, "CTA");
  decode_row(
      nvgpu_core::ReadAndDecode<nvgpu_core::LaneEntry>(m_lane_section_sp, core),
      lane_entry, "lane");
  decode_row(
      nvgpu_core::ReadAndDecode<nvgpu_core::WarpEntry>(warp_section_sp, core),
      warp_entry, "warp", &warp_error);
  decode_row(
      nvgpu_core::ReadAndDecode<nvgpu_core::SMEntry>(sm_section_sp, core),
      sm_entry, "SM", &sm_error);
  decode_row(nvgpu_core::ReadAndDecode<nvgpu_core::DeviceEntry>(
                 device_section_sp, core),
             device_entry, "device");

  // Find the grid.
  std::optional<nvgpu_core::GridEntry> grid_entry;
  if (cta_entry && cta_entry->HasGridID()) {
    llvm::Expected<nvgpu_core::GridLookupResult> grid_or =
        nvgpu_core::FindGridByID(device_section_sp, cta_entry->gridId64, core);
    if (!grid_or) {
      LLDB_LOG_ERROR(log, grid_or.takeError(),
                     "Failed to find GPU grid data for thread {1}: {0}", tid);
    } else {
      grid_entry = std::move(grid_or->entry);
    }
  }

  // Build thread coordinates.
  nvgpu::ThreadCoords coords{};
  coords.physical.lane = m_lane_idx;

  if (device_entry && device_entry->HasDeviceID())
    coords.physical.device = device_entry->devId;
  if (sm_entry && sm_entry->HasSMID())
    coords.physical.sm = sm_entry->smId;
  if (warp_entry && warp_entry->HasWarpID())
    coords.physical.warp = warp_entry->warpId;
  if (warp_entry && warp_entry->HasActiveLanesMask())
    coords.is_active = warp_entry->IsLaneActive(m_lane_idx);

  if (cta_entry) {
    if (cta_entry->HasGridID())
      coords.logical.grid_id = cta_entry->gridId64;
    if (cta_entry->HasBlockIndex())
      coords.logical.block_idx = nvgpu::Dim3u{
          cta_entry->blockIdxX, cta_entry->blockIdxY, cta_entry->blockIdxZ};

    PopulateClusterCoordinates(coords, *cta_entry, grid_entry);
  }
  if (grid_entry && grid_entry->HasContextID())
    coords.logical.context_id = grid_entry->contextId;

  if (lane_entry && lane_entry->HasThreadIndex()) {
    coords.logical.thread_idx = nvgpu::Dim3u{
        lane_entry->threadIdxX, lane_entry->threadIdxY, lane_entry->threadIdxZ};
  }

  m_cuda_thread_coords = std::move(coords);

  // Build the thread name from validated coordinates.
  const auto &logical = m_cuda_thread_coords.logical;
  if (logical.block_idx && logical.thread_idx) {
    const nvgpu::Dim3u &block = *logical.block_idx;
    const nvgpu::Dim3u &thread = *logical.thread_idx;
    m_name = nvgpu::FormatThreadName(block.x, block.y, block.z, thread.x,
                                     thread.y, thread.z);
  }
  if (m_name.empty())
    m_name = "NVIDIA GPU Thread";

  // Determine stop attribution.
  if (!lane_entry)
    return;
  if (lane_entry->exception != 0) {
    m_stop_attribution = StopAttribution{lane_entry->exception, false, ""};
    return;
  }
  if (!warp_entry) {
    std::string stop_decode_error =
        llvm::formatv("failed to decode GPU warp data for lane {0}: {1}",
                      m_lane_idx, warp_error)
            .str();
    m_stop_attribution =
        StopAttribution{0, false, std::move(stop_decode_error)};
    return;
  }

  if (!m_cuda_thread_coords.is_active.value_or(false))
    return;

  const bool at_trap = warp_entry->isWarpBroken;
  if (!sm_entry) {
    std::string stop_decode_error;
    if (warp_entry->errorPCValid)
      stop_decode_error =
          llvm::formatv("failed to decode GPU SM data for lane {0}: {1}",
                        m_lane_idx, sm_error)
              .str();
    if (at_trap || !stop_decode_error.empty())
      m_stop_attribution =
          StopAttribution{0, at_trap, std::move(stop_decode_error)};
    return;
  }

  const uint32_t attributed_exception =
      warp_entry->errorPCValid ? sm_entry->exception : 0;
  if (attributed_exception != 0 || at_trap)
    m_stop_attribution = StopAttribution{attributed_exception, at_trap, ""};
}

ThreadNVGPUCore::ThreadNVGPUCore(Process &process, tid_t tid,
                                 SectionSP sm_section_sp,
                                 const nvgpu_core::SMEntry &sm_entry)
    : Thread(process, tid), m_sm_section_sp(std::move(sm_section_sp)),
      m_lane_idx(0) {
  m_name = nvgpu::FormatSMExceptionThreadName(
      nvgpu::DecodeHwIdx(GetDeviceSection()->GetID()), sm_entry.smId);

  if (sm_entry.errorPCValid)
    m_error_pc = sm_entry.errorPC;

  // The SM row is the only record of this fault, so its exception is the stop
  // reason outright: there is no lane row that could take precedence, and no
  // surviving warp whose active-lane mask could gate it.
  m_stop_attribution =
      StopAttribution{sm_entry.exception, /*at_trap=*/false, ""};

  nvgpu::ThreadCoords coords{};
  if (sm_entry.HasSMID())
    coords.physical.sm = sm_entry.smId;

  auto &nvgpu_process = static_cast<ProcessNVGPUCore &>(process);
  llvm::Expected<nvgpu_core::DeviceEntry> device_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::DeviceEntry>(
          GetDeviceSection(), nvgpu_process.GetCoreObjectFile());
  if (device_or && device_or->HasDeviceID())
    coords.physical.device = device_or->devId;
  else if (!device_or)
    LLDB_LOG_ERROR(GetLog(LLDBLog::Process), device_or.takeError(),
                   "Failed to decode GPU device data for SM-exception thread "
                   "{1}: {0}",
                   tid);

  m_cuda_thread_coords = std::move(coords);
}

ThreadNVGPUCore::~ThreadNVGPUCore() { DestroyThread(); }

// The 5-deep parent chain (lane -> warp -> cta -> sm -> device -> nvgpucore
// root) is guaranteed intact by `ObjectFileELF::BuildNVGPUSectionList`: a
// lane-backed ThreadNVGPUCore can only be constructed from a lane container
// that the builder produced, and every container the builder produces has all
// of its ancestors. The SM-exception stand-in enters that chain at the SM, so
// only the levels below it are absent.
SectionSP ThreadNVGPUCore::GetWarpSection() const {
  if (!m_lane_section_sp)
    return nullptr;
  return m_lane_section_sp->GetParent();
}

SectionSP ThreadNVGPUCore::GetCTASection() const {
  SectionSP warp_sp = GetWarpSection();
  if (!warp_sp)
    return nullptr;
  return warp_sp->GetParent();
}

SectionSP ThreadNVGPUCore::GetDeviceSection() const {
  return m_sm_section_sp->GetParent();
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

StructuredData::ObjectSP ThreadNVGPUCore::FetchThreadExtendedInfo() {
  auto info_sp = std::make_shared<StructuredData::Dictionary>();
  info_sp->AddItem(nvgpu::g_thread_coords_key,
                   nvgpu::toStructuredData(m_cuda_thread_coords));
  return info_sp;
}
