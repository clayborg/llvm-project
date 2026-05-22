//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H

#include "CudbgEntryParser.h"

#include "lldb/Target/Thread.h"
#include "lldb/lldb-forward.h"

#include <optional>

namespace lldb_private {

/// One thread per active GPU lane. Identifies its lane via a `lldb::SectionSP`
/// to the lane container in the synthetic NVGPU hierarchy
/// (`nvgpucore.devN.smN.ctaN.warpN.laneN`). The lane container's data window
/// is its row in the nvgpu-lane-table; reading that section gives the
/// `CudbgThreadTableEntry` directly. Per-lane register / predicate /
/// local-memory sections live as named children (`regs`, `preds`, `local`).
class ThreadNVGPUCore : public Thread {
public:
  /// Construct a thread for one lane.
  ///
  /// \param[in] process
  ///     The owning process.
  ///
  /// \param[in] tid
  ///     The LLDB thread ID assigned to this thread.
  ///
  /// \param[in] lane_section_sp
  ///     The nvgpu-lane container `Section` for this thread's lane.
  ///
  /// \param[in] lane_idx
  ///     The lane index 0..31 within the parent warp.
  ThreadNVGPUCore(Process &process, lldb::tid_t tid,
                  lldb::SectionSP lane_section_sp, uint32_t lane_idx);

  ~ThreadNVGPUCore() override;

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

  const char *GetName() override;

  lldb::SectionSP GetLaneSection() const { return m_lane_section_sp; }

  uint32_t GetLaneIndex() const { return m_lane_idx; }

  lldb::SectionSP GetWarpSection() const;
  lldb::SectionSP GetCTASection() const;
  lldb::SectionSP GetSMSection() const;
  lldb::SectionSP GetDeviceSection() const;

  /// Return the CUDA exception code attributed to this thread, or
  /// `CUDBG_EXCEPTION_NONE` (zero) if this thread did not participate in a
  /// fault.
  uint32_t GetAttributedException() const {
    return m_stop_attribution ? m_stop_attribution->attributed_exception : 0;
  }

  /// True if this lane stopped at an inline `trap;` / `__trap()`.
  bool IsAtTrap() const {
    return m_stop_attribution && m_stop_attribution->at_trap;
  }

protected:
  bool CalculateStopInfo() override;

private:
  lldb::SectionSP m_lane_section_sp;
  uint32_t m_lane_idx;
  std::string m_name;
  std::optional<nvgpu_core::StopAttribution> m_stop_attribution;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
