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

/// One thread per active GPU lane, plus one stand-in thread per faulted SM
/// that has no surviving lane.
///
/// A lane-backed thread identifies its lane via a `lldb::SectionSP` to the
/// lane container in the synthetic NVGPU hierarchy
/// (`nvgpucore.devN.smN.ctaN.warpN.laneN`). The lane container's data window
/// is its row in the nvgpu-lane-table; reading that section gives the
/// `CudbgThreadTableEntry` directly. Per-lane register / predicate /
/// local-memory sections live as named children (`regs`, `preds`, `local`).
///
/// When a kernel faults but its warps run to completion before the coredump
/// is written, the SM row still records the exception type and error PC while
/// every warp is gone. No lane is left to attribute the fault to, so the
/// plugin stands in for it with a lane-less thread built straight from the SM
/// container: the SM's exception becomes its stop reason and the SM's
/// `errorPC` its single frame, which is enough to symbolicate the fault site
/// against the loaded cubin modules. Such a thread has no lane, warp, or CTA
/// -- those accessors return null, and register reads and address-space reads
/// that need them fail. cuda-gdb surfaces the same state as a one-line "The
/// exception was triggered at PC ..." diagnostic rather than a thread
/// (`cuda-exceptions.c:print_exception_origin`).
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

  /// Construct the stand-in thread for an SM-level exception that has no
  /// surviving kernel context (see the class documentation).
  ///
  /// \param[in] process
  ///     The owning process.
  ///
  /// \param[in] tid
  ///     The LLDB thread ID assigned to this thread.
  ///
  /// \param[in] sm_section_sp
  ///     The nvgpu-sm container `Section` whose row recorded the fault.
  ///
  /// \param[in] sm_entry
  ///     That SM's already-decoded row, supplying the exception code and
  ///     error PC.
  ThreadNVGPUCore(Process &process, lldb::tid_t tid,
                  lldb::SectionSP sm_section_sp,
                  const nvgpu_core::SMEntry &sm_entry);

  ~ThreadNVGPUCore() override;

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

  const char *GetName() override;

  /// The lane container backing this thread, or null for the SM-exception
  /// stand-in.
  lldb::SectionSP GetLaneSection() const { return m_lane_section_sp; }

  uint32_t GetLaneIndex() const { return m_lane_idx; }

  /// Null for the SM-exception stand-in, which has no lane and therefore no
  /// warp or CTA either.
  lldb::SectionSP GetWarpSection() const;
  lldb::SectionSP GetCTASection() const;

  lldb::SectionSP GetSMSection() const { return m_sm_section_sp; }
  lldb::SectionSP GetDeviceSection() const;

  /// Return the CUDA exception code attributed to this thread, or
  /// `CUDBG_EXCEPTION_NONE` (zero) if this thread did not participate in a
  /// fault.
  uint32_t GetAttributedException() const {
    return m_stop_attribution ? m_stop_attribution->attributed_exception : 0;
  }

  /// Address the hardware blamed for the fault, or std::nullopt when none was
  /// recorded. Only ever set for the SM-exception stand-in; a lane-backed
  /// thread exposes its warp's error PC through the `errorPC` register
  /// instead.
  std::optional<lldb::addr_t> GetErrorPC() const { return m_error_pc; }

  /// True if this lane stopped at an inline `trap;` / `__trap()`.
  bool IsAtTrap() const {
    return m_stop_attribution && m_stop_attribution->at_trap;
  }

protected:
  Unwind &GetUnwinder() override;

  bool CalculateStopInfo() override;

private:
  /// Lane container for this thread's lane, or null for the SM-exception
  /// stand-in.
  lldb::SectionSP m_lane_section_sp;
  /// SM container this thread sits on: derived from the lane's parent chain,
  /// or handed directly to the SM-exception constructor.
  lldb::SectionSP m_sm_section_sp;
  /// Lane index 0..31 within the parent warp; zero and unused for the
  /// SM-exception stand-in.
  uint32_t m_lane_idx;
  std::optional<lldb::addr_t> m_error_pc;
  std::string m_name;
  std::optional<nvgpu_core::StopAttribution> m_stop_attribution;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
