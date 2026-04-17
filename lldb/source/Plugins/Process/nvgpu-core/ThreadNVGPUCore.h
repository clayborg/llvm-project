//===-- ThreadNVGPUCore.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H

#include "NVGPUCoreData.h"
#include "lldb/Target/Thread.h"

class ProcessNVGPUCore;

namespace lldb_private {

class ThreadNVGPUCore : public Thread {
public:
  /// Construct a thread for one lane in the corefile. Resolves the
  /// device/SM/CTA/warp/lane hierarchy pointers from the process's
  /// `NVGPUCoreData` here in the constructor; pointers stay null for
  /// any level whose coord is out of range.
  ///
  /// \pre `process` must be a `ProcessNVGPUCore`.
  ///
  /// \param[in] process
  ///     The owning process.
  ///
  /// \param[in] tid
  ///     The LLDB thread ID assigned to this thread.
  ///
  /// \param[in] coords
  ///     The lane coords this thread represents.
  ThreadNVGPUCore(Process &process, lldb::tid_t tid,
                  const NVGPULaneCoords &coords);

  ~ThreadNVGPUCore() override;

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

  const char *GetName() override;

  const NVGPULaneCoords &GetCoords() const { return m_coords; }

  /// Accessors for the pre-resolved hierarchy pointers. Return null if the
  /// corresponding level did not resolve (see m_* invariant below).
  const DeviceData *GetDevice() const { return m_dev; }
  const CTAData *GetCTA() const { return m_cta; }
  const WarpData *GetWarp() const { return m_warp; }
  const LaneData *GetLane() const { return m_lane; }

  /// Return the CUDA exception code attributed to this thread, or
  /// `CUDBG_EXCEPTION_NONE` (zero) if this thread did not participate in
  /// a fault. Precedence:
  ///
  ///   1. Per-lane exception is definitive (precise exceptions). A lane
  ///      that didn't execute the faulting instruction can't carry an
  ///      exception from it, so activeLanesMask gating is implicit here.
  ///   2. Otherwise, if this lane's warp caused the SM fault
  ///      (`errorPCValid`) AND this lane was active at fault time
  ///      (`activeLanesMask` bit set), borrow the kind from the SM
  ///      entry -- the only place the kind lives in the corefile format.
  ///      Diverged (inactive) lanes must not inherit the fault. This
  ///      matches the attribution logic in the live-debug NVGPU plugin
  ///      (lldb-server `DeviceState-Decoding.cpp`'s is_active gate).
  ///   3. An SM-level exception alone never attributes: an SM fault
  ///      doesn't mean every warp on that SM faulted.
  ///
  /// \return
  ///     The attributed `CUDBGException_t` as a `uint32_t`, or zero
  ///     (`CUDBG_EXCEPTION_NONE`) if no exception is attributed.
  uint32_t GetAttributedException() const;

protected:
  bool CalculateStopInfo() override;

private:
  NVGPULaneCoords m_coords;

  /// Hierarchy pointers resolved in the constructor. Nested-null-valid:
  /// a non-null later pointer implies all earlier ones are non-null
  /// (m_lane => m_warp => m_cta => m_sm => m_dev). All null if the coords
  /// don't match any entry in the corefile.
  const DeviceData *m_dev = nullptr;
  const SMData *m_sm = nullptr;
  const CTAData *m_cta = nullptr;
  const WarpData *m_warp = nullptr;
  const LaneData *m_lane = nullptr;

  std::string m_name;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
