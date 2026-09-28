//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_UNWINDNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_UNWINDNVGPUCORE_H

#include "lldb/Target/Unwind.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace lldb_private {

class UnwindLLDB;

/// Unwind for NVGPU corefile threads. Walks a precomputed list of frame PCs
/// -- the driver's per-lane backtrace table when local memory is absent, or
/// the SM's error PC alone for the lane-less SM-exception thread -- and
/// otherwise delegates to DWARF-CFI via `UnwindLLDB`.
class UnwindNVGPUCore : public Unwind {
public:
  UnwindNVGPUCore(Thread &thread);
  ~UnwindNVGPUCore() override = default;

protected:
  void DoClear() override;

  uint32_t DoGetFrameCount() override;

  bool DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                             lldb::addr_t &pc,
                             bool &behaves_like_zeroth_frame) override;

  lldb::RegisterContextSP
  DoCreateRegisterContextForFrame(StackFrame *frame) override;

private:
  void EnsureInitialized();

  std::unique_ptr<UnwindLLDB> m_dwarf_unwinder_up;
  /// Frame PCs for the PC-only synthetic unwind, frame 0 first.
  llvm::SmallVector<lldb::addr_t, 8> m_synthetic_pcs;
  bool m_use_synthetic_pcs = false;
  bool m_initialized = false;

  UnwindNVGPUCore(const UnwindNVGPUCore &) = delete;
  const UnwindNVGPUCore &operator=(const UnwindNVGPUCore &) = delete;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_UNWINDNVGPUCORE_H
