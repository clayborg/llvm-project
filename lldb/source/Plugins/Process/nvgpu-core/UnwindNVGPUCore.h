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

/// Unwind for NVGPU corefile threads. Uses the driver's per-lane backtrace
/// table when local memory is absent; otherwise delegates to DWARF-CFI via
/// `UnwindLLDB`.
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
  llvm::SmallVector<lldb::addr_t, 8> m_table_pcs;
  bool m_use_backtrace_table = false;
  bool m_initialized = false;

  UnwindNVGPUCore(const UnwindNVGPUCore &) = delete;
  const UnwindNVGPUCore &operator=(const UnwindNVGPUCore &) = delete;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_UNWINDNVGPUCORE_H
