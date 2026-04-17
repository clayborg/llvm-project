//===-- RegisterContextNVGPUCore.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H

#include "NVGPUCoreData.h"
#include "lldb/Target/RegisterContext.h"

class ObjectFileELF;

namespace lldb_private {

class RegisterContextNVGPUCore : public RegisterContext {
public:
  /// Construct a RegisterContext that borrows its hierarchy pointers
  /// from the owning `ThreadNVGPUCore`. No additional walk of the
  /// device/SM/CTA/warp/lane tables happens here.
  ///
  /// \pre `thread` must be a `ThreadNVGPUCore`.
  ///
  /// \param[in] thread
  ///     The owning thread.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used for lazy register section
  ///     reads).
  RegisterContextNVGPUCore(Thread &thread, ObjectFileELF *core);

  ~RegisterContextNVGPUCore() override;

  void InvalidateAllRegisters() override {}

  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

private:
  ObjectFileELF *m_core = nullptr;

  /// Borrowed from the owning ThreadNVGPUCore. Null if the thread's coords
  /// didn't resolve to a valid warp/lane. Register reads pull bytes directly
  /// from these, aliasing the corefile mmap via NVGPUCoreData's lazy caches
  /// -- no per-RegisterContext copy of the register data. Ordered top-down
  /// (warp before lane) to match ThreadNVGPUCore's hierarchy field order.
  const WarpData *m_warp = nullptr;
  const LaneData *m_lane = nullptr;

  /// Device-advertised register counts (clamped to the SASS max). Used
  /// to bound register indices in ReadRegister.
  uint32_t m_num_gp_regs = 0;
  uint32_t m_num_pred_regs = 0;
  uint32_t m_num_uniform_regs = 0;
  uint32_t m_num_uniform_pred_regs = 0;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
