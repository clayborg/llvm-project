//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H

#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/NVGPU/SASSRegisterInfo.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class RegisterContextNVGPUCore : public RegisterContext {
public:
  /// Construct a RegisterContext for a `ThreadNVGPUCore`. Decodes the lane,
  /// warp, and device rows and copies the lane's per-class register slices
  /// (R/P/UR/UP) into `m_register_data`, which is laid out exactly like
  /// `sass::RegisterLayout`. After construction, `ReadRegister` /
  /// `ReadAllRegisterValues` are direct memory reads against that buffer.
  ///
  /// \pre `thread` must be a `ThreadNVGPUCore`.
  ///
  /// \param[in] thread
  ///     The owning thread; must be a `ThreadNVGPUCore`.
  ///
  /// \param[in] core
  ///     ObjectFile for the corefile, used to read the lane / warp / device
  ///     section data. May be null, in which case the register buffer is
  ///     left zero-initialized.
  RegisterContextNVGPUCore(Thread &thread, ObjectFile *core);

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
  /// Packed register buffer matching `sass::RegisterLayout`. The constructor
  /// fills it from the corefile's lane / warp / register / predicate
  /// sections; everything not present in the corefile stays zero. Each
  /// `RegisterInfo::byte_offset` / `byte_size` from `sass::GetRegisterInfos()`
  /// indexes directly into this buffer, so register reads are a single
  /// `SetFromMemoryData` call -- no per-class dispatch.
  sass::RegisterLayout m_register_data{};
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_REGISTERCONTEXTNVGPUCORE_H
