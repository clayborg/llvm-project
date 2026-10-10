//===-- RegisterContextIntelGT.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_REGISTERCONTEXTINTELGT_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_REGISTERCONTEXTINTELGT_H

#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/lldb-types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace lldb_private {
namespace lldb_server {

class EUThreadIntelGT;
class ProcessIntelGT;

/// NativeRegisterContext for a single SIMD lane within an EU thread; per-lane
/// view over device-level register metadata discovered lazily on first access.
class RegisterContextIntelGT : public NativeRegisterContext {
public:
  RegisterContextIntelGT(NativeThreadProtocol &native_thread,
                         const ProcessIntelGT *process,
                         std::shared_ptr<EUThreadIntelGT> eu_thread,
                         uint32_t device_index, uint32_t lane_id,
                         uint32_t simd_width, bool is_shadow_thread);

  // ----- NativeRegisterContext interface ------------------------------------

  uint32_t GetRegisterCount() const override;
  uint32_t GetUserRegisterCount() const override;
  const RegisterInfo *GetRegisterInfoAtIndex(uint32_t reg) const override;
  uint32_t GetRegisterSetCount() const override;
  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;
  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;
  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  std::vector<uint32_t>
  GetExpeditedRegisters(ExpeditedRegs exp_type) const override;

  // ----- Additional helpers -------------------------------------------------

  void InvalidateAllRegisters();
  uint32_t GetPCRegisterNumber() const;

private:
  Status ReadRegisterSet(uint32_t regset_index);

  /// Trigger lazy device-level register discovery if needed.
  void EnsureLazyDiscovery() const;

  const ProcessIntelGT *m_process;
  std::shared_ptr<EUThreadIntelGT> m_eu_thread;
  uint32_t m_device_index;
  uint32_t m_lane_id;
  uint32_t m_simd_width;
  bool m_is_shadow_thread;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_REGISTERCONTEXTINTELGT_H
