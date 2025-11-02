//===-- RegisterContextAmdGpuImpl.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTAMDGPUIMPL_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTAMDGPUIMPL_H

#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lldb_private {

class RegisterValue;

/// \class RegisterContextAmdGpuImpl
///
/// Shared implementation for AMDGPU register context that can be used by both
/// live debugging (lldb-server) and core file debugging (amdgpu-core plugin).
/// This class contains all the common logic for querying AMD dbgapi for
/// register information and reading/writing register values.
class RegisterContextAmdGpuImpl {
public:
  RegisterContextAmdGpuImpl(amd_dbgapi_architecture_id_t architecture_id);
  ~RegisterContextAmdGpuImpl() = default;

  /// Invalidate all register values.
  void InvalidateAllRegisters();

  /// Read a single register from AMD dbgapi.
  /// \param[in] wave_id The AMD dbgapi wave ID to read from.
  /// \param[in] reg_info The register information.
  /// \return Status object indicating success or failure.
  Status ReadRegister(std::optional<amd_dbgapi_wave_id_t> wave_id,
                      const RegisterInfo *reg_info);

  /// Write a single register value to the local buffer.
  /// \param[in] reg_info The register information.
  /// \param[in] reg_value The value to write.
  /// \return Status object indicating success or failure.
  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value);

  /// Read all registers from AMD dbgapi.
  /// \param[in] wave_id The AMD dbgapi wave ID to read from.
  /// \return Status object indicating success or failure.
  Status ReadAllRegisters(std::optional<amd_dbgapi_wave_id_t> wave_id);

  /// Get register value from the local buffer.
  /// \param[in] reg_info The register information.
  /// \param[out] reg_value The register value to populate.
  /// \return Status object indicating success or failure.
  Status GetRegisterValue(const RegisterInfo *reg_info,
                          RegisterValue &reg_value) const;

  // Accessors
  size_t GetRegisterCount() const;
  size_t GetRegisterBufferSize() const;
  uint32_t GetPCRegisterNumber() const;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) const;
  const RegisterSet *GetRegisterSet(size_t set_index) const;
  size_t GetRegisterSetCount() const;

  amd_dbgapi_register_id_t GetAMDRegisterID(uint32_t lldb_regnum) const;

  // Access to register data buffer for bulk operations
  uint8_t *GetRegisterDataBuffer() { return m_register_data.data(); }
  const uint8_t *GetRegisterDataBuffer() const {
    return m_register_data.data();
  }
  bool IsRegisterValid(uint32_t lldb_regnum) const;

private:
  /// Initialize register information from AMD dbgapi.
  /// This queries the dbgapi for all registers, their properties, and
  /// organizes them into register sets.
  /// \return true if initialization succeeded, false otherwise.
  bool InitializeRegisterInfo();

  /// Internal implementation for InitializeRegisterInfo
  bool InitializeRegisterInfoImpl();

  /// Initialize the register data buffer.
  void InitializeRegisterData();

  // AMD dbgapi architecture ID
  amd_dbgapi_architecture_id_t m_architecture_id;

  // Register data and validity tracking (per-instance)
  std::vector<uint8_t> m_register_data;
  std::vector<bool> m_register_valid;

  // Architecture-specific register information (indexed by architecture_id)
  struct ArchitectureRegisterInfo {
    std::vector<RegisterInfo> reg_infos;
    std::vector<RegisterSet> register_sets;
    std::unordered_map<uint32_t, amd_dbgapi_register_id_t>
        lldb_num_to_amd_reg_id;
    size_t register_count = 0;
    size_t register_buffer_size = 0;
    uint32_t pc_register_num = 0;
  };

  static std::unordered_map<uint64_t, ArchitectureRegisterInfo>
      s_architecture_register_infos;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTAMDGPUIMPL_H
