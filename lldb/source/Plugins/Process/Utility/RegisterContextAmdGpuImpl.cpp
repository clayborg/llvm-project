//===-- RegisterContextAmdGpuImpl.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextAmdGpuImpl.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <unordered_map>
#include <algorithm>

using namespace lldb;
using namespace lldb_private;

// Define static members
std::vector<RegisterInfo> RegisterContextAmdGpuImpl::g_reg_infos;
std::vector<RegisterSet> RegisterContextAmdGpuImpl::g_register_sets;
std::unordered_map<uint32_t, amd_dbgapi_register_id_t>
    RegisterContextAmdGpuImpl::g_lldb_num_to_amd_reg_id;

RegisterContextAmdGpuImpl::RegisterContextAmdGpuImpl(
    amd_dbgapi_architecture_id_t architecture_id)
    : m_architecture_id(architecture_id) {}

bool RegisterContextAmdGpuImpl::InitializeRegisterInfo() {
  // Only initialize once (check the static shared variable)
  if (!g_reg_infos.empty())
    return true;

  amd_dbgapi_status_t status;

  // Define custom hash functions for register IDs
  struct RegisterClassIdHash {
    std::size_t operator()(const amd_dbgapi_register_class_id_t &id) const {
      return std::hash<uint64_t>{}(id.handle);
    }
  };
  struct RegisterClassIdEqual {
    bool operator()(const amd_dbgapi_register_class_id_t &lhs,
                    const amd_dbgapi_register_class_id_t &rhs) const {
      return lhs.handle == rhs.handle;
    }
  };

  struct RegisterIdHash {
    std::size_t operator()(const amd_dbgapi_register_id_t &id) const {
      return std::hash<uint64_t>{}(id.handle);
    }
  };
  struct RegisterIdEqual {
    bool operator()(const amd_dbgapi_register_id_t &lhs,
                    const amd_dbgapi_register_id_t &rhs) const {
      return lhs.handle == rhs.handle;
    }
  };

  // Get register class ids
  size_t register_class_count;
  amd_dbgapi_register_class_id_t *register_class_ids;
  status = amd_dbgapi_architecture_register_class_list(
      m_architecture_id, &register_class_count, &register_class_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Thread),
              "Failed to get register class list from amd-dbgapi");
    return false;
  }

  // Get register class names
  std::unordered_map<amd_dbgapi_register_class_id_t, std::string,
                     RegisterClassIdHash, RegisterClassIdEqual>
      register_class_names;
  for (size_t i = 0; i < register_class_count; ++i) {
    char *bytes;
    status = amd_dbgapi_architecture_register_class_get_info(
        register_class_ids[i], AMD_DBGAPI_REGISTER_CLASS_INFO_NAME,
        sizeof(bytes), &bytes);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(LLDBLog::Thread),
                "Failed to get register class name from amd-dbgapi");
      return false;
    }
    register_class_names.emplace(register_class_ids[i], bytes);
  }

  // Get all register count
  size_t register_count;
  amd_dbgapi_register_id_t *register_ids;
  status = amd_dbgapi_architecture_register_list(
      m_architecture_id, &register_count, &register_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Thread),
              "Failed to get register list from amd-dbgapi");
    return false;
  }
  m_register_count = register_count;

  std::unordered_map<amd_dbgapi_register_class_id_t,
                     std::vector<amd_dbgapi_register_id_t>, RegisterClassIdHash,
                     RegisterClassIdEqual>
      register_class_to_register_ids;
  for (size_t i = 0; i < register_class_count; ++i) {
    for (size_t j = 0; j < register_count; ++j) {
      amd_dbgapi_register_class_state_t register_class_state;
      status = amd_dbgapi_register_is_in_register_class(
          register_class_ids[i], register_ids[j], &register_class_state);
      if (status == AMD_DBGAPI_STATUS_SUCCESS &&
          register_class_state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER) {
        register_class_to_register_ids[register_class_ids[i]].push_back(
            register_ids[j]);
        break;
      }
    }
  }

  std::vector<amd_dbgapi_register_properties_t> all_register_properties;
  all_register_properties.resize(register_count);
  for (size_t regnum = 0; regnum < register_count; ++regnum) {
    auto &register_properties = all_register_properties[regnum];
    if (amd_dbgapi_register_get_info(
            register_ids[regnum], AMD_DBGAPI_REGISTER_INFO_PROPERTIES,
            sizeof(register_properties),
            &register_properties) != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(LLDBLog::Thread),
                "Failed to get register properties from amd-dbgapi");
      return false;
    }
  }

  std::vector<int> dwarf_regnum_to_gdb_regnum;
  std::unordered_map<amd_dbgapi_register_id_t, std::string, RegisterIdHash,
                     RegisterIdEqual>
      register_names;
  for (size_t i = 0; i < register_count; ++i) {
    // Get register name
    char *bytes;
    status = amd_dbgapi_register_get_info(
        register_ids[i], AMD_DBGAPI_REGISTER_INFO_NAME, sizeof(bytes), &bytes);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      register_names[register_ids[i]] = bytes;
      free(bytes);
    }

    // Get register DWARF number
    uint64_t dwarf_num;
    status = amd_dbgapi_register_get_info(register_ids[i],
                                          AMD_DBGAPI_REGISTER_INFO_DWARF,
                                          sizeof(dwarf_num), &dwarf_num);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      if (dwarf_num >= dwarf_regnum_to_gdb_regnum.size())
        dwarf_regnum_to_gdb_regnum.resize(dwarf_num + 1, -1);

      dwarf_regnum_to_gdb_regnum[dwarf_num] = i;
    }
  }

  amd_dbgapi_register_id_t pc_register_id;
  status = amd_dbgapi_architecture_get_info(
      m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
      sizeof(pc_register_id), &pc_register_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Thread),
              "Failed to get PC register from amd-dbgapi");
    return false;
  }

  // Initialize register infos
  g_reg_infos.resize(register_count);

  // Map from register class ID to register numbers for that class
  std::unordered_map<amd_dbgapi_register_class_id_t, std::vector<uint32_t>,
                     RegisterClassIdHash, RegisterClassIdEqual>
      register_class_to_lldb_regnums;

  // Populate register infos with register information from AMD dbgapi
  for (size_t i = 0; i < register_count; ++i) {
    amd_dbgapi_register_id_t reg_id = register_ids[i];
    RegisterInfo &reg_info = g_reg_infos[i];

    // Set register name from AMD dbgapi
    auto name_it = register_names.find(reg_id);
    if (name_it != register_names.end()) {
      reg_info.name = strdup(name_it->second.c_str());
      reg_info.alt_name = nullptr;
    } else {
      // Fallback name if not found
      char name[16];
      snprintf(name, sizeof(name), "reg%zu", i);
      reg_info.name = strdup(name);
      reg_info.alt_name = nullptr;
    }

    // Get register size from AMD dbgapi
    uint64_t reg_size;
    status = amd_dbgapi_register_get_info(reg_id, AMD_DBGAPI_REGISTER_INFO_SIZE,
                                          sizeof(reg_size), &reg_size);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      reg_info.byte_size = reg_size;
    } else {
      reg_info.byte_size = 8; // Default to 64-bit registers
    }
    reg_info.byte_offset = m_register_buffer_size;
    m_register_buffer_size += reg_info.byte_size;

    // Set encoding and format based on register name
    std::string reg_name =
        name_it != register_names.end() ? name_it->second : "";

    // Check if register name contains indicators of its type
    if (reg_name.find("float") != std::string::npos ||
        reg_name.find("fp") != std::string::npos) {
      reg_info.encoding = eEncodingIEEE754;
      reg_info.format = eFormatFloat;
    } else if (reg_name.find("vec") != std::string::npos ||
               reg_name.find("simd") != std::string::npos) {
      reg_info.encoding = eEncodingVector;
      reg_info.format = eFormatVectorOfUInt8;
    } else if (reg_info.byte_size > 8) {
      reg_info.encoding = eEncodingVector;
      reg_info.format = eFormatVectorOfUInt8;
    } else {
      // Default for other types
      reg_info.encoding = eEncodingUint;
      reg_info.format = eFormatHex;
    }

    // Set register kinds
    reg_info.kinds[eRegisterKindLLDB] = i;
    g_lldb_num_to_amd_reg_id[i] = reg_id;

    // Set DWARF register number if available
    uint64_t dwarf_num;
    status = amd_dbgapi_register_get_info(
        reg_id, AMD_DBGAPI_REGISTER_INFO_DWARF, sizeof(dwarf_num), &dwarf_num);
    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      reg_info.kinds[eRegisterKindDWARF] = dwarf_num;
    } else {
      reg_info.kinds[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }

    // Set EH_FRAME register number (same as DWARF for now)
    reg_info.kinds[eRegisterKindEHFrame] = reg_info.kinds[eRegisterKindDWARF];

    // Set generic register kind
    reg_info.kinds[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;

    // Check if this is the PC register
    if (reg_id.handle == pc_register_id.handle) {
      reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
      m_pc_register_num = i;
    }

    // Add this register to its register classes
    for (size_t j = 0; j < register_class_count; ++j) {
      amd_dbgapi_register_class_state_t register_class_state;
      status = amd_dbgapi_register_is_in_register_class(
          register_class_ids[j], reg_id, &register_class_state);
      if (status == AMD_DBGAPI_STATUS_SUCCESS &&
          register_class_state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER) {
        register_class_to_lldb_regnums[register_class_ids[j]].push_back(i);
      }
    }
  }

  // Create register sets from register classes
  g_register_sets.clear();

  for (size_t i = 0; i < register_class_count; ++i) {
    auto class_id = register_class_ids[i];
    auto name_it = register_class_names.find(class_id);
    if (name_it == register_class_names.end()) {
      continue;
    }

    auto regnums_it = register_class_to_lldb_regnums.find(class_id);
    if (regnums_it == register_class_to_lldb_regnums.end() ||
        regnums_it->second.empty()) {
      continue;
    }

    // Create a new register set for this class
    RegisterSet reg_set;
    reg_set.name = strdup(name_it->second.c_str());

    // Create short name from the full name
    std::string short_name = name_it->second;
    size_t space_pos = short_name.find(' ');
    if (space_pos != std::string::npos) {
      short_name = short_name.substr(0, space_pos);
    } else if (short_name.length() > 3) {
      short_name = short_name.substr(0, 3);
    }
    std::transform(short_name.begin(), short_name.end(), short_name.begin(),
                   ::tolower);
    reg_set.short_name = strdup(short_name.c_str());

    // Get register numbers for this class
    const auto &regnums = regnums_it->second;

    // Store register numbers in a static container
    static std::vector<std::vector<uint32_t>> all_reg_nums;
    all_reg_nums.push_back(regnums);

    reg_set.registers = all_reg_nums.back().data();
    reg_set.num_registers = all_reg_nums.back().size();
    g_register_sets.push_back(reg_set);
  }

  return true;
}

void RegisterContextAmdGpuImpl::InitializeRegisterData() {
  m_register_data.resize(m_register_buffer_size);
  m_register_valid.resize(m_register_count, false);
}

void RegisterContextAmdGpuImpl::InvalidateAllRegisters() {
  for (size_t i = 0; i < m_register_count; ++i)
    m_register_valid[i] = false;
}

Status RegisterContextAmdGpuImpl::ReadRegister(
    std::optional<amd_dbgapi_wave_id_t> wave_id, const RegisterInfo *reg_info) {
  Status error;
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  assert(lldb_reg_num < m_register_count);

  auto amd_reg_id = g_lldb_num_to_amd_reg_id[lldb_reg_num];

  if (!wave_id) {
    // No wave ID, return without error (dummy values)
    return error;
  }

  amd_dbgapi_register_exists_t exists;
  amd_dbgapi_status_t amd_status =
      amd_dbgapi_wave_register_exists(wave_id.value(), amd_reg_id, &exists);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error = Status::FromErrorStringWithFormat(
        "Failed to check register %s existence due to error %d", reg_info->name,
        amd_status);
    return error;
  }
  if (exists != AMD_DBGAPI_REGISTER_PRESENT) {
    error = Status::FromErrorStringWithFormat(
        "Failed to read register %s due to register not present",
        reg_info->name);
    return error;
  }

  amd_status = amd_dbgapi_prefetch_register(wave_id.value(), amd_reg_id,
                                            m_register_data.size() - lldb_reg_num);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error = Status::FromErrorStringWithFormat(
        "Failed to prefetch register %s due to error %d", reg_info->name,
        amd_status);
    return error;
  }

  // Read the register value
  amd_status =
      amd_dbgapi_read_register(wave_id.value(), amd_reg_id, 0, reg_info->byte_size,
                               m_register_data.data() + reg_info->byte_offset);
  if (amd_status != AMD_DBGAPI_STATUS_SUCCESS) {
    error = Status::FromErrorStringWithFormat(
        "Failed to read register %s due to error %d", reg_info->name,
        amd_status);
    return error;
  }
  m_register_valid[lldb_reg_num] = true;
  return error;
}

Status
RegisterContextAmdGpuImpl::WriteRegister(const RegisterInfo *reg_info,
                                         const RegisterValue &reg_value) {
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  const void *new_value = reg_value.GetBytes();
  memcpy(m_register_data.data() + reg_info->byte_offset, new_value,
         reg_info->byte_size);
  m_register_valid[lldb_reg_num] = true;
  return Status();
}

Status RegisterContextAmdGpuImpl::ReadAllRegisters(std::optional<amd_dbgapi_wave_id_t> wave_id) {
  if (wave_id.has_value()) {
    for (size_t i = 0; i < g_reg_infos.size(); ++i) {
      Status error = ReadRegister(wave_id, &g_reg_infos[i]);
      if (error.Fail())
        return error;
     }
  } else {
    // Fill all registers with unique values for testing
    for (uint32_t i = 0; i < g_reg_infos.size(); ++i) {
      memcpy(m_register_data.data() + g_reg_infos[i].byte_offset, &i,
             sizeof(i));
    }
  }
  return Status();
}

Status
RegisterContextAmdGpuImpl::GetRegisterValue(const RegisterInfo *reg_info,
                                            RegisterValue &reg_value) const {
  reg_value.SetBytes(m_register_data.data() + reg_info->byte_offset,
                     reg_info->byte_size, lldb::eByteOrderLittle);
  return Status();
}

const RegisterInfo *
RegisterContextAmdGpuImpl::GetRegisterInfoAtIndex(size_t reg) const {
  if (reg < m_register_count)
    return &g_reg_infos[reg];
  return nullptr;
}

const RegisterSet *
RegisterContextAmdGpuImpl::GetRegisterSet(size_t set_index) const {
  if (set_index < g_register_sets.size())
    return &g_register_sets[set_index];
  return nullptr;
}

amd_dbgapi_register_id_t
RegisterContextAmdGpuImpl::GetAMDRegisterID(uint32_t lldb_regnum) const {
  auto it = g_lldb_num_to_amd_reg_id.find(lldb_regnum);
  if (it != g_lldb_num_to_amd_reg_id.end())
    return it->second;
  return amd_dbgapi_register_id_t{0};
}

bool RegisterContextAmdGpuImpl::IsRegisterValid(uint32_t lldb_regnum) const {
  if (lldb_regnum < m_register_count)
    return m_register_valid[lldb_regnum];
  return false;
}
