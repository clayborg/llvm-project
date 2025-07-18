//===-- RegisterContextNVIDIAGPU.cpp //------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVIDIAGPU.h"

#include "NVIDIAGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ThreadNVIDIAGPU.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

#define REG_OFFSET(Reg) offsetof(RegisterContextNVIDIAGPU::RegisterContext, Reg)

/// LLDB register numbers must start at 0 and be contiguous with no gaps.
/// See
/// https://github.com/NVIDIA/cuda-gdb/blob/nvidia-gdb-13.2/gdb/cuda/cuda-tdep.h#L111
enum LLDBRegNum : uint32_t {
  LLDB_PC = 0,
  LLDB_ERROR_PC,
  LLDB_SP, // r1
  LLDB_FP, // r2
  kNumRegs
};

/// DWARF register numbers should match the register numbers that the compiler
/// uses in the DWARF debug info. They can be any number and do not need to
/// be in increasing order or consective, there can be gaps. The compiler has
/// dedicated register numbers for any DWARF that references registers, like
/// location expressions and .debug_frame unwind info.
enum DWARFRegNum : uint32_t {
  DWARF_PC = 128,
  DWARF_ERROR_PC,
  DWARF_SP,
  DWARF_FP,
};

/// Compiler registers should match the register numbers that the compiler
/// uses in runtime information. They can be any number and do not need to
/// be in increasing order or consective, there can be gaps. The compiler has
/// dedicated register numbers for any runtime information that references
/// registers, like .eh_frame unwind info. Many times these numbers match the
/// DWARF register numbers, but not always.
enum CompilerRegNum : uint32_t {
  EH_FRAME_PC = 1000,
  EH_FRAME_ERROR_PC,
  EH_FRAME_SP,
  EH_FRAME_FP,
};

static uint32_t g_gpr_regnums[] = {LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP};

static const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     sizeof(g_gpr_regnums) / sizeof(g_gpr_regnums[0]), g_gpr_regnums}};

/// Define all of the information about all registers. The register info structs
/// are accessed by the LLDB register numbers, which are defined above.
static const RegisterInfo g_reg_infos[LLDBRegNum::kNumRegs] = {
    {
        "PC",           // RegisterInfo::name
        nullptr,        // RegisterInfo::alt_name
        8,              // RegisterInfo::byte_size
        REG_OFFSET(PC), // RegisterInfo::byte_offset
        eEncodingUint,  // RegisterInfo::encoding
        eFormatHex,     // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_PC,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_PC,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_PC, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_PC, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_PC  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "errorPC",           // RegisterInfo::name
        nullptr,             // RegisterInfo::alt_name
        8,                   // RegisterInfo::byte_size
        REG_OFFSET(errorPC), // RegisterInfo::byte_offset
        eEncodingUint,       // RegisterInfo::encoding
        eFormatHex,          // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_ERROR_PC, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_ERROR_PC,    // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_ARG1, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_ERROR_PC, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_ERROR_PC  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "SP",           // RegisterInfo::name
        nullptr,        // RegisterInfo::alt_name
        4,              // RegisterInfo::byte_size
        REG_OFFSET(SP), // RegisterInfo::byte_offset
        eEncodingUint,  // RegisterInfo::encoding
        eFormatHex,     // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_SP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_SP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_SP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_SP  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "FP",           // RegisterInfo::name
        nullptr,        // RegisterInfo::alt_name
        4,              // RegisterInfo::byte_size
        REG_OFFSET(FP), // RegisterInfo::byte_offset
        eEncodingUint,  // RegisterInfo::encoding
        eFormatHex,     // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_FP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_FP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_FP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_FP  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
};

RegisterContextNVIDIAGPU::RegisterContextNVIDIAGPU(ThreadNVIDIAGPU &thread)
    : NativeRegisterContext(thread) {
  InvalidateAllRegisters();
}

void RegisterContextNVIDIAGPU::InvalidateAllRegisters() {
  m_regs_value_is_valid.reset();
  m_did_read_already = false;
}

ThreadNVIDIAGPU &RegisterContextNVIDIAGPU::GetGPUThread() {
  return static_cast<ThreadNVIDIAGPU &>(GetThread());
}

CUDBGAPI RegisterContextNVIDIAGPU::GetDebuggerAPI() {
  return GetGPUThread().GetGPU().GetDebuggerAPI();
}

void RegisterContextNVIDIAGPU::ReadAllRegsFromDevice() {
  if (m_did_read_already)
    return;

  m_did_read_already = true;

  Log *log = GetLog(GDBRLog::Plugin);

  PhysicalCoords physical_coords = GetGPUThread().GetPhysicalCoords();

  if (!physical_coords.IsValid()) {
    m_regs.regs.PC = 0;
    m_regs_value_is_valid[LLDB_PC] = true;
    m_regs.regs.errorPC = -1;
    m_regs_value_is_valid[LLDB_ERROR_PC] = false;
    for (size_t i = LLDB_SP; i < kNumRegs; i++) {
      m_regs.data[i] = 0;
      m_regs_value_is_valid[i] = true;
    }
    return;
  }

  if (physical_coords.dev_id == -1 || physical_coords.sm_id == -1 ||
      physical_coords.warp_id == -1 || physical_coords.lane_id == -1) {
    LLDB_LOG(log, "ReadRegs skipped because of invalid physical coords");
    return;
  }

  CUDBGAPI api = GetDebuggerAPI();

  {
    uint64_t pc = 0;
    CUDBGResult res = api->readVirtualPC(
        physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
        physical_coords.lane_id, &pc);
    if (res == CUDBG_SUCCESS) {
      m_regs.regs.PC = pc;
      m_regs_value_is_valid[LLDB_PC] = true;
    } else {
      m_regs_value_is_valid[LLDB_PC] = false;
    }
  }

  {
    uint64_t error_pc = -1;
    bool error_pc_valid = false;
    CUDBGResult res =
        api->readErrorPC(physical_coords.dev_id, physical_coords.sm_id,
                         physical_coords.warp_id, &error_pc, &error_pc_valid);
    // use valid
    if (res == CUDBG_SUCCESS) {
      m_regs.regs.errorPC = error_pc;
      m_regs_value_is_valid[LLDB_ERROR_PC] = true;
    } else {
      m_regs.regs.errorPC = -1;
      m_regs_value_is_valid[LLDB_ERROR_PC] = false;
    }
  }

  for (size_t reg_num = LLDB_SP; reg_num < kNumRegs; reg_num++) {
    uint32_t val = 0;
    CUDBGResult res = api->readRegister(
        physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
        physical_coords.lane_id, reg_num, &val);
    if (res == CUDBG_SUCCESS) {
      m_regs.data[reg_num] = val;
      m_regs_value_is_valid[reg_num] = true;
    } else {
      m_regs_value_is_valid[reg_num] = false;
    }
  }
}

uint32_t RegisterContextNVIDIAGPU::GetRegisterSetCount() const {
  return sizeof(g_reg_sets) / sizeof(g_reg_sets[0]);
}

uint32_t RegisterContextNVIDIAGPU::GetRegisterCount() const { return kNumRegs; }

uint32_t RegisterContextNVIDIAGPU::GetUserRegisterCount() const {
  return GetRegisterCount();
}

const RegisterInfo *
RegisterContextNVIDIAGPU::GetRegisterInfoAtIndex(uint32_t reg) const {
  if (reg < kNumRegs)
    return &g_reg_infos[reg];
  return nullptr;
}

const RegisterSet *
RegisterContextNVIDIAGPU::GetRegisterSet(uint32_t set_index) const {
  if (set_index < GetRegisterSetCount())
    return &g_reg_sets[set_index];
  return nullptr;
}

Status RegisterContextNVIDIAGPU::ReadRegister(const RegisterInfo *reg_info,
                                              RegisterValue &reg_value) {
  ReadAllRegsFromDevice();
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  reg_value.SetUInt64(m_regs.data[lldb_reg_num]);

  return Status();
}

Status RegisterContextNVIDIAGPU::WriteRegister(const RegisterInfo *reg_info,
                                               const RegisterValue &reg_value) {
  const uint32_t lldb_reg_num = reg_info->kinds[eRegisterKindLLDB];
  bool success = false;
  uint64_t new_value = reg_value.GetAsUInt64(UINT64_MAX, &success);
  if (!success)
    return Status::FromErrorString("register write failed");
  m_regs.data[lldb_reg_num] = new_value;
  m_regs_value_is_valid[lldb_reg_num] = true;
  return Status();
}

Status RegisterContextNVIDIAGPU::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ReadAllRegsFromDevice();
  const size_t regs_byte_size = sizeof(m_regs);
  data_sp.reset(new DataBufferHeap(regs_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  memcpy(dst, &m_regs.data[0], sizeof(m_regs));
  return Status();
}

Status RegisterContextNVIDIAGPU::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  const size_t regs_byte_size = sizeof(m_regs);

  if (!data_sp) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextNVIDIAGPU::ReadAllRegisterValues() invalid data_sp "
        "provided");
  }

  if (data_sp->GetByteSize() != regs_byte_size) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextNVIDIAGPU::WriteAllRegisterValues() data_sp contained "
        "mismatched data size, expected %" PRIu64 ", actual %" PRIu64,
        regs_byte_size, data_sp->GetByteSize());
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    return Status::FromErrorStringWithFormat(
        "RegisterContextNVIDIAGPU::WriteAllRegisterValues() "
        "DataBuffer::GetBytes() returned a null pointer");
  }
  memcpy(&m_regs.data[0], src, sizeof(m_regs));
  return Status();
}

std::vector<uint32_t>
RegisterContextNVIDIAGPU::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    g_expedited_regs.push_back(LLDB_PC);
    g_expedited_regs.push_back(LLDB_ERROR_PC);
    g_expedited_regs.push_back(LLDB_SP);
    g_expedited_regs.push_back(LLDB_FP);
  }
  return g_expedited_regs;
}
