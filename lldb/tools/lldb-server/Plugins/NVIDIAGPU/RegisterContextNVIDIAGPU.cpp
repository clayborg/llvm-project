//===-- RegisterContextNVIDIAGPU.cpp //------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVIDIAGPU.h"

#include "NVIDIAGPU.h"
#include "ThreadNVIDIAGPU.h"
#include "Utils.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace llvm;

#define REG_OFFSET(Reg) offsetof(ThreadRegistersValues, Reg)

#define R_REG_OFFSET(Index)                                                    \
  offsetof(ThreadRegistersValues, R) +                                         \
      Index * sizeof(ThreadRegistersValues::R[0])

// The number of elements in this list must match kNumRRegs.
#define EXPAND_R_REGISTERS(PREFIX)                                             \
  PREFIX##_R0, PREFIX##_R1, PREFIX##_R2, PREFIX##_R3, PREFIX##_R4, PREFIX##_R5,      \
      PREFIX##_R6, PREFIX##_R7, PREFIX##_R8, PREFIX##_R9, PREFIX##_R10,             \
      PREFIX##_R11, PREFIX##_R12, PREFIX##_R13, PREFIX##_R14, PREFIX##_R15,         \
      PREFIX##_R16, PREFIX##_R17, PREFIX##_R18, PREFIX##_R19, PREFIX##_R20,         \
      PREFIX##_R21, PREFIX##_R22, PREFIX##_R23, PREFIX##_R24, PREFIX##_R25,         \
      PREFIX##_R26, PREFIX##_R27, PREFIX##_R28, PREFIX##_R29, PREFIX##_R30,         \
      PREFIX##_R31

/// LLDB register numbers must start at 0 and be contiguous with no gaps.
/// See
/// https://github.com/NVIDIA/cuda-gdb/blob/nvidia-gdb-13.2/gdb/cuda/cuda-tdep.h#L111
enum LLDBRegNum : uint32_t {
  LLDB_PC = 0,
  LLDB_ERROR_PC,
  LLDB_SP,
  LLDB_FP,
  LLDB_RA,
  EXPAND_R_REGISTERS(LLDB),
  kNumRegs,
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
  DWARF_RA,
  EXPAND_R_REGISTERS(DWARF),
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
  EH_FRAME_RA,
  EXPAND_R_REGISTERS(EH_FRAME),
};

static uint32_t g_gpr_regnums[] = {LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP,
                                   LLDB_RA};
static uint32_t g_r_regnums[] = {EXPAND_R_REGISTERS(LLDB)};

static const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     sizeof(g_gpr_regnums) / sizeof(g_gpr_regnums[0]), g_gpr_regnums},
    {"R Registers", "r", kNumRRegs, g_r_regnums}};

#define GENERATE_R_REGISTER_INFO(regnum)                                       \
  {                                                                            \
    "R" #regnum, nullptr, 4, R_REG_OFFSET(regnum), eEncodingUint, eFormatHex,  \
    {                                                                          \
      EH_FRAME_R##regnum, DWARF_R##regnum, LLDB_INVALID_REGNUM,                \
          LLDB_R##regnum, LLDB_R##regnum                                       \
    }                                                                          \
  }

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
            LLDB_PC, // RegisterInfo::kinds[eRegisterKindLLDB]
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
            EH_FRAME_ERROR_PC,   // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_ERROR_PC,      // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_ERROR_PC, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_ERROR_PC, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "RA",             // RegisterInfo::alt_name
        "R[20-21]",       // RegisterInfo::name
        8,                // RegisterInfo::byte_size
        R_REG_OFFSET(20), // RegisterInfo::byte_offset
        eEncodingUint,    // RegisterInfo::encoding
        eFormatHex,       // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_RA,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_RA,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_RA, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_RA, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_RA  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "SP",            // RegisterInfo::name
        "R[0]",          // RegisterInfo::alt_name
        4,               // RegisterInfo::byte_size
        R_REG_OFFSET(0), // RegisterInfo::byte_offset
        eEncodingUint,   // RegisterInfo::encoding
        eFormatHex,      // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_SP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_SP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_SP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "FP",            // RegisterInfo::name
        "R[1]",          // RegisterInfo::alt_name
        4,               // RegisterInfo::byte_size
        R_REG_OFFSET(1), // RegisterInfo::byte_offset
        eEncodingUint,   // RegisterInfo::encoding
        eFormatHex,      // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_FP,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_FP,               // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_REGNUM_GENERIC_FP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    // The number of elements in this list must match kNumRRegs.
    GENERATE_R_REGISTER_INFO(0),
    GENERATE_R_REGISTER_INFO(1),
    GENERATE_R_REGISTER_INFO(2),
    GENERATE_R_REGISTER_INFO(3),
    GENERATE_R_REGISTER_INFO(4),
    GENERATE_R_REGISTER_INFO(5),
    GENERATE_R_REGISTER_INFO(6),
    GENERATE_R_REGISTER_INFO(7),
    GENERATE_R_REGISTER_INFO(8),
    GENERATE_R_REGISTER_INFO(9),
    GENERATE_R_REGISTER_INFO(10),
    GENERATE_R_REGISTER_INFO(11),
    GENERATE_R_REGISTER_INFO(12),
    GENERATE_R_REGISTER_INFO(13),
    GENERATE_R_REGISTER_INFO(14),
    GENERATE_R_REGISTER_INFO(15),
    GENERATE_R_REGISTER_INFO(16),
    GENERATE_R_REGISTER_INFO(17),
    GENERATE_R_REGISTER_INFO(18),
    GENERATE_R_REGISTER_INFO(19),
    GENERATE_R_REGISTER_INFO(20),
    GENERATE_R_REGISTER_INFO(21),
    GENERATE_R_REGISTER_INFO(22),
    GENERATE_R_REGISTER_INFO(23),
    GENERATE_R_REGISTER_INFO(24),
    GENERATE_R_REGISTER_INFO(25),
    GENERATE_R_REGISTER_INFO(26),
    GENERATE_R_REGISTER_INFO(27),
    GENERATE_R_REGISTER_INFO(28),
    GENERATE_R_REGISTER_INFO(29),
    GENERATE_R_REGISTER_INFO(30),
    GENERATE_R_REGISTER_INFO(31),
};

RegisterContextNVIDIAGPU::RegisterContextNVIDIAGPU(ThreadNVIDIAGPU &thread)
    : NativeRegisterContext(thread) {}

void RegisterContextNVIDIAGPU::InvalidateAllRegisters() { m_regs.reset(); }

ThreadNVIDIAGPU &RegisterContextNVIDIAGPU::GetGPUThread() {
  return static_cast<ThreadNVIDIAGPU &>(GetThread());
}

CUDBGAPI RegisterContextNVIDIAGPU::GetDebuggerAPI() {
  return GetGPUThread().GetGPU().GetDebuggerAPI();
}

const ThreadRegistersWithValidity &
RegisterContextNVIDIAGPU::ReadAllRegsFromDevice() {
  if (m_regs)
    return *m_regs;

  m_regs.emplace();
  ThreadRegistersWithValidity &regs = *m_regs;

  PhysicalCoords physical_coords = GetGPUThread().GetPhysicalCoords();

  if (!physical_coords.IsValid()) {
    // We need to send always a PC to the client upon stop events, otherwise the
    // client will be in a borked state.
    regs.val.PC = 0;
    regs.is_valid.PC = true;
    return regs;
  }

  CUDBGAPI api = GetDebuggerAPI();

  {
    uint64_t pc = 0;
    CUDBGResult res = api->readVirtualPC(
        physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
        physical_coords.lane_id, &pc);
    if (res == CUDBG_SUCCESS) {
      regs.val.PC = pc;
      regs.is_valid.PC = true;
    }
  }

  {
    uint64_t error_pc = 0;
    bool error_pc_valid = false;
    CUDBGResult res =
        api->readErrorPC(physical_coords.dev_id, physical_coords.sm_id,
                         physical_coords.warp_id, &error_pc, &error_pc_valid);
    if (res == CUDBG_SUCCESS && error_pc_valid) {
      regs.val.errorPC = error_pc;
      regs.is_valid.errorPC = true;
    }
  }

  DeviceInformation &device_info =
      GetGPUThread().GetGPU().GetDeviceInformation(physical_coords.dev_id);
  size_t num_regs = device_info.GetNumRRegisters();
  num_regs = std::min(num_regs, kNumRRegs);

  CUDBGResult res = api->readRegisterRange(physical_coords.dev_id, physical_coords.sm_id,
                               physical_coords.warp_id, physical_coords.lane_id,
                               0, num_regs, regs.val.R);

  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("RegisterContextNVIDIAGPU::ReadAllRegsFromDevice(). "
                           "readRegisterRange failed: {0}",
                           cudbgGetErrorString(res));
  }

  for (size_t i = 0; i < num_regs; ++i)
    regs.is_valid.R[i] = true;

  return regs;
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
  const ThreadRegistersWithValidity &regs = ReadAllRegsFromDevice();
  int reg_num = reg_info->kinds[eRegisterKindLLDB];

  if (reg_num == LLDB_SP)
    reg_num = LLDB_R0;
  if (reg_num == LLDB_FP)
    reg_num = LLDB_R1;

  if (reg_num == LLDB_PC) {
    if (!regs.is_valid.PC)
      return Status("PC register is invalid");
  } else if (reg_num == LLDB_ERROR_PC) {
    if (!regs.is_valid.errorPC)
      return Status("errorPC register is invalid");
  } else if (reg_num == LLDB_RA) {
    if (!regs.is_valid.R[20] || !regs.is_valid.R[21])
      return Status("RA register is invalid");
  } else {
    int r_index = reg_num - LLDB_R0;

    if (r_index < 0 || r_index >= (int)kNumRRegs)
      return Status::FromErrorStringWithFormatv("unknown R{0} register",
                                                r_index);

    if (!regs.is_valid.R[r_index])
      return Status::FromErrorStringWithFormatv("R{0} register is invalid",
                                                r_index);
  }

  Status error;
  reg_value.SetFromMemoryData(
      *reg_info, (const uint8_t *)&regs.val + reg_info->byte_offset,
      reg_info->byte_size, lldb::eByteOrderLittle, error);
  return error;
}

Status RegisterContextNVIDIAGPU::WriteRegister(const RegisterInfo *reg_info,
                                               const RegisterValue &reg_value) {
  return Status("WriteRegister unimplemented");
}

Status RegisterContextNVIDIAGPU::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return Status("ReadAllRegisterValues unimplemented");
}

Status RegisterContextNVIDIAGPU::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return Status("WriteAllRegisterValues unimplemented");
}

std::vector<uint32_t>
RegisterContextNVIDIAGPU::GetExpeditedRegisters(ExpeditedRegs expType) const {
  static std::vector<uint32_t> g_expedited_regs;
  if (g_expedited_regs.empty()) {
    g_expedited_regs.push_back(LLDB_PC);
    g_expedited_regs.push_back(LLDB_ERROR_PC);
    g_expedited_regs.push_back(LLDB_SP);
    g_expedited_regs.push_back(LLDB_FP);
    g_expedited_regs.push_back(LLDB_RA);
    g_expedited_regs.insert(g_expedited_regs.end(), {EXPAND_R_REGISTERS(LLDB)});
  }
  return g_expedited_regs;
}

std::optional<uint64_t> RegisterContextNVIDIAGPU::ReadErrorPC() {
  const ThreadRegistersWithValidity &regs = ReadAllRegsFromDevice();
  if (regs.is_valid.errorPC)
    return regs.val.errorPC;
  return std::nullopt;
}

ThreadRegisterValidity::ThreadRegisterValidity() {
  PC = false;
  errorPC = false;
  for (size_t i = 0; i < kNumRRegs; ++i)
    R[i] = false;
}
