//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextNVIDIAGPU.h"

#include "../Utils/Utils.h"
#include "NVIDIAGPU.h"
#include "ThreadNVIDIAGPU.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace llvm;

// Include cudadebugger.h for register class definitions
#include "SASSRegisterNumbers.h"
#include "cudadebugger.h"

#define REG_OFFSET(Reg) offsetof(ThreadRegistersValues, Reg)

#define R_REG_OFFSET(Index)                                                    \
  offsetof(ThreadRegistersValues, regular) +                                   \
      Index * sizeof(ThreadRegistersValues::regular[0])

#define P_REG_OFFSET(Index)                                                    \
  offsetof(ThreadRegistersValues, predicate) +                                 \
      Index * sizeof(ThreadRegistersValues::predicate[0])

#define UR_REG_OFFSET(Index)                                                   \
  offsetof(ThreadRegistersValues, uniform) +                                   \
      Index * sizeof(ThreadRegistersValues::uniform[0])

#define UP_REG_OFFSET(Index)                                                   \
  offsetof(ThreadRegistersValues, uniform_predicate) +                         \
      Index * sizeof(ThreadRegistersValues::uniform_predicate[0])

// Include generated register definitions for all SASS registers
#include "RegisterDefinitionsSASS.inc"

// Use the 255-register versions
#define EXPAND_REGULAR_REGISTERS(PREFIX) EXPAND_REGULAR_REGISTERS_255(PREFIX)
#define EXPAND_UNIFORM_REGISTERS(PREFIX) EXPAND_UNIFORM_REGISTERS_255(PREFIX)
#define EXPAND_PREDICATE_REGISTERS(PREFIX) EXPAND_PREDICATE_REGISTERS_8(PREFIX)
#define EXPAND_UNIFORM_PREDICATE_REGISTERS(PREFIX)                             \
  EXPAND_UNIFORM_PREDICATE_REGISTERS_8(PREFIX)

/// LLDB register numbers must start at 0 and be contiguous with no gaps.
/// See
/// https://github.com/NVIDIA/cuda-gdb/blob/nvidia-gdb-13.2/gdb/cuda/cuda-tdep.h#L111
enum LLDBRegNum : uint32_t {
  LLDB_PC = sass::LLDB_PC,
  LLDB_ERROR_PC = sass::LLDB_ERROR_PC,
  LLDB_SP = sass::LLDB_SP,
  LLDB_FP = sass::LLDB_FP,
  LLDB_RA = sass::LLDB_RA,
  EXPAND_REGULAR_REGISTERS(LLDB),
  LLDB_RZ,                                  // R255 - zero register
  EXPAND_PREDICATE_REGISTERS(LLDB),         // Predicate registers
  EXPAND_UNIFORM_REGISTERS(LLDB),           // Uniform registers
  LLDB_URZ,                                 // UR255 - uniform zero register
  EXPAND_UNIFORM_PREDICATE_REGISTERS(LLDB), // Uniform predicate registers
  kNumRegs,
};

/// DWARF register numbers should match the register numbers that the compiler
/// uses in the DWARF debug info. They can be any number and do not need to
/// be in increasing order or consective, there can be gaps. The compiler has
/// dedicated register numbers for any DWARF that references registers, like
/// location expressions and .debug_frame unwind info.
///
/// For SASS/NVPTX, the compiler uses CUDA-encoded DWARF numbers where the
/// register class is encoded in the top 8 bits and register number in lower 24.
///
/// SASS Register ABI Layout:
/// - r0: Scratch (caller-save)
/// - r1: Stack Pointer (SP) - Special, preserved
/// - r2: Frame Pointer (FP) when required - Preserved (callee-save)
/// - r3: Scratch (caller-save)
/// - r4-r15: Scratch (caller-save) - Used for parameter passing and return
/// values
/// - r16-r31: Preserved (callee-save) - Note: r20-r21 store 64-bit return
/// address
/// - r32-r35: Scratch (caller-save)
/// - r36-r39: Preserved (callee-save)
/// - r40-r43: Scratch (caller-save)
/// - r44-r47: Preserved (callee-save)
/// - r48-r51: Scratch (caller-save)
/// - r52-r55: Preserved (callee-save)
/// - r56-r59: Scratch (caller-save)
/// - r60-r62: Preserved (callee-save)
/// - r64-r254: Alternating groups of 4 (scratch: r64-r67, r72-r75, etc.;
///              preserved: r68-r71, r76-r79, etc.)
/// - r255: Zero register (RZ) - Special, always zero

// Use pseudo-DWARF register numbers from shared header for PC and ErrorPC

enum DWARFRegNum : uint32_t {
  DWARF_PC = sass::DWARF_PSEUDO_PC, // Pseudo-DWARF number for PC
  DWARF_ERROR_PC =
      sass::DWARF_PSEUDO_ERROR_PC, // Pseudo-DWARF number for ErrorPC
  DWARF_SP =
      sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL,
                                    sass::SASS_SP_REG), // R1 (Stack Pointer)
  DWARF_FP =
      sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL,
                                    sass::SASS_FP_REG), // R2 (Frame Pointer)
  // Note: No DWARF_RA because return address spans R20-R21 (64-bit).
  // DWARF cannot represent multi-register values.
  // R registers use CUDA encoding - all 255 registers
  GENERATE_DWARF_REGULAR_DEFS(),
  // R255 - zero register
  DWARF_RZ =
      sass::GetDWARFEncodedRegister(REG_CLASS_REG_FULL, sass::SASS_ZERO_REG),
  // Predicate registers use CUDA encoding
  GENERATE_DWARF_PREDICATE_DEFS(),
  // Uniform registers use CUDA encoding - all 255 uniform registers
  GENERATE_DWARF_UNIFORM_DEFS(),
  // UR255 - uniform zero register
  DWARF_URZ =
      sass::GetDWARFEncodedRegister(REG_CLASS_UREG_FULL, sass::SASS_ZERO_REG),
  // Uniform predicate registers use CUDA encoding
  GENERATE_DWARF_UNIFORM_PREDICATE_DEFS()
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
  EXPAND_REGULAR_REGISTERS(EH_FRAME),
  EH_FRAME_RZ = 2500,                           // R255 - zero register
  EXPAND_PREDICATE_REGISTERS(EH_FRAME),         // Predicate registers
  EXPAND_UNIFORM_REGISTERS(EH_FRAME),           // Uniform registers
  EH_FRAME_URZ = 2501,                          // UR255 - uniform zero register
  EXPAND_UNIFORM_PREDICATE_REGISTERS(EH_FRAME), // Uniform predicate registers
};

static uint32_t g_gpr_regnums[] = {LLDB_PC, LLDB_ERROR_PC, LLDB_SP, LLDB_FP,
                                   LLDB_RA};
// All 255 R registers
static uint32_t g_regular_regnums[] = {EXPAND_REGULAR_REGISTERS(LLDB), LLDB_RZ};
// All 8 predicate registers
static uint32_t g_predicate_regnums[] = {EXPAND_PREDICATE_REGISTERS(LLDB)};
// All 255 uniform registers
static uint32_t g_uniform_regnums[] = {EXPAND_UNIFORM_REGISTERS(LLDB),
                                       LLDB_URZ};
// All 8 uniform predicate registers
static uint32_t g_uniform_predicate_regnums[] = {
    EXPAND_UNIFORM_PREDICATE_REGISTERS(LLDB)};

static const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     sizeof(g_gpr_regnums) / sizeof(g_gpr_regnums[0]), g_gpr_regnums},
    {"Regular Registers", "r", LLDB_RZ - LLDB_R0 + 1, g_regular_regnums},
    {"Predicate Registers", "p", kNumPRegs, g_predicate_regnums},
    {"Uniform Registers", "ur", LLDB_URZ - LLDB_UR0 + 1, g_uniform_regnums},
    {"Uniform Predicate Registers", "up", kNumUPRegs,
     g_uniform_predicate_regnums}};

/// Define all of the information about all registers. The register info structs
/// are accessed by the LLDB register numbers, which are defined above.
static const RegisterInfo g_reg_infos[LLDBRegNum::kNumRegs] = {
    {
        "PC",               // RegisterInfo::name
        nullptr,            // RegisterInfo::alt_name
        8,                  // RegisterInfo::byte_size
        REG_OFFSET(PC),     // RegisterInfo::byte_offset
        eEncodingUint,      // RegisterInfo::encoding
        eFormatAddressInfo, // RegisterInfo::format
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
        eFormatAddressInfo,  // RegisterInfo::format
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
        "RA",               // RegisterInfo::alt_name
        "R[20-21]",         // RegisterInfo::name
        8,                  // RegisterInfo::byte_size
        R_REG_OFFSET(20),   // RegisterInfo::byte_offset
        eEncodingUint,      // RegisterInfo::encoding
        eFormatAddressInfo, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_RA,            // RegisterInfo::kinds[eRegisterKindEHFrame]
            LLDB_INVALID_REGNUM,    // No DWARF number - RA spans R20-R21
            LLDB_REGNUM_GENERIC_RA, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_RA, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_RA  // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "SP",               // RegisterInfo::name
        "R[1]",             // RegisterInfo::alt_name
        4,                  // RegisterInfo::byte_size (32-bit)
        R_REG_OFFSET(1),    // RegisterInfo::byte_offset
        eEncodingUint,      // RegisterInfo::encoding
        eFormatAddressInfo, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_SP, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R1,    // RegisterInfo::kinds[eRegisterKindDWARF] - R1
            LLDB_REGNUM_GENERIC_SP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_SP, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    {
        "FP",               // RegisterInfo::name
        "R[2]",             // RegisterInfo::alt_name
        4,                  // RegisterInfo::byte_size (32-bit)
        R_REG_OFFSET(2),    // RegisterInfo::byte_offset
        eEncodingUint,      // RegisterInfo::encoding
        eFormatAddressInfo, // RegisterInfo::format
        {
            // RegisterInfo::kinds[]
            EH_FRAME_FP, // RegisterInfo::kinds[eRegisterKindEHFrame]
            DWARF_R2,    // RegisterInfo::kinds[eRegisterKindDWARF] - R2
            LLDB_REGNUM_GENERIC_FP, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_FP, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    // The number of elements in this list must match kNumRRegs (255).
    GENERATE_ALL_REGULAR_REGISTER_INFO(),
    // R255 - zero register
    {
        "RZ",                     // RegisterInfo::name
        "R255",                   // RegisterInfo::alt_name
        4,                        // RegisterInfo::byte_size
        REG_OFFSET(regular_zero), // RegisterInfo::byte_offset
        eEncodingUint,            // RegisterInfo::encoding
        eFormatHex,               // RegisterInfo::format
        {
            // RegisterInfo::kinds
            2500, // RegisterInfo::kinds[eRegisterKindEHFrame]
            sass::GetDWARFEncodedRegister(
                REG_CLASS_REG_FULL,
                sass::SASS_ZERO_REG), // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_RZ, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_RZ, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    // Predicate registers - all 8 predicate registers (P0-P7)
    GENERATE_ALL_PREDICATE_REGISTER_INFO(),
    // Uniform registers - all 255 uniform registers (UR0-UR254)
    GENERATE_ALL_UNIFORM_REGISTER_INFO(),
    // UR255 - uniform zero register
    {
        "URZ",                    // RegisterInfo::name
        "UR255",                  // RegisterInfo::alt_name
        4,                        // RegisterInfo::byte_size
        REG_OFFSET(uniform_zero), // RegisterInfo::byte_offset
        eEncodingUint,            // RegisterInfo::encoding
        eFormatHex,               // RegisterInfo::format
        {
            // RegisterInfo::kinds
            2501, // RegisterInfo::kinds[eRegisterKindEHFrame]
            sass::GetDWARFEncodedRegister(
                REG_CLASS_UREG_FULL,
                sass::SASS_ZERO_REG), // RegisterInfo::kinds[eRegisterKindDWARF]
            LLDB_INVALID_REGNUM, // RegisterInfo::kinds[eRegisterKindGeneric]
            LLDB_URZ, // RegisterInfo::kinds[eRegisterKindProcessPlugin]
            LLDB_URZ, // RegisterInfo::kinds[eRegisterKindLLDB]
        },
        nullptr, // RegisterInfo::value_regs
        nullptr, // RegisterInfo::invalidate_regs
        nullptr, // RegisterInfo::flags_type
    },
    // Uniform predicate registers - all 8 uniform predicate registers (UP0-UP7)
    GENERATE_ALL_UNIFORM_PREDICATE_REGISTER_INFO()};

RegisterContextNVIDIAGPU::RegisterContextNVIDIAGPU(ThreadNVIDIAGPU &thread)
    : NativeRegisterContext(thread) {}

void RegisterContextNVIDIAGPU::InvalidateAllRegisters() { m_regs.reset(); }

ThreadNVIDIAGPU &RegisterContextNVIDIAGPU::GetGPUThread() {
  return static_cast<ThreadNVIDIAGPU &>(GetThread());
}

CUDBGAPI RegisterContextNVIDIAGPU::GetDebuggerAPI() {
  return GetGPUThread().GetGPU().GetDebuggerAPI();
}

static void
ReadRegularRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                               const PhysicalCoords &physical_coords,
                               ThreadRegistersWithValidity &regs) {
  size_t num_regs = device_info.GetSMs()[physical_coords.sm_id]
                        .GetWarps()[physical_coords.warp_id]
                        .GetCurrentNumRegularRegisters();

  CUDBGResult res = api->readRegisterRange(
      physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
      physical_coords.thread_id, 0, num_regs, regs.val.regular);

  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVIDIAGPU::ReadAllRegsFromDevice(). "
                           "readRegisterRange failed: {}",
                           cudbgGetErrorString(res));

  for (size_t i = 0; i < num_regs; ++i)
    regs.is_valid.regular[i] = true;
}

static void
ReadPredicateRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                                 const PhysicalCoords &physical_coords,
                                 ThreadRegistersWithValidity &regs) {
  size_t num_regs = device_info.GetNumPredicateRegisters();
  if (num_regs == 0)
    return;

  CUDBGResult res = api->readPredicates(
      physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
      physical_coords.thread_id, num_regs, regs.val.predicate);

  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVIDIAGPU::ReadAllRegsFromDevice(). "
                           "readPredicates failed: {}",
                           cudbgGetErrorString(res));

  for (size_t i = 0; i < num_regs; ++i)
    regs.is_valid.predicate[i] = true;
}

static void
ReadUniformRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                               const PhysicalCoords &physical_coords,
                               ThreadRegistersWithValidity &regs) {
  size_t num_regs = device_info.GetNumUniformRegisters();
  if (num_regs == 0)
    return;

  CUDBGResult res = api->readUniformRegisterRange(
      physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id, 0,
      num_regs, regs.val.uniform);

  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVIDIAGPU::ReadAllRegsFromDevice(). "
                           "readUniformRegisterRange failed: {}",
                           cudbgGetErrorString(res));

  for (size_t i = 0; i < num_regs; ++i)
    regs.is_valid.uniform[i] = true;
}

static void
ReadUniformPredicateRegistersFromDevice(DeviceState &device_info, CUDBGAPI api,
                                        const PhysicalCoords &physical_coords,
                                        ThreadRegistersWithValidity &regs) {
  size_t num_regs = device_info.GetNumUniformPredicateRegisters();
  if (num_regs == 0)
    return;

  CUDBGResult res = api->readUniformPredicates(
      physical_coords.dev_id, physical_coords.sm_id, physical_coords.warp_id,
      num_regs, regs.val.uniform_predicate);

  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("RegisterContextNVIDIAGPU::ReadAllRegsFromDevice(). "
                           "readUniformPredicates failed: {}",
                           cudbgGetErrorString(res));

  for (size_t i = 0; i < num_regs; ++i) {
    regs.val.uniform_predicate[i] = regs.val.uniform_predicate[i] & 0x1;
    regs.is_valid.uniform_predicate[i] = true;
  }
}

const ThreadRegistersWithValidity &
RegisterContextNVIDIAGPU::ReadAllRegsFromDevice() {
  if (m_regs)
    return *m_regs;

  m_regs.emplace();
  ThreadRegistersWithValidity &regs = *m_regs;
  ThreadNVIDIAGPU &thread = GetGPUThread();
  const ThreadState *thread_state = thread.GetThreadState();

  if (!thread_state) {
    // We need to send always a PC to the client upon stop events, otherwise the
    // client will be in a borked state.
    regs.val.PC = 0;
    regs.is_valid.PC = true;
    return regs;
  }

  CUDBGAPI api = GetDebuggerAPI();
  const PhysicalCoords &physical_coords = thread_state->GetPhysicalCoords();

  {
    regs.val.PC = thread_state->GetPC();
    regs.is_valid.PC = true;
  }

  {
    if (const ExceptionInfo *exception = thread_state->GetException();
        exception && exception->errorPC.has_value()) {
      regs.val.errorPC = *exception->errorPC;
      regs.is_valid.errorPC = true;
    }
  }

    DeviceState &device_info =
        GetGPUThread().GetGPU().GetAllDevices()[physical_coords.dev_id];

    ReadRegularRegistersFromDevice(device_info, api, physical_coords, regs);
    ReadPredicateRegistersFromDevice(device_info, api, physical_coords, regs);
    ReadUniformRegistersFromDevice(device_info, api, physical_coords, regs);
    ReadUniformPredicateRegistersFromDevice(device_info, api, physical_coords, regs);

    {
      regs.val.regular_zero = 0;
      regs.is_valid.regular_zero = true;
    }

    {
      regs.val.uniform_zero = 0;
      regs.is_valid.uniform_zero = true;
    }

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
    reg_num = LLDB_R1;
  if (reg_num == LLDB_FP)
    reg_num = LLDB_R2;

  if (reg_num == LLDB_PC) {
    if (!regs.is_valid.PC)
      return Status("PC register is invalid");
  } else if (reg_num == LLDB_ERROR_PC) {
    if (!regs.is_valid.errorPC)
      return Status("errorPC register is invalid");
  } else if (reg_num == LLDB_RA) {
    if (!regs.is_valid.regular[sass::SASS_RA_REG_LO] ||
        !regs.is_valid.regular[sass::SASS_RA_REG_HI])
      return Status("RA register is invalid");
  } else if (reg_num >= static_cast<int>(kNumRegs)) {
    return Status::FromErrorStringWithFormatv("unknown register #{}", reg_num);
  } else if (int up_index = reg_num - LLDB_UP0; up_index >= 0) {
    if (!regs.is_valid.uniform_predicate[up_index])
      return Status::FromErrorStringWithFormatv("UP{} register is invalid",
                                                up_index);
  } else if (int p_index = reg_num - LLDB_P0; p_index >= 0) {
    if (!regs.is_valid.predicate[p_index])
      return Status::FromErrorStringWithFormatv("P{} register is invalid",
                                                p_index);
  } else if (int ur_index = reg_num - LLDB_UR0; ur_index >= 0) {
    if (!regs.is_valid.uniform[ur_index])
      return Status::FromErrorStringWithFormatv("UR{} register is invalid",
                                                ur_index);
  } else if (int r_index = reg_num - LLDB_R0; r_index >= 0) {
    if (!regs.is_valid.regular[r_index])
      return Status::FromErrorStringWithFormatv("R{} register is invalid",
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
    g_expedited_regs.insert(g_expedited_regs.end(),
                            {EXPAND_REGULAR_REGISTERS(LLDB)});
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
    regular[i] = false;
  regular_zero = false;
  for (size_t i = 0; i < kNumPRegs; ++i)
    predicate[i] = false;
  for (size_t i = 0; i < kNumURRegs; ++i)
    uniform[i] = false;
  uniform_zero = false;
  for (size_t i = 0; i < kNumUPRegs; ++i)
    uniform_predicate[i] = false;
}
