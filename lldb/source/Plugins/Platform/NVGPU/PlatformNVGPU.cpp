//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformNVGPU.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"

#include "llvm/TargetParser/Triple.h"

#include "cudadebugger.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_NVGPU;

LLDB_PLUGIN_DEFINE(PlatformNVGPU)

namespace {
#define LLDB_PROPERTIES_platformnvgpuuser
#include "PlatformNVGPUUserProperties.inc"

enum {
#define LLDB_PROPERTIES_platformnvgpuuser
#include "PlatformNVGPUUserPropertiesEnum.inc"
};
} // namespace

PlatformNVGPU::PluginProperties::PluginProperties() {
  m_collection_sp = std::make_shared<OptionValueProperties>(
      PlatformNVGPU::GetPluginNameStatic(/*is_host=*/false));
  m_collection_sp->Initialize(g_platformnvgpuuser_properties);
}

FileSpec PlatformNVGPU::PluginProperties::GetNvdisasmPath() {
  return GetPropertyAtIndexAs<FileSpec>(ePropertyNvdisasmPath, {});
}

PlatformNVGPU::PluginProperties &PlatformNVGPU::GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

static uint32_t g_initialize_count = 0;

PlatformSP PlatformNVGPU::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (!create && arch)
    create = arch->GetTriple().isNVPTX();
  if (create)
    return PlatformSP(new PlatformNVGPU());
  return PlatformSP();
}

llvm::StringRef PlatformNVGPU::GetPluginDescriptionStatic(bool is_host) {
  return "NVGPU platform plug-in.";
}

void PlatformNVGPU::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformNVGPU::GetPluginNameStatic(false),
        PlatformNVGPU::GetPluginDescriptionStatic(false),
        PlatformNVGPU::CreateInstance, PlatformNVGPU::DebuggerInitialize);
  }
}

void PlatformNVGPU::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(
          debugger, GetPluginNameStatic(/*is_host=*/false))) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        "Properties for the NVGPU platform plugin.",
        /*is_global_property=*/true);
  }
}

void PlatformNVGPU::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformNVGPU::CreateInstance);

  Platform::Terminate();
}

PlatformNVGPU::PlatformNVGPU() : Platform(/*is_host=*/false) {
  m_supported_architectures = CreateArchList(
      {llvm::Triple::nvptx, llvm::Triple::nvptx64}, llvm::Triple::CUDA);
}

std::vector<ArchSpec>
PlatformNVGPU::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

void PlatformNVGPU::GetStatus(Stream &strm) { Platform::GetStatus(strm); }

void PlatformNVGPU::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformNVGPU::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                        ConstString name) {
  return {};
}

CompilerType PlatformNVGPU::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}

lldb::ProcessSP PlatformNVGPU::Attach(ProcessAttachInfo &attach_info,
                                      Debugger &debugger, Target *target,
                                      Status &error) {
  error = Status::FromErrorString("PlatformNVGPU::Attach() not implemented");
  return lldb::ProcessSP();
}

llvm::Error PlatformNVGPU::LocationToValue(RegisterContext *reg_ctx,
                                           lldb::RegisterKind reg_kind,
                                           uint32_t location, Value &value) {
  TargetSP target_sp = reg_ctx->CalculateTarget();
  if (!target_sp)
    return llvm::createStringError("missing register context");

  size_t length = sizeof(uint32_t);
  value.SetValueType(Value::ValueType::Scalar);

  uint32_t offset = location & 0x00FFFFFF;
  uint32_t location_class = location >> 24;
  switch (location_class) {
  case REG_CLASS_REG_PRED:
  case REG_CLASS_REG_FULL:
  case REG_CLASS_REG_HALF:
  case REG_CLASS_UREG_PRED:
  case REG_CLASS_UREG_FULL:
  case REG_CLASS_UREG_HALF: {
    bool half = false;
    if (location_class == REG_CLASS_REG_HALF) {
      location = REG_CLASS_REG_FULL << 24 | offset;
      half = true;
    } else if (location_class == REG_CLASS_UREG_HALF) {
      location = REG_CLASS_UREG_FULL << 24 | offset;
      half = true;
    }

    RegisterValue reg_value;
    llvm::Error error = reg_ctx->ReadRegister(reg_kind, location, reg_value);

    if (error) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "failed to read register");
    }

    if (!reg_value.GetScalarValue(value.GetScalar())) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to get scalar value from register");
    }

    if (half) {
      if (value.GetScalar().ExtractBitfield(length * 8, 0)) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register bitfield extraction failed");
      }
    } else {
      const RegisterInfo *reg_info =
          reg_ctx->GetRegisterInfo(reg_kind, location);
      if (reg_info)
        value.SetContext(Value::ContextType::RegisterInfo,
                         const_cast<RegisterInfo *>(reg_info));
    }
    break;
  }
  case REG_CLASS_LMEM_REG_OFFSET:
  case REG_CLASS_MEM_LOCAL: {
    lldb::addr_t value_addr = offset;

    if (location_class == REG_CLASS_LMEM_REG_OFFSET) {
      lldb::StackFrameSP frame_sp = reg_ctx->CalculateStackFrame();
      if (!frame_sp) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "failed to calculate stack frame");
      }
      value_addr = frame_sp->GetStackID().GetCallFrameAddressWithoutMetadata() + offset;
    }

    ThreadSP thread_sp = reg_ctx->GetThread().shared_from_this();
    ProcessSP process_sp(thread_sp->GetProcess());

    if (ABI *abi = process_sp->GetABI().get()) {
      ExecutionContext exe_ctx;
      reg_ctx->CalculateExecutionContext(exe_ctx);
      value = Scalar(value_addr);
      value.SetValueType(Value::ValueType::LoadAddress);
      value.SetAddressSpace(abi->GetDefaultStackAddressSpace(), &exe_ctx);
    }
    break;
  }
  case REG_CLASS_INVALID: {
    value.ResizeData(length);
    // Note that "0" is not a correct value for the unknown bits.
    // It would be better to also return a mask of valid bits together
    // with the expression result, so the debugger can print missing
    // members as "<optimized out>" or something.
    ::memset(value.GetBuffer().GetBytes(), 0, length);
    break;
  }
  }

  return llvm::Error::success();
}

std::optional<llvm::Error>
PlatformNVGPU::ReadVirtualRegister(RegisterContext *reg_ctx,
                                   lldb::RegisterKind reg_kind,
                                   lldb::regnum64_t reg_num, Value &value) {
  Log *log = GetLog(LLDBLog::Modules);
  LLDB_LOG(log, "ReadVirtualRegister: reg_kind={0}, reg_num={1}", reg_kind,
           reg_num);
  lldb::StackFrameSP frame_sp = reg_ctx->CalculateStackFrame();
  uint64_t locations =
      FindRegisterLocations(frame_sp->GetFrameCodeAddress().GetModule(),
                            frame_sp->GetStackID().GetPC(), reg_num);
  if (locations == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register location not found");
  }

  Value low_half_value;
  llvm::Error error = LocationToValue(reg_ctx, reg_kind, locations & 0xFFFFFFFF,
                                      low_half_value);

  if (error)
    return std::move(error);

  // If there is more than one location, we need to form a composite
  uint32_t top_location = locations >> 32;
  if (top_location == 0) {
    value = std::move(low_half_value);
    return llvm::Error::success();
  }

  value.AppendDataToHostBuffer(low_half_value);
  Value hi_half_value;
  error = LocationToValue(reg_ctx, reg_kind, top_location, hi_half_value);

  if (error)
    return std::move(error);

  value.AppendDataToHostBuffer(hi_half_value);
  return llvm::Error::success();
}

///   The PTX to SASS register map table is made of a series of entries,
///   one per function. Each function entry is made of a list of register
///   mappings, from a PTX register to a SASS register. The table size is
///   saved in the first 32 bits.
///
///   | fct name | number of entries |
///   | idx | ptx_reg | sass_reg | start | end |
///   | idx | ptx_reg | sass_reg | start | end |
///   ...
///   | idx | ptx_reg | sass_reg | start | end |
///   | fct name | number of entries |
///   | idx | ptx_reg | sass_reg | start | end |
///   ...
///   ...
///
///   A PTX reg is mapped to one more SASS registers. If a PTX register
///   is mapped to more than one SASS register, multiple entries are
///   required and the 'idx' field is incremented by 1 for each one of
///   them. The 'start' and 'end' addresses indicate the physical address
///   between which the mapping is valid.
///
///   The 8 high bits of a sass_reg are the register class (see cudadebugger.h).
///   The low 24 bits are either the register index, or the offset in local
///   memory, or the stack pointer register index and the offset.
///
void PlatformNVGPU::RecordLoadedModule(const lldb::ModuleSP &module_sp,
                                       Target &target) {
  Log *log = GetLog(LLDBLog::Modules);
  std::string module_name = module_sp->GetSpecificationDescription();
  if (m_entries.find(module_sp) != m_entries.end()) {
    LLDB_LOG(log, "RecordLoadedModule: module {0} already loaded", module_name);
    return;
  }

  ObjectFile *obj_file = module_sp->GetObjectFile();
  if (!obj_file) {
    LLDB_LOG(log, "RecordLoadedModule: no object file for module {0}",
             module_name);
    return;
  }

  SectionList *sections = obj_file->GetSectionList();
  if (!sections) {
    LLDB_LOG(log, "RecordLoadedModule: no section list for module {0}",
             module_name);
    return;
  }
  // Find .nv_debug_info_reg_sass section
  ConstString section_name(".nv_debug_info_reg_sass");
  SectionSP section_sp = sections->FindSectionByName(section_name);
  if (!section_sp) {
    LLDB_LOG(log,
             "RecordLoadedModule: .nv_debug_info_reg_sass section not "
             "found in module {0}",
             module_name);
    return;
  }

  // Read section data
  DataExtractor data;
  if (!obj_file->ReadSectionData(section_sp.get(), data)) {
    LLDB_LOG(log,
             "RecordLoadedModule: failed to read section data from module {0}",
             module_name);
    return;
  }

  lldb::offset_t offset = 0;

  // Read header
  if (!data.ValidOffsetForDataOfSize(offset, 8)) {
    LLDB_LOG(log, "RecordLoadedModule: section too small for header");
    return;
  }

  const char *function_name = data.GetCStr(&offset);
  uint32_t num_entries = data.GetU32(&offset);

  LLDB_LOG(log, "RecordLoadedModule: function={0}, num_entries={1}",
           function_name, num_entries);

  // Find the function loaded start and end addresses in the module
  lldb::addr_t func_start = 0;
  lldb::addr_t func_end = 0;
  SymbolContextList sc_list;
  module_sp->FindFunctions(RegularExpression(function_name),
                           ModuleFunctionSearchOptions(), sc_list);
  uint32_t i = 0;
  for (; i < sc_list.GetSize(); ++i) {
    SymbolContext sc;
    sc_list.GetContextAtIndex(i, sc);

    if (sc.function && sc.function->GetAddressRanges().size() == 1) {
      AddressRange func_range = sc.function->GetAddressRanges()[0];
      func_start = func_range.GetBaseAddress().GetLoadAddress(&target);
      func_end = func_start + func_range.GetByteSize();
      LLDB_LOG(log, "Function %s: [0x%" PRIx64 " - 0x%" PRIx64 ")\n",
               function_name, func_start, func_end);
      break;
    }
  }

  if (i == sc_list.GetSize()) {
    LLDB_LOG(log, "Function %s symbol not found.", function_name);
    return;
  }

  PTXPRegMap &ptx_reg_map = m_entries[module_sp];
  ptx_reg_map.clear();

  // Parse each entry, but we don't support overlapping code ranges for
  // the same PTX register with the same index. If such cases are found,
  // the last entrywill overwrite the previous one.
  for (uint32_t i = 0; i < num_entries; ++i) {
    // Read PTX location index
    if (!data.ValidOffsetForDataOfSize(offset, 4)) {
      LLDB_LOG(log, "RecordLoadedModule: truncated entry {0} index", i);
      return;
    }

    uint32_t idx = data.GetU32(&offset);
    // We only support up to 64-bit PTX registers
    if (idx > 1) {
      LLDB_LOG(log, "RecordLoadedModule: malformed entry {0} with index {1}", i,
               idx);
      return;
    }

    std::string reg_name = std::string(data.GetCStr(&offset));

    if (reg_name.size() > sizeof(uint64_t)) {
      LLDB_LOG(log,
               "RecordLoadedModule: at entry {0} register name {1} too long", i,
               idx);

      data.GetU32(&offset);
      data.GetU32(&offset);
      data.GetU32(&offset);
      continue;
    }

    // Get register ID from reg_name and skipp the '\0' at the end
    uint64_t reg_num = 0;
    for (uint32_t j = 0; j < reg_name.size(); j++) {
      reg_num <<= 8;
      reg_num |= reg_name[j];
    }

    if (!data.ValidOffsetForDataOfSize(offset, 4)) {
      LLDB_LOG(log,
               "RecordLoadedModule: truncated entry {0} location at index {1}",
               i, idx);
      return;
    }

    uint32_t location = data.GetU32(&offset);

    // Read PC range
    if (!data.ValidOffsetForDataOfSize(offset, 16)) {
      LLDB_LOG(log, "RecordLoadedModule: truncated entry {0} PC range", i);
      return;
    }

    lldb::addr_t pc_start = func_start + data.GetU32(&offset);
    lldb::addr_t pc_end = func_start + data.GetU32(&offset);
    lldb::addr_t pc_extended_end = func_end;

    auto map_iter = ptx_reg_map.find(reg_num);
    if (map_iter == ptx_reg_map.end()) {
      PTXPieceToSassEntry entry;
      entry.pc_start = pc_start;
      entry.pc_end = pc_end;
      entry.pc_extended_end = pc_extended_end;
      entry.reg_name = reg_name;
      entry.locations |= ((uint64_t)location) << (idx * 32);
      ptx_reg_map[reg_num].push_back(entry);
      continue;
    }

    std::list<PTXPieceToSassEntry>::iterator iter = map_iter->second.begin();
    std::list<PTXPieceToSassEntry>::iterator prev = map_iter->second.end();

    for (; iter != map_iter->second.end(); prev = iter, iter++) {
      // The list is ordered from low address to high, so this is not the
      // correct place.
      if (pc_start >= iter->pc_end)
        continue;

      // New entry has some overlap with the current range, so we need to
      // split the existing range and insert a new entry.
      if (pc_start > iter->pc_start) {
        PTXPieceToSassEntry entry;
        entry.pc_start = iter->pc_start;
        entry.pc_end = pc_start;
        entry.pc_extended_end = pc_start;
        entry.reg_name = reg_name;
        entry.locations = iter->locations;
        iter->pc_start = pc_start;
        iter = ptx_reg_map[reg_num].insert(iter, entry);
        continue;
      }

      // We found the same range start but the the end could still
      // be different.
      if (pc_start == iter->pc_start) {
        // If the end of the range is the same, then just update the existing
        // location. This is valid because we don't support PTX embeded
        // overalpping locations.
        if (pc_end == iter->pc_end) {
          iter->locations |= ((uint64_t)location) << (idx * 32);
          break;
        }
        // If the end of the new range is later, split the range into two
        // ranges.
        if (pc_end > iter->pc_end) {
          iter->locations |= ((uint64_t)location) << (idx * 32);
          pc_start = iter->pc_end;
          continue;
        }

        // If the new range is shorter, update the current range and insert
        // new element before it.
        PTXPieceToSassEntry entry = (*iter);
        iter->pc_start = pc_end;
        entry.pc_end = pc_end;
        entry.pc_extended_end = pc_end;
        entry.reg_name = reg_name;
        entry.locations |= ((uint64_t)location) << (idx * 32);
        map_iter->second.insert(iter, entry);
        break;
        // New range start comes before the current range start.
      }

      // New range comes before the current range in the list, but might
      // overlap with it.
      PTXPieceToSassEntry entry;
      entry.pc_start = pc_start;
      entry.pc_end = pc_end;
      entry.reg_name = reg_name;
      entry.locations |= ((uint64_t)location) << (idx * 32);
      entry.pc_extended_end = pc_extended_end;

      // Check if the next range belong to the same function.
      if (func_start <= iter->pc_start && func_end >= iter->pc_end) {
        entry.pc_extended_end = iter->pc_start;
      }

      // Correct extended range of the previous element.
      if (prev != map_iter->second.end() && func_start <= prev->pc_start &&
          func_end > prev->pc_end) {
        prev->pc_extended_end = pc_start;
      }

      // Need to split the range and continue.
      if (pc_end > iter->pc_start) {
        entry.pc_end = iter->pc_start;
        pc_start = iter->pc_start;
        iter = map_iter->second.insert(iter, entry);
        continue;
      }

      // No overlap, so we can insert the new range at the current position.
      map_iter->second.insert(iter, entry);
      break;
    }

    // New range needs to be inserted at the end of the list.
    if (iter == map_iter->second.end()) {
      PTXPieceToSassEntry entry;
      entry.pc_start = pc_start;
      entry.pc_end = pc_end;
      entry.pc_extended_end = pc_extended_end;
      entry.reg_name = reg_name;
      entry.locations |= ((uint64_t)location) << (idx * 32);
      map_iter->second.push_back(entry);
    }
  }

  return;
}

uint64_t PlatformNVGPU::FindRegisterLocations(const lldb::ModuleSP &module_sp,
                                              lldb::addr_t pc,
                                              uint64_t reg_num) {
  Log *log = GetLog(LLDBLog::Modules);
  std::string module_name = module_sp->GetSpecificationDescription();
  if (m_entries.find(module_sp) == m_entries.end()) {
    LLDB_LOG(log, "RecordLoadedModule: module {0} not found", module_name);
    return 0;
  }

  PTXPRegMap &ptx_reg_map = m_entries[module_sp];
  auto map_iter = ptx_reg_map.find(reg_num);
  if (map_iter == ptx_reg_map.end()) {
    LLDB_LOG(log, "RecordLoadedModule: PTX register mapping not found in the module {0}",
             module_name);
    return 0;
  }

  for (auto &entry : map_iter->second) {
    if (pc >= entry.pc_start && pc < entry.pc_end) {
      return entry.locations;
    }
  }

  LLDB_LOG(log, "RecordLoadedModule: PTX register location not found");
  return 0;
}
