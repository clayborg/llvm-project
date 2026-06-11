//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformNVGPU.h"
#include "cudadebugger.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"

#include <climits>
#include <map>
#include <regex>

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

/// Represents the 3D index of a CUDA block or thread.
struct Dim3 {
  int x = 0;
  int y = 0;
  int z = 0;
};

/// Represents a CUDA thread's full coordinates.
struct CUDAThreadCoord {
  Dim3 block_idx;
  Dim3 thread_idx;
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;
};

/// Parse a CUDA thread name in the format:
/// "blockIdx(x=79 y=0 z=0) threadIdx(x=14 y=0 z=0)"
static bool ParseCUDAThreadName(llvm::StringRef name, CUDAThreadCoord &coord) {
  // Pattern: blockIdx(x=N y=N z=N) threadIdx(x=N y=N z=N)
  static std::regex pattern(
      R"(blockIdx\(x=(-?\d+)\s+y=(-?\d+)\s+z=(-?\d+)\)\s+)"
      R"(threadIdx\(x=(-?\d+)\s+y=(-?\d+)\s+z=(-?\d+)\))");

  std::string name_str = name.str();
  std::smatch match;
  if (!std::regex_search(name_str, match, pattern))
    return false;

  coord.block_idx.x = std::stoi(match[1].str());
  coord.block_idx.y = std::stoi(match[2].str());
  coord.block_idx.z = std::stoi(match[3].str());
  coord.thread_idx.x = std::stoi(match[4].str());
  coord.thread_idx.y = std::stoi(match[5].str());
  coord.thread_idx.z = std::stoi(match[6].str());
  return true;
}

/// Per-dimension value tracker for the aggregated thread-list path.
///
/// Stores every distinct value seen for a single CUDA coordinate dimension
/// (e.g. blockIdx.x) within a group, so we can decide between three display
/// forms: a single value, a contiguous range, or a wildcard when the values
/// are non-contiguous.
struct DimSet {
  int min_v = INT_MAX;
  int max_v = INT_MIN;
  llvm::DenseSet<int> values;

  void Insert(int v) {
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    values.insert(v);
  }
  bool IsEmpty() const { return values.empty(); }
  bool IsSingle() const { return values.size() == 1; }
  bool IsContiguous() const {
    return !IsEmpty() &&
           static_cast<size_t>(max_v - min_v + 1) == values.size();
  }
};

/// How a group's representative location was identified.
///
/// Resolution order: LineAndFunction first (most specific), then FunctionOnly
/// (e.g. function missing line info), then PCOnly (no symbol resolution at
/// all). Each tier produces a distinct group key and a distinct rendering.
/// FilteredOut is a synthetic tier used to collapse all threads dropped by a
/// list-time filter (e.g. --exceptions) into a single summary entry that is
/// rendered alongside the real groups.
///
/// Empty and Tombstone are sentinels reserved for llvm::DenseMapInfo<GroupKey>
/// and are never produced by BuildGroupKey.
enum class GroupKind {
  LineAndFunction,
  FunctionOnly,
  PCOnly,
  FilteredOut,
  Empty,
  Tombstone,
};

/// Identity of an aggregated thread group. Only fields relevant to the kind
/// are populated; the rest stay defaulted and are ignored by equality and
/// hashing (e.g. `pc` is meaningful only for PCOnly groups).
struct GroupKey {
  GroupKind kind = GroupKind::PCOnly;
  std::string file;
  uint32_t line = 0;
  std::string function;
  lldb::addr_t pc = LLDB_INVALID_ADDRESS;
  lldb::StopReason stop_reason = lldb::eStopReasonInvalid;
  uint64_t stop_value = 0;

  bool operator==(const GroupKey &other) const {
    if (kind != other.kind)
      return false;
    // Sentinel-like kinds (FilteredOut, plus DenseMap Empty/Tombstone)
    // carry no additional identity beyond the kind itself.
    if (kind == GroupKind::FilteredOut || kind == GroupKind::Empty ||
        kind == GroupKind::Tombstone)
      return true;
    if (stop_reason != other.stop_reason || stop_value != other.stop_value)
      return false;
    switch (kind) {
    case GroupKind::LineAndFunction:
      return line == other.line && file == other.file &&
             function == other.function;
    case GroupKind::FunctionOnly:
      return function == other.function;
    case GroupKind::PCOnly:
      return pc == other.pc;
    case GroupKind::FilteredOut:
    case GroupKind::Empty:
    case GroupKind::Tombstone:
      llvm_unreachable("sentinel kinds handled above");
    }
    return false;
  }
};

struct GroupKeyHash {
  size_t operator()(const GroupKey &k) const {
    switch (k.kind) {
    case GroupKind::LineAndFunction:
      return llvm::hash_combine(static_cast<int>(k.kind),
                                llvm::StringRef(k.file), k.line,
                                llvm::StringRef(k.function),
                                static_cast<int>(k.stop_reason), k.stop_value);
    case GroupKind::FunctionOnly:
      return llvm::hash_combine(static_cast<int>(k.kind),
                                llvm::StringRef(k.function),
                                static_cast<int>(k.stop_reason), k.stop_value);
    case GroupKind::PCOnly:
      return llvm::hash_combine(static_cast<int>(k.kind), k.pc,
                                static_cast<int>(k.stop_reason), k.stop_value);
    case GroupKind::FilteredOut:
    case GroupKind::Empty:
    case GroupKind::Tombstone:
      return llvm::hash_value(static_cast<int>(k.kind));
    }
    return 0;
  }
};

namespace llvm {
template <> struct DenseMapInfo<::GroupKey> {
  static ::GroupKey getEmptyKey() {
    ::GroupKey k;
    k.kind = ::GroupKind::Empty;
    return k;
  }
  static ::GroupKey getTombstoneKey() {
    ::GroupKey k;
    k.kind = ::GroupKind::Tombstone;
    return k;
  }
  static unsigned getHashValue(const ::GroupKey &k) {
    return static_cast<unsigned>(::GroupKeyHash{}(k));
  }
  static bool isEqual(const ::GroupKey &lhs, const ::GroupKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

/// Aggregated representation of all threads sharing one GroupKey.
struct ThreadGroup {
  GroupKey key;
  llvm::SmallVector<CUDAThreadCoord, 32> coords;
  ThreadSP representative_thread;
  /// SymbolContext that produced the group key for the representative thread,
  /// kept around so the renderer can pull display fields (module name, frame
  /// address) without re-resolving.
  SymbolContext representative_sc;
  /// Address used to resolve `representative_sc`. May be the per-warp errorPC.
  lldb::addr_t representative_address = LLDB_INVALID_ADDRESS;
  /// True when grouping is anchored on errorPC rather than the per-thread PC.
  bool used_error_pc = false;
  /// True when the user-selected thread belongs to this group; used to
  /// prefix the rendered row with "*".
  bool contains_selected = false;
  DimSet bx, by, bz, tx, ty, tz;
};

/// Format a DimSet as either name=v, name=[min...max], or name=*.
static void FormatDimSet(Stream &strm, const char *name, const DimSet &dim) {
  if (dim.IsEmpty()) {
    strm.Printf("%s=*", name);
    return;
  }
  if (dim.IsSingle()) {
    strm.Printf("%s=%d", name, dim.min_v);
    return;
  }
  if (dim.IsContiguous()) {
    strm.Printf("%s=[%d...%d]", name, dim.min_v, dim.max_v);
    return;
  }
  strm.Printf("%s=*", name);
}

/// Format an aggregated blockIdx/threadIdx triple using DimSet wildcards.
static void FormatDim3Set(Stream &strm, const char *prefix, const DimSet &x,
                          const DimSet &y, const DimSet &z) {
  strm.Printf("%s(", prefix);
  FormatDimSet(strm, "x", x);
  strm.PutChar(' ');
  FormatDimSet(strm, "y", y);
  strm.PutChar(' ');
  FormatDimSet(strm, "z", z);
  strm.PutChar(')');
}

/// Snapshot of the inputs we need to build groups, captured up front so we do
/// not hold the thread-list lock while iterating frames and symbols.
struct ThreadSnapshot {
  ThreadSP thread_sp;
  CUDAThreadCoord coord;
  lldb::addr_t pc = LLDB_INVALID_ADDRESS;
  /// Per-warp errorPC reported by the NVIDIA debugger backend, or
  /// LLDB_INVALID_ADDRESS when no valid errorPC is available. The backend
  /// reports 0 for warps that are not at a fault. When a warp hits an
  /// exception, the per-lane PC may have advanced past the actual fault site
  /// due to instruction slippage; errorPC pinpoints the real fault address and
  /// lets us aggregate threads from different warps that hit the same fault.
  lldb::addr_t error_pc = LLDB_INVALID_ADDRESS;
  lldb::StopReason stop_reason = lldb::eStopReasonInvalid;
  uint64_t stop_value = 0;
};

/// Accumulate one snapshot's coordinates into a group: per-dim DimSets and
/// the coords vector. The caller is responsible for setting `representative_*`
/// fields on first insertion.
static void AccumulateSnapshotIntoGroup(ThreadGroup &group,
                                        const ThreadSnapshot &snap) {
  group.coords.push_back(snap.coord);
  group.bx.Insert(snap.coord.block_idx.x);
  group.by.Insert(snap.coord.block_idx.y);
  group.bz.Insert(snap.coord.block_idx.z);
  group.tx.Insert(snap.coord.thread_idx.x);
  group.ty.Insert(snap.coord.thread_idx.y);
  group.tz.Insert(snap.coord.thread_idx.z);
}

/// Collect per-thread snapshots needed to build groups. Threads without CUDA
/// coordinates, register contexts, or PCs are dropped here so neither group
/// builder has to special-case them. List-time filters such as
/// --exceptions are applied later in the group-building phase by routing
/// non-matching threads to a synthetic FilteredOut group.
static llvm::SmallVector<ThreadSnapshot, 64>
CollectThreadSnapshots(Process &process, bool only_threads_with_stop_reason) {
  // Snapshot the list of threads under the list lock, then iterate them
  // outside of it. Looking each thread up by ID inside the loop would be
  // O(N) per call (linear scan + mutex op + UpdateThreadListIfNeeded check),
  // turning the whole collection step quadratic for tens of thousands of
  // GPU lanes.
  llvm::SmallVector<ThreadSP, 64> threads;
  {
    std::lock_guard<std::recursive_mutex> guard(
        process.GetThreadList().GetMutex());
    ThreadList &thread_list = process.GetThreadList();
    uint32_t num_threads = thread_list.GetSize();
    threads.reserve(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i)
      threads.push_back(thread_list.GetThreadAtIndex(i));
  }

  // The errorPC RegisterInfo is identical across every NVGPU thread because
  // they all share the SASS register layout. Look it up once on the first
  // valid register context and reuse the pointer for every subsequent
  // thread instead of a string scan per thread. nullopt = not yet looked up;
  // a populated nullptr = looked up but the register isn't exposed.
  std::optional<const RegisterInfo *> err_pc_info;

  llvm::SmallVector<ThreadSnapshot, 64> snapshots;
  snapshots.reserve(threads.size());

  for (const ThreadSP &thread_sp : threads) {
    if (!thread_sp)
      continue;

    StopInfoSP stop_info_sp = thread_sp->GetStopInfo();
    if (only_threads_with_stop_reason &&
        (!stop_info_sp || !stop_info_sp->ShouldShow()))
      continue;

    const char *name = thread_sp->GetName();
    if (!name)
      continue;

    CUDAThreadCoord coord;
    if (!ParseCUDAThreadName(name, coord))
      continue;
    coord.tid = thread_sp->GetID();

    RegisterContextSP reg_ctx_sp = thread_sp->GetRegisterContext();
    if (!reg_ctx_sp)
      continue;

    lldb::addr_t pc = reg_ctx_sp->GetPC();
    if (pc == LLDB_INVALID_ADDRESS)
      continue;

    if (!err_pc_info)
      err_pc_info = reg_ctx_sp->GetRegisterInfoByName("errorPC");

    ThreadSnapshot snap;
    snap.thread_sp = thread_sp;
    snap.coord = coord;
    snap.pc = pc;
    if (*err_pc_info) {
      RegisterValue val;
      if (reg_ctx_sp->ReadRegister(*err_pc_info, val)) {
        lldb::addr_t err_pc =
            val.GetAsUInt64(/*fail_value=*/LLDB_INVALID_ADDRESS);
        // The backend reports errorPC == 0 for warps that are not at a fault.
        // Treat that as "no errorPC" so those threads fall back to their
        // per-thread PC instead of all collapsing onto address 0x0.
        if (err_pc != 0)
          snap.error_pc = err_pc;
      }
    }
    if (stop_info_sp) {
      snap.stop_reason = stop_info_sp->GetStopReason();
      snap.stop_value = stop_info_sp->GetValue();
    }
    snapshots.push_back(std::move(snap));
  }
  return snapshots;
}

/// Resolution result for a thread's location: the SymbolContext we used and
/// the address it was resolved from (errorPC when available, otherwise the
/// thread's own PC).
struct ResolvedLocation {
  SymbolContext sc;
  lldb::addr_t address = LLDB_INVALID_ADDRESS;
  /// True when `address` is the per-warp errorPC instead of the per-thread PC.
  bool used_error_pc = false;
};

/// Per-call cache mapping a load address to its resolved SymbolContext. Used
/// to dedupe DWARF lookups across the (potentially tens of thousands of)
/// threads that share a small set of distinct PCs / errorPCs.
using LocationCache = llvm::DenseMap<lldb::addr_t, SymbolContext>;

/// Resolve `pc` to a SymbolContext, consulting `cache` first. Empty
/// SymbolContext on failure (matches existing behaviour).
static SymbolContext ResolveSymbolContextCached(lldb::addr_t pc, Target &target,
                                                LocationCache &cache) {
  auto it = cache.find(pc);
  if (it != cache.end())
    return it->second;
  SymbolContext sc;
  Address addr;
  if (addr.SetLoadAddress(pc, &target))
    addr.CalculateSymbolContext(&sc, eSymbolContextEverything);
  cache[pc] = sc;
  return sc;
}

/// Resolve a thread's location, preferring the per-warp errorPC when
/// available so that warps suffering from exception-induced PC slippage all
/// resolve to the same fault site. SymbolContext lookups are deduped across
/// threads via `cache`.
static ResolvedLocation ResolveLocation(const ThreadSnapshot &snap,
                                        Target &target, LocationCache &cache) {
  ResolvedLocation out;
  if (snap.error_pc != LLDB_INVALID_ADDRESS) {
    out.sc = ResolveSymbolContextCached(snap.error_pc, target, cache);
    out.address = snap.error_pc;
    out.used_error_pc = true;
    return out;
  }
  out.sc = ResolveSymbolContextCached(snap.pc, target, cache);
  out.address = snap.pc;
  return out;
}

/// Test whether a snapshot's stop reason matches a `--stop-reason` filter.
/// An unset filter matches everything; otherwise the snapshot's stop reason
/// must equal the filter exactly. This mirrors the per-thread filtering done
/// by `thread list -v` so both rendering paths agree.
static bool MatchesStopReasonFilter(const ThreadSnapshot &snap,
                                    std::optional<lldb::StopReason> filter) {
  if (!filter)
    return true;
  return snap.stop_reason == *filter;
}

/// Pick the most specific tier (line+function > function-only > pc-only) that
/// the resolved location supports and produce the corresponding GroupKey.
/// When `stop_reason_filter` is set and the snapshot does not match it,
/// returns a synthetic FilteredOut key so all such threads aggregate into a
/// single summary group.
static GroupKey
BuildGroupKey(const ThreadSnapshot &snap, const ResolvedLocation &loc,
              std::optional<lldb::StopReason> stop_reason_filter) {
  GroupKey key;
  if (!MatchesStopReasonFilter(snap, stop_reason_filter)) {
    key.kind = GroupKind::FilteredOut;
    return key;
  }

  key.stop_reason = snap.stop_reason;
  key.stop_value = snap.stop_value;

  ConstString fn = loc.sc.GetFunctionName(Mangled::ePreferDemangled);
  llvm::StringRef fn_ref = fn.GetStringRef();

  if (loc.sc.line_entry.IsValid() &&
      loc.sc.line_entry.line != LLDB_INVALID_LINE_NUMBER && !fn_ref.empty()) {
    key.kind = GroupKind::LineAndFunction;
    key.file = loc.sc.line_entry.GetFile().GetPath(/*denormalize=*/false);
    key.line = loc.sc.line_entry.line;
    key.function = fn_ref.str();
    return key;
  }

  if (!fn_ref.empty()) {
    key.kind = GroupKind::FunctionOnly;
    key.function = fn_ref.str();
    return key;
  }

  key.kind = GroupKind::PCOnly;
  key.pc = loc.address;
  return key;
}

/// Render one aggregated group: count, coordinates with wildcards, stop
/// reason on the first line, then a single indented location line whose
/// contents depend on the group's tier.
static void RenderAggregatedGroup(Stream &strm, const ThreadGroup &group) {
  strm.Indent();
  strm.Printf("%c %zu thread(s)", group.contains_selected ? '*' : ' ',
              group.coords.size());

  // For PC-only groups (no function name was resolvable), keep an
  // "at pc=0x..." prefix so users can still navigate to the address. When the
  // group was anchored on the per-warp errorPC (NVIDIA exception), label it
  // as such so it is not confused with the per-thread PC.
  if (group.key.kind == GroupKind::PCOnly) {
    strm.Printf(" at %s=0x%llx", group.used_error_pc ? "errorPC" : "pc",
                static_cast<unsigned long long>(group.key.pc));
  }
  strm.PutCString(": ");

  FormatDim3Set(strm, "blockIdx", group.bx, group.by, group.bz);
  strm.PutChar(' ');
  FormatDim3Set(strm, "threadIdx", group.tx, group.ty, group.tz);

  if (group.key.kind == GroupKind::FilteredOut) {
    strm.PutCString(", hidden by --stop-reason filter");
    strm.EOL();
    return;
  }

  if (group.key.stop_reason != lldb::eStopReasonInvalid) {
    if (StopInfoSP stop_info = group.representative_thread->GetStopInfo())
      strm.Printf(", stop reason = %s", stop_info->GetDescription());
  }
  strm.EOL();

  strm.IndentMore();
  strm.IndentMore();
  strm.Indent();

  const SymbolContext &sc = group.representative_sc;
  ConstString module_name;
  if (sc.module_sp)
    module_name = sc.module_sp->GetFileSpec().GetFilename();

  switch (group.key.kind) {
  case GroupKind::LineAndFunction: {
    if (module_name)
      strm.Printf("%s`", module_name.GetCString());
    // Display just the basename so the source location reads like the rest
    // of LLDB output (`module`function at file.cu:line`). The full path is
    // retained in the GroupKey for equality so that two files with the same
    // basename but different paths still hash to distinct groups.
    llvm::StringRef basename = llvm::sys::path::filename(group.key.file);
    strm.Printf("%s at %s:%u", group.key.function.c_str(),
                basename.str().c_str(), group.key.line);
    break;
  }
  case GroupKind::FunctionOnly:
    if (module_name)
      strm.Printf("%s`", module_name.GetCString());
    strm.PutCString(group.key.function);
    break;
  case GroupKind::FilteredOut:
  case GroupKind::Empty:
  case GroupKind::Tombstone:
    llvm_unreachable("sentinel kinds handled above or never rendered");
  case GroupKind::PCOnly: {
    // sc.target_sp is not populated by Address::CalculateSymbolContext, so
    // pull the target from the representative thread directly.
    Target &target = group.representative_thread->GetProcess()->GetTarget();
    Address resolved_addr;
    resolved_addr.SetLoadAddress(group.representative_address, &target);
    strm.Printf("0x%0*" PRIx64 " ",
                target.GetArchitecture().GetAddressByteSize() * 2,
                static_cast<uint64_t>(group.representative_address));
    StackFrameSP frame_sp =
        group.representative_thread->GetStackFrameAtIndex(0);
    ExecutionContext exe_ctx(frame_sp);
    sc.DumpStopContext(&strm, exe_ctx.GetBestExecutionContextScope(),
                       resolved_addr,
                       /*show_fullpaths=*/false,
                       /*show_module=*/true, /*show_inlined_frames=*/true,
                       /*show_function_arguments=*/false,
                       /*show_function_name=*/true);
    break;
  }
  }
  strm.EOL();
  strm.IndentLess();
  strm.IndentLess();
}

/// Build aggregated groups using the most specific available identity per
/// thread, then render them in first-seen order. The FilteredOut summary
/// group (when present) is moved to the end of the list so the summary line
/// always trails the real entries.
static size_t
RenderAggregated(Process &process, Stream &strm,
                 bool only_threads_with_stop_reason,
                 std::optional<lldb::StopReason> stop_reason_filter) {
  llvm::SmallVector<ThreadSnapshot, 64> snapshots =
      CollectThreadSnapshots(process, only_threads_with_stop_reason);
  if (snapshots.empty())
    return 0;

  lldb::tid_t selected_tid = LLDB_INVALID_THREAD_ID;
  if (ThreadSP selected_thread = process.GetThreadList().GetSelectedThread())
    selected_tid = selected_thread->GetID();

  // Map each GroupKey to an index into `groups`. Insertion order in `groups`
  // gives us deterministic output; the map is only used for O(1) lookup of
  // an existing group when a snapshot's key matches.
  llvm::DenseMap<GroupKey, size_t> key_to_index;
  llvm::SmallVector<ThreadGroup, 8> groups;

  Target &target = process.GetTarget();
  LocationCache location_cache;
  for (const ThreadSnapshot &snap : snapshots) {
    ResolvedLocation loc = ResolveLocation(snap, target, location_cache);
    GroupKey key = BuildGroupKey(snap, loc, stop_reason_filter);
    auto [it, inserted] =
        key_to_index.try_emplace(std::move(key), groups.size());
    if (inserted) {
      ThreadGroup g;
      g.key = it->first;
      g.representative_thread = snap.thread_sp;
      g.representative_sc = loc.sc;
      g.representative_address = loc.address;
      g.used_error_pc = loc.used_error_pc;
      groups.push_back(std::move(g));
    }
    ThreadGroup &group = groups[it->second];
    AccumulateSnapshotIntoGroup(group, snap);
    if (selected_tid != LLDB_INVALID_THREAD_ID &&
        snap.coord.tid == selected_tid)
      group.contains_selected = true;
  }

  // Move the FilteredOut summary group, if any, to the end of the list so the
  // summary line always trails the real entries. Relative order of the real
  // groups is preserved.
  ThreadGroup *filtered_it =
      std::find_if(groups.begin(), groups.end(), [](const ThreadGroup &g) {
        return g.key.kind == GroupKind::FilteredOut;
      });
  if (filtered_it != groups.end())
    std::rotate(filtered_it, filtered_it + 1, groups.end());

  process.GetStatus(strm);

  for (size_t i = 0; i < groups.size(); ++i) {
    if (i > 0)
      strm.EOL();
    RenderAggregatedGroup(strm, groups[i]);
  }
  return groups.size();
}

size_t PlatformNVGPU::GetGPUThreadStatus(
    Process &process, Stream &strm, bool only_threads_with_stop_reason,
    std::optional<lldb::StopReason> stop_reason_filter) {
  return RenderAggregated(process, strm, only_threads_with_stop_reason,
                          stop_reason_filter);
}

bool PlatformNVGPU::ParseGPUThreadName(llvm::StringRef name, GPUDim3 &block_idx,
                                       GPUDim3 &thread_idx) {
  // Pattern: blockIdx(x=N y=N z=N) threadIdx(x=N y=N z=N)
  static std::regex pattern(
      R"(blockIdx\(x=(-?\d+)\s+y=(-?\d+)\s+z=(-?\d+)\)\s+)"
      R"(threadIdx\(x=(-?\d+)\s+y=(-?\d+)\s+z=(-?\d+)\))");

  std::string name_str = name.str();
  std::smatch match;
  if (!std::regex_search(name_str, match, pattern))
    return false;

  block_idx.x = std::stoi(match[1].str());
  block_idx.y = std::stoi(match[2].str());
  block_idx.z = std::stoi(match[3].str());
  thread_idx.x = std::stoi(match[4].str());
  thread_idx.y = std::stoi(match[5].str());
  thread_idx.z = std::stoi(match[6].str());
  return true;
}

lldb::ThreadSP PlatformNVGPU::FindGPUThread(Process &process,
                                            const GPUDim3 &block_idx,
                                            const GPUDim3 &thread_idx) {
  ThreadList &thread_list = process.GetThreadList();
  std::lock_guard<std::recursive_mutex> guard(thread_list.GetMutex());

  uint32_t num_threads = thread_list.GetSize();
  for (uint32_t i = 0; i < num_threads; ++i) {
    ThreadSP thread_sp = thread_list.GetThreadAtIndex(i);
    if (!thread_sp)
      continue;

    const char *name = thread_sp->GetName();
    if (!name)
      continue;

    GPUDim3 actual_block_idx, actual_thread_idx;
    if (!ParseGPUThreadName(name, actual_block_idx, actual_thread_idx))
      continue;

    // Helper lambda to check if a coordinate matches.
    auto coordinate_matches = [](int actual_value,
                                 const std::optional<int> &pattern) -> bool {
      return !pattern.has_value() || actual_value == pattern.value();
    };

    // Check if this thread matches the pattern.
    if (coordinate_matches(actual_block_idx.x.value_or(0), block_idx.x) &&
        coordinate_matches(actual_block_idx.y.value_or(0), block_idx.y) &&
        coordinate_matches(actual_block_idx.z.value_or(0), block_idx.z) &&
        coordinate_matches(actual_thread_idx.x.value_or(0), thread_idx.x) &&
        coordinate_matches(actual_thread_idx.y.value_or(0), thread_idx.y) &&
        coordinate_matches(actual_thread_idx.z.value_or(0), thread_idx.z))
      return thread_sp;
  }

  return nullptr;
}
