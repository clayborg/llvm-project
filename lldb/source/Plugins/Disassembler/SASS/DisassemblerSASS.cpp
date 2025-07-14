//===-- DisassemblerSASS.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DisassemblerSASS.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/TildeExpressionResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <vector>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(DisassemblerSASS)

template <typename T, typename E>
constexpr bool is_a_substring_of_any(const E &element,
                                     std::initializer_list<T> set) {
  for (const T &v : set)
    if (element.find(v) != std::string::npos)
      return true;
  return false;
}

static bool
checkBooleanAttribute(const std::map<std::string, std::string> &attributes,
                      const std::string &key) {
  auto it = attributes.find(key);
  if (it == attributes.end())
    return false;
  return llvm::is_contained({"True", "true"}, it->second);
}

static std::string colorizeString(const std::string &text,
                                  llvm::raw_ostream::Colors color) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream.enable_colors(true);
  stream.changeColor(color);
  stream << text;
  stream.resetColor();
  return stream.str();
}

/// Per-module SM architecture cache with its own mutex
/// This allows concurrent SM extraction for different modules
class ModuleSM {
public:
  explicit ModuleSM(const std::string &cache_key) : m_cache_key(cache_key) {}

  /// Thread-safe SM architecture extraction for this specific module
  llvm::Expected<std::string> findSM(const Address &base_addr);

private:
  std::string m_cache_key;
  std::optional<std::string> m_cached_sm_arch;
  std::mutex m_mutex;

  // Delete copy constructor and assignment operator
  ModuleSM(const ModuleSM &) = delete;
  ModuleSM &operator=(const ModuleSM &) = delete;
};

/// Singleton class to manage caches for DisassemblerSASS
/// This replaces static variables to follow LLVM best practices
class DisassemblerSASSCache {
public:
  static DisassemblerSASSCache &getInstance() {
    static DisassemblerSASSCache instance;
    return instance;
  }

  // nvdisasm path caching
  std::optional<FileSpec> cached_nvdisasm_path;
  std::once_flag search_once_nvdisasm;

  // Per-module SM architecture caching
  std::shared_ptr<ModuleSM> getModuleSM(const std::string &cache_key);

private:
  DisassemblerSASSCache() = default;
  ~DisassemblerSASSCache() = default;

  // Map of module cache keys to ModuleSM instances
  std::map<std::string, std::shared_ptr<ModuleSM>> m_module_sm_map;
  std::mutex m_module_map_mutex;

  // Delete copy constructor and assignment operator
  DisassemblerSASSCache(const DisassemblerSASSCache &) = delete;
  DisassemblerSASSCache &operator=(const DisassemblerSASSCache &) = delete;
};

/// Implementation of DisassemblerSASSCache::getModuleSM
std::shared_ptr<ModuleSM>
DisassemblerSASSCache::getModuleSM(const std::string &cache_key) {
  std::lock_guard<std::mutex> lock(m_module_map_mutex);

  auto it = m_module_sm_map.find(cache_key);
  if (it != m_module_sm_map.end()) {
    return it->second;
  }

  // Create new ModuleSM instance for this module
  auto module_sm = std::make_shared<ModuleSM>(cache_key);
  m_module_sm_map[cache_key] = module_sm;
  return module_sm;
}

/// Implementation of ModuleSM::findSM
llvm::Expected<std::string> ModuleSM::findSM(const Address &base_addr) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // Check if we already have the cached result
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_cached_sm_arch.has_value()) {
      LLDB_LOG(log, "ModuleSM::findSM: Using cached SM architecture: {0}",
               *m_cached_sm_arch);
      return *m_cached_sm_arch;
    }
  }

  // Get the module from the address
  lldb::ModuleSP module_sp = base_addr.GetModule();
  if (!module_sp)
    return llvm::createStringError("No module found for address");

  // Get the object file
  ObjectFile *obj_file = module_sp->GetObjectFile();
  if (!obj_file)
    return llvm::createStringError("No object file found in module");

  // Check ELF header ABI version - the CUDA info layout is only valid for ABI
  // version 8 We need to verify this is an ELF file with the correct ABI
  // version
  if (obj_file->GetPluginName() != "elf")
    return llvm::createStringError("Object file is not ELF format");

  // For ELF files, we need to check the ABI version in the ELF header
  // The CUDA note structure layout is specifically for ABI version 8
  DataExtractor header_data;
  lldb::offset_t hdr_offset = 0;
  if (obj_file->GetData(0, 64, header_data) < 16)
    return llvm::createStringError("Failed to read ELF header");

  // Check ELF magic and get ABI version (at offset 8 in e_ident)
  hdr_offset = 0;
  if (header_data.GetU32(&hdr_offset) !=
      0x464C457F) { // ELF magic in little endian
    return llvm::createStringError("Invalid ELF magic bytes");
  }

  hdr_offset = 8; // EI_ABIVERSION offset
  uint8_t abi_version = header_data.GetU8(&hdr_offset);
  if (abi_version != 8) {
    return llvm::createStringError(
        "ELF ABI version %d != 8, CUDA note layout requires ABI version 8",
        abi_version);
  }

  LLDB_LOG(log, "ModuleSM::findSM: ELF ABI version 8 confirmed");

  // Find the .note.nv.cuinfo section
  SectionList *section_list = module_sp->GetSectionList();
  if (!section_list)
    return llvm::createStringError("No section list found in module");

  SectionSP cuinfo_section_sp =
      section_list->FindSectionByName(ConstString(".note.nv.cuinfo"));
  if (!cuinfo_section_sp)
    return llvm::createStringError(".note.nv.cuinfo section not found");

  // Read the section data
  DataExtractor section_data;
  if (obj_file->ReadSectionData(cuinfo_section_sp.get(), section_data) == 0)
    return llvm::createStringError(
        "Failed to read .note.nv.cuinfo section data");

  // Parse the ELF note header
  lldb::offset_t offset = 0;

  // ELF note header structure
  struct {
    uint32_t nameSize;
    uint32_t descSize;
    uint32_t noteType;
  } note_header;

  // Read note header
  note_header.nameSize = section_data.GetU32(&offset);
  note_header.descSize = section_data.GetU32(&offset);
  note_header.noteType = section_data.GetU32(&offset);

  LLDB_LOG(log,
           "ModuleSM::findSM: Note header - nameSize={0}, descSize={1}, "
           "noteType={2}",
           note_header.nameSize, note_header.descSize, note_header.noteType);

  // Read the name (should be "NVIDIA CUDA" or similar)
  std::string note_name;
  for (uint32_t i = 0;
       i < note_header.nameSize && offset < section_data.GetByteSize(); ++i) {
    char c = section_data.GetU8(&offset);
    if (c != 0)
      note_name += c;
  }

  // Align to 4-byte boundary after name
  while (offset % 4 != 0 && offset < section_data.GetByteSize())
    offset++;

  LLDB_LOG(log, "ModuleSM::findSM: Note name: '{0}'", note_name);

  // Check if we have enough data for the CUDA info structure
  if (note_header.descSize < 6 || offset + 6 > section_data.GetByteSize())
    return llvm::createStringError("Insufficient data for CUDA info structure");

  // Read CUDA-specific fields
  uint16_t noteVersion = section_data.GetU16(&offset);
  uint16_t cudaVirtSm = section_data.GetU16(&offset);
  uint16_t cudaToolKitVersion = section_data.GetU16(&offset);

  LLDB_LOG(log,
           "ModuleSM::findSM: noteVersion={0}, cudaVirtSm={1}, "
           "cudaToolKitVersion={2}",
           noteVersion, cudaVirtSm, cudaToolKitVersion);

  // Ensure this is note version 2
  if (noteVersion != 2)
    return llvm::createStringError("Unsupported note version %d, expected 2",
                                   noteVersion);

  // Convert cudaVirtSm to SM architecture string
  if (cudaVirtSm > 0) {
    LLDB_LOG(log, "ModuleSM::findSM: Extracted SM architecture: SM{0}",
             cudaVirtSm);

    // Cache the successful result
    std::string sm_arch = llvm::formatv("SM{0}", cudaVirtSm).str();
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_cached_sm_arch = sm_arch;
    }

    return sm_arch;
  }

  return llvm::createStringError("Invalid cudaVirtSm value %d", cudaVirtSm);
}

// SASS Instruction implementation with nvdisasm JSON schema support
// Provides accurate instruction classification using schema attributes when
// available, with comprehensive fallback pattern implementation for maximum
// compatibility across different nvdisasm versions and CUDA architectures.
class InstructionSASS : public lldb_private::Instruction {
public:
  InstructionSASS(DisassemblerSASS &disasm,
                  const lldb_private::Address &address,
                  const std::string &opcode, const std::string &operands,
                  AddressClass addr_class)
      : InstructionSASS(disasm, address, opcode, operands, "", "", {}, {},
                        addr_class) {}

  InstructionSASS(DisassemblerSASS &disasm,
                  const lldb_private::Address &address,
                  const std::string &opcode, const std::string &operands,
                  const std::string &predicate, const std::string &extra,
                  const std::map<std::string, std::string> &other_attributes,
                  const std::vector<std::string> &other_flags,
                  AddressClass addr_class)
      : Instruction(address, addr_class),
        m_disasm_wp(std::static_pointer_cast<DisassemblerSASS>(
            disasm.shared_from_this())),
        m_predicate(predicate), m_extra(extra),
        m_other_attributes(other_attributes), m_other_flags(other_flags) {

    // Cache attribute lookups to avoid repeated map searches
    m_is_control_flow = checkBooleanAttribute(other_attributes, "control-flow");
    m_is_subroutine_call =
        checkBooleanAttribute(other_attributes, "subroutine-call");
    m_is_barrier = checkBooleanAttribute(other_attributes, "barrier");
    m_is_load = checkBooleanAttribute(other_attributes, "load");

    // Set the protected members from the parent class directly
    m_opcode_name = opcode;
    m_mnemonics = operands;
    m_comment = extra; // Use extra field as comment

    // Set markup members with colors
    m_markup_opcode_name =
        colorizeString(opcode, llvm::raw_ostream::Colors::GREEN);
    m_markup_mnemonics =
        colorizeString(" " + operands, llvm::raw_ostream::Colors::CYAN);

    // Mark that we've already calculated the strings
    m_calculated_strings = true;
  }

  ~InstructionSASS() override = default;

  bool DoesBranch() override {
    // Use cached schema attributes if available. Otherwise, fallback to pattern
    // matching.
    if (m_is_control_flow.has_value())
      return *m_is_control_flow;

    // Comprehensive branch detection based on CUDA instruction patterns
    // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

    // Turing+ control flow instructions
    // BRA - Branch
    // BRX - Branch indirect
    // JMP - Jump
    // JMX - Jump indirect
    // RET - Return
    // BRK - Break
    // CONT - Continue
    // SSY - Set Synchronization Point
    // BPT - Breakpoint
    // EXIT - Thread exit
    // SYNC - Synchronize
    // BREAK - Break
    // KILL - Kill thread
    // NANOSLEEP - Nanosleep
    // RTT - Return to top
    // WARPSYNC - Warp sync
    // YIELD - Yield
    // BMOV - Branch move
    // RPCMOV - RPC move
    // ACQBULK - Acquire bulk
    // ENDCOLLECTIVE - End collective
    if (is_a_substring_of_any(
            m_opcode_name,
            {"BRA",          "BRX",   "JMP",  "JMX",       "RET",
             "BRK",          "CONT",  "SSY",  "BPT",       "EXIT",
             "SYNC",         "BREAK", "KILL", "NANOSLEEP", "RTT",
             "WARPSYNC",     "YIELD", "BMOV", "RPCMOV",    "ACQBULK",
             "ENDCOLLECTIVE"})) {
      m_is_control_flow = true;
    } else {
      m_is_control_flow = false;
    }

    return *m_is_control_flow;
  }

  bool HasDelaySlot() override {
    // SASS doesn't have delay slots
    return false;
  }

  bool IsLoad() override {
    // Use cached schema attributes if available. Otherwise, fallback to pattern
    // matching.
    if (m_is_load.has_value())
      return *m_is_load;

    // Comprehensive load detection based on CUDA instruction patterns
    // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

    // Standard load instructions
    // LD - Load
    // LDU - Load uniform
    // LDC - Load constant
    // LDS - Load shared
    // LDG - Load global
    // LDL - Load local
    if (is_a_substring_of_any(m_opcode_name,
                              {"LD", "LDU", "LDC", "LDS", "LDG", "LDL"})) {
      m_is_load = true;
    } else {
      m_is_load = false;
    }

    return *m_is_load;
  }

  bool IsAuthenticated() override {
    // SASS doesn't have authenticated instructions
    return false;
  }

  bool IsCall() override {
    // Use cached schema attributes if available. Otherwise, fallback to pattern
    // matching.
    if (m_is_subroutine_call.has_value())
      return *m_is_subroutine_call;

    // Comprehensive call detection based on CUDA instruction patterns
    // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

    // CALL - Call subroutine
    if (is_a_substring_of_any(m_opcode_name, {"CALL"}))
      m_is_subroutine_call = true;
    else
      m_is_subroutine_call = false;

    return *m_is_subroutine_call;
  }

  bool IsBarrier() {
    // Use cached schema attributes if available. Otherwise, fallback to pattern
    // matching.
    if (m_is_barrier.has_value())
      return *m_is_barrier;

    // Comprehensive barrier detection based on CUDA instruction patterns
    // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

    // MEMBAR, DEPBAR, UCGABAR_* - all covered with BAR pattern
    // BAR - Memory barrier (covers MEMBAR, DEPBAR, UCGABAR_*)
    if (is_a_substring_of_any(m_opcode_name, {"BAR"}))
      m_is_barrier = true;
    else
      m_is_barrier = false;

    return *m_is_barrier;
  }

  const std::string &GetPredicate() const { return m_predicate; }
  const std::string &GetExtra() const { return m_extra; }
  const std::map<std::string, std::string> &GetOtherAttributes() const {
    return m_other_attributes;
  }
  const std::vector<std::string> &GetOtherFlags() const {
    return m_other_flags;
  }

  void CalculateMnemonicOperandsAndComment(
      const ExecutionContext *exe_ctx) override {
    // Already calculated in constructor, nothing to do
  }

  size_t Decode(const Disassembler &disassembler, const DataExtractor &data,
                lldb::offset_t data_offset) override {
    // For SASS instructions parsed from nvdisasm, we don't need to decode
    // the bytes ourselves - the parsing was already done
    return GetOpcode().GetByteSize();
  }

  void SetOpcode(const void *opcode_data, size_t opcode_data_len) {
    m_opcode.SetOpcodeBytes(opcode_data, opcode_data_len);
  }

private:
  std::weak_ptr<DisassemblerSASS> m_disasm_wp;
  std::string m_predicate;
  std::string m_extra;
  std::map<std::string, std::string> m_other_attributes;
  std::vector<std::string> m_other_flags;

  // Cached attribute flags to avoid repeated lookups
  std::optional<bool> m_is_control_flow;
  std::optional<bool> m_is_subroutine_call;
  std::optional<bool> m_is_barrier;
  std::optional<bool> m_is_load;
};

// DisassemblerSASS implementation

DisassemblerSASS::DisassemblerSASS(const ArchSpec &arch, const char *flavor,
                                   const char *cpu, const char *features)
    : Disassembler(arch, flavor), m_valid(false) {

  Log *log = GetLog(LLDBLog::Disassembler);

  // Check if this is an NVPTX architecture (which produces SASS when compiled)
  if (arch.GetTriple().getArch() != llvm::Triple::nvptx &&
      arch.GetTriple().getArch() != llvm::Triple::nvptx64) {
    LLDB_LOG(log, "DisassemblerSASS: Unsupported architecture: {0}",
             arch.GetTriple().getArchName());
    return;
  }

  // Try to find nvdisasm
  if (llvm::Expected<FileSpec> nvdisasm_or = FindNvdisasm()) {
    m_nvdisasm_path = *nvdisasm_or;
    m_valid = true;
    LLDB_LOG(log,
             "DisassemblerSASS: Successfully initialized with nvdisasm at: {0}",
             m_nvdisasm_path.GetPath());
  } else {
    m_valid = false;
    Debugger::ReportError(llvm::toString(nvdisasm_or.takeError()));
  }
}

DisassemblerSASS::~DisassemblerSASS() = default;

lldb::DisassemblerSP DisassemblerSASS::CreateInstance(const ArchSpec &arch,
                                                      const char *flavor,
                                                      const char *cpu,
                                                      const char *features) {
  if (arch.GetTriple().getArch() == llvm::Triple::nvptx ||
      arch.GetTriple().getArch() == llvm::Triple::nvptx64) {
    auto disasm_sp =
        std::make_shared<DisassemblerSASS>(arch, flavor, cpu, features);
    if (disasm_sp && disasm_sp->IsValid())
      return disasm_sp;
  }
  return lldb::DisassemblerSP();
}

size_t DisassemblerSASS::DecodeInstructions(const Address &base_addr,
                                            const DataExtractor &data,
                                            lldb::offset_t data_offset,
                                            size_t num_instructions,
                                            bool append, bool data_from_file) {
  if (!append)
    m_instruction_list.Clear();

  if (!IsValid()) {
    Log *log = GetLog(LLDBLog::Disassembler);
    LLDB_LOG(log, "DisassemblerSASS::DecodeInstructions: Cannot disassemble - "
                  "nvdisasm not available");

    // Report error to debugger
    Debugger::ReportError(
        "nvdisasm not found. Please install CUDA toolkit or set "
        "CUDA_HOME environment variable.");
    return 0;
  }

  Log *log = GetLog(LLDBLog::Disassembler);
  LLDB_LOG(log,
           "DisassemblerSASS::DecodeInstructions: base_addr={0:x}, "
           "data_offset={1}, num_instructions={2}",
           base_addr.GetFileAddress(), data_offset, num_instructions);

  if (llvm::Expected<size_t> result_or =
          DisassembleWithNvdisasm(data, base_addr, num_instructions)) {
    return *result_or;
  } else {
    std::string error_msg = llvm::toString(result_or.takeError());
    LLDB_LOG(log, "DisassembleWithNvdisasm failed: {0}", error_msg);

    // Report error to debugger
    Debugger::ReportError(error_msg);
    return 0;
  }
}

void DisassemblerSASS::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Disassembler that uses nvdisasm to "
                                "disassemble SASS (Shader Assembly) code.",
                                CreateInstance);
}

void DisassemblerSASS::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

bool DisassemblerSASS::FlavorValidForArchSpec(
    const lldb_private::ArchSpec &arch, const char *flavor) {
  // Accept any flavor for NVPTX or default to "default"
  return (arch.GetTriple().getArch() == llvm::Triple::nvptx ||
          arch.GetTriple().getArch() == llvm::Triple::nvptx64) &&
         (flavor == nullptr || llvm::StringRef(flavor) == "default");
}

bool DisassemblerSASS::IsValid() const { return m_valid; }

llvm::Expected<FileSpec> DisassemblerSASS::FindNvdisasm() {
  // Use singleton cache to avoid repeated filesystem searches
  // Only cache successful results - failures are re-attempted each time
  auto &cache = DisassemblerSASSCache::getInstance();

  // If we already found it successfully, return cached result
  if (cache.cached_nvdisasm_path.has_value())
    return *cache.cached_nvdisasm_path;

  Log *log = GetLog(LLDBLog::Disassembler);

  // Helper lambda to handle successful nvdisasm discovery
  auto handle_nvdisasm_found = [&](const std::string &path,
                                   const char *source) -> FileSpec {
    FileSpec nvdisasm_path(path);
    LLDB_LOG(log, "Found nvdisasm {0} at: {1}", source, path);
    // Cache the successful result
    std::call_once(cache.search_once_nvdisasm, [&cache, &nvdisasm_path]() {
      cache.cached_nvdisasm_path = nvdisasm_path;
    });
    return *cache.cached_nvdisasm_path;
  };

  // First check if nvdisasm is available in PATH
  if (llvm::ErrorOr<std::string> path_result =
          llvm::sys::findProgramByName("nvdisasm")) {
    return handle_nvdisasm_found(*path_result, "in PATH");
  }

  // Try to find nvdisasm in common locations
  llvm::SmallVector<llvm::StringRef, 4> search_paths = {
      "/usr/local/cuda/bin", // Common CUDA installation
      "/opt/cuda/bin",       // Alternative installation
      "/usr/bin",            // System installation
  };

  // Also check CUDA_HOME environment variable
  std::string
      cuda_bin_dir_storage; // Storage for the StringRef at the correct scope
  const char *cuda_home = getenv("CUDA_HOME");
  if (cuda_home) {
    std::filesystem::path cuda_bin_dir =
        std::filesystem::path(cuda_home) / "bin";
    cuda_bin_dir_storage = cuda_bin_dir.string();
    search_paths.insert(search_paths.begin(), cuda_bin_dir_storage);
  }

  // Use findProgramByName with custom search paths
  if (llvm::ErrorOr<std::string> nvdisasm_result =
          llvm::sys::findProgramByName("nvdisasm", search_paths)) {
    return handle_nvdisasm_found(*nvdisasm_result, "in custom paths");
  }

  // Build error message with searched paths (add nvdisasm to each path for
  // display)
  std::string searched_paths;
  for (size_t i = 0; i < search_paths.size(); ++i) {
    if (i > 0)
      searched_paths += ", ";
    searched_paths += search_paths[i].str() + "/nvdisasm";
  }

  std::string error_msg =
      "nvdisasm not found or not executable. Please install CUDA toolkit, "
      "ensure nvdisasm is in your PATH, or set CUDA_HOME environment "
      "variable. Searched paths: " +
      searched_paths;

  LLDB_LOG(log, "nvdisasm not found in any search paths. Searched: {0}",
           searched_paths);

  return llvm::createStringError(error_msg);
}

llvm::Expected<std::string>
DisassemblerSASS::ExtractSmArchFromModule(const Address &base_addr) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // Get the module from the address
  lldb::ModuleSP module_sp = base_addr.GetModule();
  if (!module_sp)
    return llvm::createStringError("No module found for address");

  // Use singleton cache to get the per-module SM extractor
  auto &cache = DisassemblerSASSCache::getInstance();

  // Generate cache key for in-memory modules
  // For memory-only modules, prefer UUID, fallback to module pointer
  std::string cache_key;
  if (module_sp->GetUUID().IsValid())
    cache_key = module_sp->GetUUID().GetAsString();
  else {
    // Fallback to module pointer for modules without UUID
    cache_key = llvm::formatv("module_ptr_{0:x}",
                              reinterpret_cast<uintptr_t>(module_sp.get()))
                    .str();
  }
  LLDB_LOG(log, "ExtractSmArchFromModule: Cache key: {0}", cache_key);

  // Get the per-module SM extractor and delegate to it
  auto module_sm = cache.getModuleSM(cache_key);
  return module_sm->findSM(base_addr);
}

llvm::Expected<size_t>
DisassemblerSASS::DisassembleWithNvdisasm(const DataExtractor &data,
                                          const Address &base_addr,
                                          size_t max_instructions) {
  Log *log = GetLog(LLDBLog::Disassembler);

  if (!IsValid()) {
    LLDB_LOG(log, "DisassemblerSASS is not valid - nvdisasm not available");
    return llvm::createStringError("nvdisasm not available");
  }

  // Extract CUDA SM version from the module
  std::string sm_arch;
  if (llvm::Expected<std::string> sm_arch_or =
          ExtractSmArchFromModule(base_addr)) {
    sm_arch = *sm_arch_or;
  } else {
    LLDB_LOG(log, "Failed to extract SM architecture: {0}",
             llvm::toString(sm_arch_or.takeError()));
    // TODO DTCLLDB-63: Add a fallback here that the user can configure
    return llvm::createStringError(
        "Failed to extract SM architecture from CUDA module");
  }

  // Create a temporary file for the binary data
  llvm::SmallString<128> temp_file_path;
  int temp_fd;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "lldb_sass", "cubin", temp_fd, temp_file_path);
  if (ec)
    return llvm::createStringError(
        ec, "Failed to create temporary sass file for nvdisasm");

  // RAII cleanup for temporary cubin file
  llvm::FileRemover temp_file_remover(temp_file_path);

  // Write the binary data to the temporary file using LLVM's raw_fd_ostream
  {
    llvm::raw_fd_ostream file_stream(temp_fd, /*shouldClose=*/true);
    file_stream.write(reinterpret_cast<const char *>(data.GetDataStart()),
                      data.GetByteSize());

    if (file_stream.has_error()) {
      return llvm::createStringError("Failed to write data to temporary file");
    }
    // file_stream destructor automatically closes temp_fd and flushes
  }

  // Run nvdisasm safely without shell injection risks

  // Obtain location of nvdisasm
  std::string nvdisasm_path_str = m_nvdisasm_path.GetPath();

  // Prepare arguments for nvdisasm (including program name as argv[0])
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(nvdisasm_path_str);
  args.push_back("-b");
  args.push_back(sm_arch.c_str()); // Use extracted SM architecture
  args.push_back("--emit-json");
  args.push_back(temp_file_path.c_str());

  LLDB_LOG(log, "Running nvdisasm: {0} {1} {2} {3} {4}", args[0], args[1],
           args[2], args[3], args[4]);

  // Create temporary files for stdout and stderr redirection
  llvm::SmallString<128> stdout_path, stderr_path;
  std::error_code stdout_ec =
      llvm::sys::fs::createTemporaryFile("nvdisasm_stdout", "", stdout_path);
  if (stdout_ec) {
    return llvm::createStringError(
        stdout_ec, "Failed to create temporary output file for nvdisasm");
  }

  // RAII cleanup for stdout temporary file
  llvm::FileRemover stdout_file_remover(stdout_path);

  std::error_code stderr_ec =
      llvm::sys::fs::createTemporaryFile("nvdisasm_stderr", "", stderr_path);
  if (stderr_ec) {
    return llvm::createStringError(
        stderr_ec, "Failed to create temporary error file for nvdisasm");
  }

  // RAII cleanup for stderr temporary file
  llvm::FileRemover stderr_file_remover(stderr_path);

  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      std::nullopt,                 // stdin
      llvm::StringRef(stdout_path), // stdout
      llvm::StringRef(stderr_path)  // stderr
  };

  int exit_code = llvm::sys::ExecuteAndWait(nvdisasm_path_str, args,
                                            /*env=*/std::nullopt, redirects);

  // Read stdout and stderr
  auto stdout_buffer = llvm::MemoryBuffer::getFile(stdout_path);
  if (!stdout_buffer) {
    return llvm::createStringError(stdout_buffer.getError(),
                                   "Failed to read nvdisasm stdout");
  }
  std::string json_output = stdout_buffer.get()->getBuffer().str();

  auto stderr_buffer = llvm::MemoryBuffer::getFile(stderr_path);

  // Temporary files will be automatically cleaned up by FileRemover destructors

  if (exit_code != 0) {
    std::string stderr_str;
    if (stderr_buffer)
      stderr_str = stderr_buffer.get()->getBuffer().str();
    return llvm::createStringError("nvdisasm failed with exit code %d: %s",
                                   exit_code, stderr_str.c_str());
  }

  LLDB_LOG(log, "nvdisasm output: {0}", json_output);

  // Parse JSON output
  if (llvm::Expected<size_t> parse_result_or =
          ParseJsonOutput(json_output, base_addr, max_instructions)) {
    LLDB_LOG(log, "Parsed {0} instructions", *parse_result_or);
    return *parse_result_or;
  } else {
    return parse_result_or.takeError();
  }
}

llvm::Expected<size_t>
DisassemblerSASS::ParseJsonOutput(const std::string &json_output,
                                  const Address &base_addr,
                                  size_t max_instructions) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // Parse JSON output from nvdisasm
  llvm::Expected<llvm::json::Value> json_value = llvm::json::parse(json_output);
  if (!json_value) {
    LLDB_LOG(log, "Failed to parse JSON output from nvdisasm: {0}",
             llvm::toString(json_value.takeError()));
    return llvm::createStringError("Failed to parse JSON output from nvdisasm");
  }

  // The output is an array with two elements: [metadata, functions_array]
  const llvm::json::Array *root_array = json_value->getAsArray();
  if (!root_array || root_array->size() < 2) {
    LLDB_LOG(log,
             "JSON output is not an array or has insufficient elements. "
             "Expected: [metadata, functions_array], got size: {0}",
             root_array ? root_array->size() : 0);
    return llvm::createStringError(
        "JSON output is not an array or has insufficient elements");
  }

  // Validate metadata structure (first element)
  const llvm::json::Object *metadata = (*root_array)[0].getAsObject();
  if (!metadata)
    return llvm::createStringError("First element (metadata) is not an object");

  // Log some metadata information if available
  if (auto producer = metadata->getString("Producer"))
    LLDB_LOG(log, "nvdisasm producer: {0}", producer->str());
  if (auto sm_obj = metadata->getObject("SM"))
    if (auto version_obj = sm_obj->getObject("version"))
      if (auto major = version_obj->getInteger("major"))
        if (auto minor = version_obj->getInteger("minor"))
          LLDB_LOG(log, "Target SM version: {0}.{1}", *major, *minor);

  // The second element contains the functions array
  const llvm::json::Array *functions_array = (*root_array)[1].getAsArray();
  if (!functions_array)
    return llvm::createStringError("Second element is not an array");

  LLDB_LOG(log, "Found {0} functions in nvdisasm output",
           functions_array->size());

  size_t instructions_parsed = 0;

  // Process each function
  for (const llvm::json::Value &func_value : *functions_array) {
    const llvm::json::Object *func_obj = func_value.getAsObject();
    if (!func_obj)
      continue;

    // Get function metadata
    auto function_name = func_obj->getString("function-name");
    auto start_addr = func_obj->getInteger("start");
    auto length = func_obj->getInteger("length");

    if (!start_addr) {
      LLDB_LOG(log, "Function missing required 'start' field");
      continue;
    }

    // Log function information
    LLDB_LOG(log, "Processing function: name='{0}', start={1:x}, length={2}",
             function_name ? function_name->str() : "<unknown>", *start_addr,
             length ? *length : 0);

    // Get SASS instructions array
    const llvm::json::Array *sass_instructions =
        func_obj->getArray("sass-instructions");
    if (!sass_instructions) {
      LLDB_LOG(log, "Function has no 'sass-instructions' array");
      continue;
    }

    // Process each instruction
    size_t instruction_offset = 0;
    for (const llvm::json::Value &inst_value : *sass_instructions) {
      if (max_instructions != 0 && instructions_parsed >= max_instructions)
        break;

      const llvm::json::Object *inst_obj = inst_value.getAsObject();
      if (!inst_obj)
        continue;

      // Extract instruction information from JSON
      auto opcode_val = inst_obj->getString("opcode");
      auto operands_val = inst_obj->getString("operands");
      auto predicate_val = inst_obj->getString("predicate");
      auto extra_val = inst_obj->getString("extra");

      if (!opcode_val)
        continue;

      // Create address for this instruction
      // nvdisasm 'start' field represents offset within the disassembled
      // region, so: final_address = base_addr + start_offset +
      // instruction_index * 8
      Address inst_addr(base_addr);
      addr_t calculated_offset = *start_addr + instruction_offset * 8;

      LLDB_LOG(log,
               "Address calculation: base_addr={0:x}, start_addr={1:x}, "
               "instruction_offset={2}, calculated_offset={3:x}",
               base_addr.GetFileAddress(), *start_addr, instruction_offset,
               calculated_offset);

      inst_addr.Slide(calculated_offset);

      // Get optional fields
      std::string operands = operands_val ? operands_val->str() : "";
      std::string predicate = predicate_val ? predicate_val->str() : "";
      std::string extra = extra_val ? extra_val->str() : "";

      // Extract other-attributes
      std::map<std::string, std::string> other_attributes;
      if (const llvm::json::Object *attrs =
              inst_obj->getObject("other-attributes")) {
        for (const auto &attr : *attrs) {
          if (auto s = attr.second.getAsString())
            other_attributes[attr.first.str()] = s->str();
          else if (auto b = attr.second.getAsBoolean())
            other_attributes[attr.first.str()] = *b ? "True" : "False";
          else {
            // Fallback for integers / nulls
            other_attributes[attr.first.str()] =
                llvm::formatv("{0}", attr.second).str();
          }
        }
      }

      // Extract other-flags
      std::vector<std::string> other_flags;
      if (const llvm::json::Array *flags = inst_obj->getArray("other-flags"))
        for (const llvm::json::Value &flag_val : *flags)
          if (auto flag_str = flag_val.getAsString())
            other_flags.push_back(flag_str->str());

      // Create the instruction with enhanced information
      auto inst_sp = std::make_shared<InstructionSASS>(
          *this, inst_addr, opcode_val->str(), operands, predicate, extra,
          other_attributes, other_flags, AddressClass::eCode);

      // Set dummy opcode bytes (we don't have the actual bytes from nvdisasm)
      // SASS instructions are 8 bytes each
      uint64_t dummy_opcode = 0;
      inst_sp->SetOpcode(&dummy_opcode, sizeof(dummy_opcode));

      lldb::InstructionSP instruction_sp =
          std::static_pointer_cast<lldb_private::Instruction>(inst_sp);
      m_instruction_list.Append(instruction_sp);
      instructions_parsed++;
      instruction_offset++;

      LLDB_LOG(log,
               "Created instruction: opcode='{0}', operands='{1}', "
               "predicate='{2}', extra='{3}', attributes={4}, flags={5} at "
               "addr={6:x}",
               opcode_val->str(), operands, predicate, extra,
               other_attributes.size(), other_flags.size(),
               inst_addr.GetFileAddress());
    }

    if (max_instructions != 0 && instructions_parsed >= max_instructions)
      break;
  }

  return instructions_parsed;
}
