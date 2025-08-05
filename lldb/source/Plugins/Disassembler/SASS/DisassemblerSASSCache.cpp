//===-- DisassemblerSASSCache.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DisassemblerSASSCache.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "llvm/Support/Program.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include <filesystem>

using namespace lldb_private;
using namespace lldb;

std::shared_ptr<ModuleSM>
DisassemblerSASSCache::GetModuleSM(const std::string &cache_key) {
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

llvm::Expected<std::string>
ModuleSM::FindSMForELFv7OrLower(lldb::ModuleSP module_sp) {
  ObjectFile *obj_file = module_sp->GetObjectFile();
  if (!obj_file)
    return llvm::createStringError("No object file found in module");

  DataExtractor header_data;
  if (obj_file->GetData(48, 4, header_data) < 4)
    return llvm::createStringError(
        "Failed to read the e_flags of the ELF header");

  lldb::offset_t hdr_offset = 0;
  uint32_t e_flags = header_data.GetU32(&hdr_offset);
  uint32_t sm_ver = e_flags & 0xFF;
  std::string sm_arch = llvm::formatv("SM{0}", sm_ver).str();
  return sm_arch;
}

llvm::Expected<std::string>
ModuleSM::FindSMForELFv8OrGreater(lldb::ModuleSP module_sp) {
  Log *log = GetLog(LLDBLog::Disassembler);

  ObjectFile *obj_file = module_sp->GetObjectFile();
  if (!obj_file)
    return llvm::createStringError("No object file found in module");

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

/// Implementation of ModuleSM::findSM
llvm::Expected<std::string> ModuleSM::FindSM(const Address &base_addr) {
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

  // Check ELF header ABI version.
  if (obj_file->GetPluginName() != ObjectFileELF::GetPluginNameStatic())
    return llvm::createStringError("Object file is not ELF format");

  // For ELF files, we need to check the ABI version in the ELF header.
  DataExtractor header_data;
  lldb::offset_t hdr_offset = 0;
  if (obj_file->GetData(0, 64, header_data) < 16)
    return llvm::createStringError("Failed to read ELF header");

  hdr_offset = 8; // EI_ABIVERSION offset
  uint8_t abi_version = header_data.GetU8(&hdr_offset);

  LLDB_LOG(log, "ModuleSM::findSM() ELF ABI version {0} identified",
           abi_version);

  llvm::Expected<std::string> sm_ver = abi_version >= 8
                                           ? FindSMForELFv8OrGreater(module_sp)
                                           : FindSMForELFv7OrLower(module_sp);
  if (sm_ver) {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_cached_sm_arch = *sm_ver;
    }
    return *sm_ver;
  }
  return llvm::createStringError("Failed to find SM architecture %s",
                                 llvm::toString(sm_ver.takeError()).c_str());
}

llvm::Expected<lldb_private::FileSpec>
DisassemblerSASSCache::GetNvdisasmPath() {
  // If we already found it successfully, return cached result
  if (m_cached_nvdisasm_path.has_value())
    return *m_cached_nvdisasm_path;

  Log *log = GetLog(LLDBLog::Disassembler);

  // Helper lambda to handle successful nvdisasm discovery
  auto handle_nvdisasm_found = [&](const std::string &path,
                                   const char *source) -> FileSpec {
    FileSpec nvdisasm_path(path);
    LLDB_LOG(log, "Found nvdisasm {0} at: {1}", source, path);
    // Cache the successful result
    std::call_once(m_search_once_nvdisasm,
                   [&]() { m_cached_nvdisasm_path = nvdisasm_path; });
    return *m_cached_nvdisasm_path;
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
