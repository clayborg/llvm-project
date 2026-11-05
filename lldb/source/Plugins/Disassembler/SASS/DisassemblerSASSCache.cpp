//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DisassemblerSASSCache.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/NVGPU/PlatformNVGPU.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include <array>
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

static llvm::SmallVector<FileSpec> FindNvdisasmCandidates() {
  llvm::SmallVector<std::string> directories;
  if (const char *path = getenv("PATH")) {
    llvm::SmallVector<llvm::StringRef> paths;
    llvm::StringRef(path).split(paths, llvm::sys::EnvPathSeparator);
    for (llvm::StringRef directory : paths)
      directories.push_back(directory.empty() ? "." : directory.str());
  }
  if (const char *cuda_home = getenv("CUDA_HOME"))
    directories.push_back((std::filesystem::path(cuda_home) / "bin").string());
  directories.append({"/usr/local/cuda/bin", "/opt/cuda/bin", "/usr/bin"});

  // A default CUDA symlink can point to a toolkit that predates the core's SM.
  llvm::SmallVector<std::string> versioned_directories;
  std::error_code ec;
  for (std::filesystem::directory_iterator it("/usr/local", ec), end;
       !ec && it != end; it.increment(ec)) {
    if (llvm::StringRef(it->path().filename().string()).starts_with("cuda-"))
      versioned_directories.push_back((it->path() / "bin").string());
  }
  llvm::sort(versioned_directories);
  directories.append(versioned_directories);

  llvm::SmallVector<FileSpec> candidates;
  llvm::StringSet<> seen;
  for (const std::string &directory : directories) {
    auto path = llvm::sys::findProgramByName("nvdisasm", {directory});
    if (!path)
      continue;
    llvm::SmallString<128> resolved;
    if (llvm::sys::fs::real_path(*path, resolved))
      resolved = *path;
    if (seen.insert(resolved).second)
      candidates.emplace_back(*path);
  }
  return candidates;
}

static bool SupportsSM(const FileSpec &nvdisasm, llvm::StringRef sm_arch) {
  llvm::SmallString<128> output_path;
  if (llvm::sys::fs::createTemporaryFile("nvdisasm_help", "", output_path))
    return false;
  llvm::FileRemover remove_output(output_path);
  std::string program = nvdisasm.GetPath();
  std::array<llvm::StringRef, 2> args = {program, "--help"};
  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(""), llvm::StringRef(output_path), llvm::StringRef("")};
  if (llvm::sys::ExecuteAndWait(program, args, std::nullopt, redirects,
                                /*SecondsToWait=*/5) != 0)
    return false;
  auto output = llvm::MemoryBuffer::getFile(output_path);
  if (!output)
    return false;
  // Match a whole architecture token, so SM90 does not match SM90a.
  return (*output)->getBuffer().contains("'" + sm_arch.str() + "'");
}

llvm::Expected<lldb_private::FileSpec>
DisassemblerSASSCache::GetNvdisasmPath(llvm::StringRef sm_arch) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // If the user has set the nvdisasm path in the nvgpu plugin properties, use
  // it regardless of the cache and the predefinedsearch paths.
  FileSpec nvdisasm_path =
      platform_NVGPU::PlatformNVGPU::GetGlobalProperties().GetNvdisasmPath();
  if (!nvdisasm_path.GetPath().empty()) {
    LLDB_LOG(log, "Using nvdisasm from the nvgpu plugin properties: {0}",
             nvdisasm_path);
    return FileSpec(nvdisasm_path);
  }

  std::lock_guard<std::mutex> lock(m_nvdisasm_mutex);
  auto cached = m_nvdisasm_paths.find(sm_arch.str());
  if (cached != m_nvdisasm_paths.end())
    return cached->second;

  auto candidates = FindNvdisasmCandidates();
  for (const FileSpec &candidate : candidates) {
    if (sm_arch.empty() || SupportsSM(candidate, sm_arch)) {
      LLDB_LOG(log, "Using nvdisasm {0} for architecture '{1}'", candidate,
               sm_arch);
      m_nvdisasm_paths.emplace(sm_arch.str(), candidate);
      return candidate;
    }
    LLDB_LOG(log, "nvdisasm {0} does not advertise support for {1}", candidate,
             sm_arch);
  }

  if (candidates.empty())
    return llvm::createStringError(
        "nvdisasm not found or not executable. Please install CUDA toolkit, "
        "ensure nvdisasm is in your PATH, or set CUDA_HOME environment "
        "variable.");
  return llvm::createStringError(
      "No installed nvdisasm advertises support for %s. Set "
      "platform.plugin.nvgpu.nvdisasm-path to a compatible CUDA toolkit.",
      sm_arch.str().c_str());
}
