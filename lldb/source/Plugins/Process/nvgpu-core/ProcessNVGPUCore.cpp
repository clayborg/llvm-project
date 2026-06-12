//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessNVGPUCore.h"
#include "CudbgEntryParser.h"
#include "SectionUtils.h"
#include "ThreadNVGPUCore.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/CUDAAddressSpaces.h"
#include "lldb/Utility/NVGPU/NVGPUSectionID.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ProcessNVGPUCore)

llvm::StringRef ProcessNVGPUCore::GetPluginDescriptionStatic() {
  return "NVIDIA GPU core dump plug-in.";
}

void ProcessNVGPUCore::Initialize() {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessNVGPUCore::Terminate() {
  PluginManager::UnregisterPlugin(ProcessNVGPUCore::CreateInstance);
}

ProcessSP ProcessNVGPUCore::CreateInstance(TargetSP target_sp,
                                           ListenerSP listener_sp,
                                           const FileSpec *crash_file,
                                           bool can_connect) {
  if (!crash_file || can_connect)
    return nullptr;

  const size_t header_size = sizeof(llvm::ELF::Elf64_Ehdr);
  DataBufferSP data_sp = FileSystem::Instance().CreateDataBuffer(
      crash_file->GetPath(), header_size, 0);
  if (!data_sp || data_sp->GetByteSize() != header_size)
    return nullptr;

  if (!elf::ELFHeader::MagicBytesMatch(data_sp->GetBytes()))
    return nullptr;

  elf::ELFHeader elf_header;
  DataExtractor data(data_sp, eByteOrderLittle, 4);
  offset_t data_offset = 0;
  if (!elf_header.Parse(data, &data_offset))
    return nullptr;

  if (elf_header.e_type != llvm::ELF::ET_CORE)
    return nullptr;

  if (elf_header.e_machine != llvm::ELF::EM_CUDA)
    return nullptr;

  return std::make_shared<ProcessNVGPUCore>(target_sp, listener_sp,
                                            *crash_file);
}

ProcessNVGPUCore::ProcessNVGPUCore(TargetSP target_sp, ListenerSP listener_sp,
                                   const FileSpec &core_file)
    : PostMortemProcess(target_sp, listener_sp, core_file) {}

ProcessNVGPUCore::~ProcessNVGPUCore() = default;

bool ProcessNVGPUCore::CanDebug(TargetSP target_sp,
                                bool plugin_specified_by_name) {
  if (!m_core_module_sp && FileSystem::Instance().Exists(m_core_file)) {
    ModuleSpec core_module_spec(m_core_file, target_sp->GetArchitecture());
    Status error(ModuleList::GetSharedModule(core_module_spec, m_core_module_sp,
                                             nullptr, nullptr));
    if (m_core_module_sp) {
      ObjectFile *core_objfile = m_core_module_sp->GetObjectFile();
      if (core_objfile && core_objfile->GetType() == ObjectFile::eTypeCoreFile)
        return true;
    }
  }
  return false;
}

ObjectFile *ProcessNVGPUCore::GetCoreObjectFile() const {
  if (!m_core_module_sp)
    return nullptr;
  return m_core_module_sp->GetObjectFile();
}

Status ProcessNVGPUCore::DoLoadCore() {
  Target &target = GetTarget();
  // Hardcoded because `ObjectFileELF::GetArchitecture()` currently returns
  // the wrong arch for EM_CUDA: the ArchSpec table rows have no
  // distinguishing `sub`, so the lookup silently picks the first row
  // (32-bit `nvptx`), and `SetArchitecture`'s OS switch has no CUDA case,
  // so vendor/OS stay unset.
  ArchSpec arch("nvptx64-nvidia-cuda");
  target.SetArchitecture(arch);

  target.SetIsGPUTarget(true);

  if (PlatformSP nvgpu_platform =
          target.GetDebugger().GetPlatformList().GetOrCreate("nvgpu"))
    target.SetPlatform(nvgpu_platform);

  ObjectFile *core = GetCoreObjectFile();
  if (!core)
    return Status::FromErrorString("core module is not an ELF object file");

  // Find the synthetic nvgpucore root container that ObjectFileELF built.
  SectionList *section_list = core->GetSectionList();
  if (!section_list)
    return Status::FromErrorString("core file has no section list");
  m_root_sp = section_list->FindSectionByType(eSectionTypeNVGPURoot,
                                              /*check_children=*/false);
  if (!m_root_sp)
    return Status::FromErrorString(
        "NVGPU corefile did not produce a nvgpucore root section "
        "(likely missing or malformed nvgpu-device-table)");

  LoadProducerInfo(*section_list);

  // Register the corefile module itself in the target's module list so it
  // shows up in `image list` and is reachable from the SBModule API. Without
  // this, target.modules only contains the cubin sub-modules added by
  // LoadCubinModules below, and Python tooling that walks target.modules
  // can't find the synthetic nvgpucore section tree.
  target.GetImages().AppendIfNeeded(m_core_module_sp);

  if (llvm::Error err = LoadCubinModules())
    return Status::FromError(std::move(err));

  // Advertise CUDA address spaces so the DWARF evaluator routes
  // address-space-qualified reads through DoReadMemory(AddressSpec).
  m_address_spaces = nvgpu::GetAddressSpaceInfos();

  // NVGPU corefiles don't carry a host process ID -- the file is GPU
  // state, not Unix process state. Use 1 to match the live NVGPU debugger.
  SetID(1);
  return Status();
}

llvm::Error ProcessNVGPUCore::LoadCubinModules() {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOG(log, "ProcessNVGPUCore::LoadCubinModules()");

  ObjectFile *core = GetCoreObjectFile();
  Target &target = GetTarget();
  ModuleList loaded_modules;
  const FileSpec &core_file = core->GetFileSpec();

  size_t cubin_count = 0;
  // Only relocated cubins become Modules. They have absolute, resolvable
  // code addresses LLDB can use for symbolication. Unrelocated (`ucubin`)
  // images are still attached to the synthetic tree for inspection via
  // `image dump sections` but aren't loaded as code Modules.
  for (const SectionSP &cubin : nvgpu_core::FindChildrenByType(
           *m_root_sp, eSectionTypeNVGPURelocatedImage)) {
    const offset_t cubin_offset = cubin->GetFileOffset();
    const offset_t cubin_size = cubin->GetFileSize();
    if (cubin_size == 0)
      continue;

    // Point the module at the cubin's range within the core file so the
    // module system mmaps the same file.
    ModuleSpec module_spec(core_file);
    module_spec.SetObjectOffset(cubin_offset);
    module_spec.SetObjectSize(cubin_size);

    Status error;
    ModuleSP module_sp =
        target.GetOrCreateModule(module_spec, /*notify=*/true, &error);
    if (!module_sp) {
      LLDB_LOG(log,
               "  failed to load cubin module {0} ({1}) at offset {2:x} "
               "({3} bytes): {4}; core file may be truncated",
               cubin_count, cubin->GetName(), cubin_offset, cubin_size, error);
      ++cubin_count;
      continue;
    }

    bool changed = false;
    module_sp->SetLoadAddress(target, 0, /*value_is_offset=*/true, changed);
    if (changed)
      loaded_modules.AppendIfNeeded(module_sp);

    if (PlatformSP platform_sp = target.GetPlatform())
      platform_sp->RecordLoadedModule(module_sp, target);

    LLDB_LOG(log, "  loaded cubin module {0} ({1}) at offset {2:x} ({3} bytes)",
             cubin_count, cubin->GetName(), cubin_offset, cubin_size);
    ++cubin_count;
  }

  target.ModulesDidLoad(loaded_modules);
  return llvm::Error::success();
}

std::optional<DataExtractor>
ProcessNVGPUCore::GetNVGPUMetadata(const SectionList &sections) {
  // The metadata leaf hangs under the nvgpucore root, so search recursively.
  SectionSP sect_sp = sections.FindSectionByType(eSectionTypeNVGPUMetadata,
                                                 /*check_children=*/true);
  if (!sect_sp)
    return std::nullopt;
  DataExtractor data;
  if (sect_sp->GetSectionData(data))
    return data;
  return std::nullopt;
}

void ProcessNVGPUCore::LoadProducerInfo(const SectionList &sections) {
  Log *log = GetLog(LLDBLog::Process);

  if (std::optional<DataExtractor> metadata = GetNVGPUMetadata(sections))
    m_producer = nvgpu_core::DecodeProducerInfo(*metadata);

  if (!m_producer) {
    // The metadata section was added in driver r565, so every supported
    // in-major coredump has it. A missing section means we can't confirm the
    // producing driver matches this build; reads stay in-bounds (absent
    // fields read back as zero via the per-entry-size zero-fill in Decode),
    // but surface a warning since GPU state may be missing or misdecoded.
    Debugger::ReportWarning(
        "NVGPU corefile has no metadata section; the producing driver version "
        "could not be determined and some GPU state may be missing or decoded "
        "incorrectly.",
        GetTarget().GetDebugger().GetID());
    return;
  }

  LLDB_LOG(log, "NVGPU corefile producer: GPU driver branch r{0}, CUDA {1}.{2}",
           m_producer->driver_branch, m_producer->cuda_major,
           m_producer->cuda_minor);

  // Single-major policy: a coredump from a different CUDA major release may
  // use an incompatible field layout. Don't fail the load (the per-entry-size
  // zero-fill still keeps reads in-bounds), but surface a warning to the user
  // -- not just the log channel -- since GPU state may be decoded incorrectly.
  if (m_producer->cuda_major != 0 &&
      m_producer->cuda_major != CUDBG_API_VERSION_MAJOR)
    Debugger::ReportWarning(
        llvm::formatv(
            "NVGPU corefile was produced by CUDA {0}.{1}, but this debugger "
            "was built for CUDA {2}.x. Cross-major-release coredumps are not "
            "supported; some GPU state may be missing or decoded incorrectly.",
            m_producer->cuda_major, m_producer->cuda_minor,
            CUDBG_API_VERSION_MAJOR)
            .str(),
        GetTarget().GetDebugger().GetID());
}

bool ProcessNVGPUCore::DoUpdateThreadList(ThreadList &old_thread_list,
                                          ThreadList &new_thread_list) {
  Log *log = GetLog(LLDBLog::Process);

  ObjectFile *core = GetCoreObjectFile();

  uint32_t tid = 0;

  for (const SectionSP &warp :
       nvgpu_core::FindDescendantsByType(*m_root_sp, eSectionTypeNVGPUWarp)) {
    llvm::Expected<nvgpu_core::WarpEntry> warp_or =
        nvgpu_core::ReadAndDecode<nvgpu_core::WarpEntry>(warp, core);
    if (!warp_or) {
      LLDB_LOG(log, "ProcessNVGPUCore: skipping warp at {0}: {1}",
               warp->GetName(), llvm::toString(warp_or.takeError()));
      continue;
    }

    for (const SectionSP &lane :
         nvgpu_core::FindChildrenByType(*warp, eSectionTypeNVGPULane)) {
      const uint32_t lane_idx = nvgpu::DecodeHwIdx(lane->GetID());
      if (!warp_or->IsLaneValid(lane_idx))
        continue;
      ++tid;
      auto thread_sp =
          std::make_shared<ThreadNVGPUCore>(*this, tid, lane, lane_idx);
      new_thread_list.AddThread(thread_sp);

      // Remember the first thread of each stop kind so an interesting
      // thread is auto-selected: prefer an exception, fall back to a trap.
      if (m_exception_tid == LLDB_INVALID_THREAD_ID &&
          thread_sp->GetAttributedException() != 0) {
        m_exception_tid = tid;
      } else if (m_stop_tid == LLDB_INVALID_THREAD_ID &&
                 thread_sp->IsAtTrap()) {
        m_stop_tid = tid;
      }
    }
  }

  LLDB_LOG(log, "ProcessNVGPUCore: created {0} threads", tid);
  return tid > 0;
}

size_t ProcessNVGPUCore::ReadMemory(addr_t addr, void *buf, size_t size,
                                    Status &error) {
  if (ABISP abi_sp = GetABI())
    addr = abi_sp->FixAnyAddress(addr);
  return DoReadMemory(addr, buf, size, error);
}

SectionSP ProcessNVGPUCore::FindGlobalMemorySection(addr_t addr) const {
  for (SectionType t :
       {eSectionTypeNVGPUGlobalMemory, eSectionTypeNVGPUManagedMemory})
    for (const SectionSP &mem : nvgpu_core::FindChildrenByType(*m_root_sp, t))
      if (mem->ContainsFileAddress(addr))
        return mem;
  return nullptr;
}

size_t ProcessNVGPUCore::DoReadMemory(addr_t addr, void *buf, size_t size,
                                      Status &error) {
  ObjectFile *core_objfile = GetCoreObjectFile();

  SectionSP region = FindGlobalMemorySection(addr);
  if (!region) {
    error = Status::FromErrorStringWithFormat(
        "core file does not contain 0x%" PRIx64, addr);
    return 0;
  }

  const offset_t offset = addr - region->GetFileAddress();
  const size_t bytes_to_read =
      std::min(static_cast<offset_t>(size), region->GetByteSize() - offset);
  if (bytes_to_read == 0)
    return 0;
  return core_objfile->CopyData(region->GetFileOffset() + offset, bytes_to_read,
                                buf);
}

/// Read `size` bytes from `addr` within `mem_section`'s data window. Returns
/// 0 if the section is null/empty or `addr` is outside its file-address
/// range.
static size_t ReadFromMemorySection(SectionSP mem_section, addr_t addr,
                                    void *buf, size_t size,
                                    ObjectFile *core) {
  if (!mem_section || !mem_section->ContainsFileAddress(addr))
    return 0;
  DataExtractor data;
  core->ReadSectionData(mem_section.get(), data);
  if (data.GetByteSize() == 0)
    return 0;
  const offset_t offset = addr - mem_section->GetFileAddress();
  const size_t bytes_to_read =
      std::min<offset_t>(size, data.GetByteSize() - offset);
  return data.CopyData(offset, bytes_to_read, buf);
}

/// Locate the grid container that `gpu_thread` is part of by matching the
/// thread's CTA `gridId64` against the device's grid children (the grid
/// subtree is a sibling of the SM subtree under each device). Returns null
/// if the grid can't be resolved -- callers treat that as "unverifiable"
/// rather than an error.
static SectionSP FindGridSection(ThreadNVGPUCore &gpu_thread,
                                 ObjectFile *core) {
  Log *log = GetLog(LLDBLog::Process);
  SectionSP cta_sp = gpu_thread.GetCTASection();
  SectionSP dev_sp = gpu_thread.GetDeviceSection();
  if (!cta_sp || !dev_sp)
    return nullptr;

  llvm::Expected<nvgpu_core::CTAEntry> cta_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::CTAEntry>(cta_sp, core);
  if (!cta_or) {
    LLDB_LOG_ERROR(log, cta_or.takeError(),
                   "grid lookup: failed to decode CTA row: {0}");
    return nullptr;
  }

  for (const SectionSP &grid_sp :
       nvgpu_core::FindChildrenByType(*dev_sp, eSectionTypeNVGPUGrid)) {
    llvm::Expected<nvgpu_core::GridEntry> grid_or =
        nvgpu_core::ReadAndDecode<nvgpu_core::GridEntry>(grid_sp, core);
    if (!grid_or) {
      LLDB_LOG_ERROR(log, grid_or.takeError(),
                     "grid lookup: failed to decode grid row: {0}");
      continue;
    }
    if (grid_or->gridId64 == cta_or->gridId64)
      return grid_sp;
  }
  return nullptr;
}

/// Test whether `addr` falls within one of the grid's constant banks. Each
/// constbank table row is its own `eSectionTypeNVGPUConstBank` leaf (fanned
/// out by `ObjectFileELF::BuildNVGPUSectionList`), so the row stride is the
/// producer's sh_entsize rather than a hardcoded size.
static bool AddressInGridConstBanks(const Section &grid, ObjectFile *core,
                                    addr_t addr) {
  Log *log = GetLog(LLDBLog::Process);
  for (const SectionSP &cb :
       nvgpu_core::FindChildrenByType(grid, eSectionTypeNVGPUConstBank)) {
    llvm::Expected<nvgpu_core::ConstBankEntry> entry_or =
        nvgpu_core::ReadAndDecode<nvgpu_core::ConstBankEntry>(cb, core);
    if (!entry_or) {
      LLDB_LOG_ERROR(log, entry_or.takeError(),
                     "constbank scan: failed to decode row: {0}");
      continue;
    }
    if (entry_or->Contains(addr))
      return true;
  }
  return false;
}

size_t ProcessNVGPUCore::DoReadMemory(const AddressSpec &addr_spec,
                                      const AddressSpaceInfo &info, void *buf,
                                      size_t size, Status &error) {
  Log *log = GetLog(LLDBLog::Process);
  addr_t addr = addr_spec.GetValue();

  LLDB_LOG(log,
           "ProcessNVGPUCore::DoReadMemory(AddressSpec) addr={0:x} space={1} "
           "size={2}",
           addr, info.name, size);

  ObjectFile *core = GetCoreObjectFile();

  // Get the thread from the AddressSpec. An absent thread is the API's
  // expected signal (not all callers attach one), so we don't surface it
  // to the user; we just log it and fall back to the selected thread
  // below.
  ThreadSP thread_sp;
  if (llvm::Expected<ThreadSP> t = addr_spec.GetThread())
    thread_sp = *t;
  else
    LLDB_LOG_ERROR(log, t.takeError(),
                   "AddressSpec has no attached thread, falling back to "
                   "selected thread: {0}");

  if (!thread_sp)
    thread_sp = GetThreadList().GetSelectedThread();
  if (!thread_sp) {
    error = Status::FromErrorString("no thread for address space read");
    return 0;
  }

  auto *gpu_thread = static_cast<ThreadNVGPUCore *>(thread_sp.get());

  if (info.value == nvgpu::ConstStorage) {
    if (SectionSP grid_sp = FindGridSection(*gpu_thread, core)) {
      if (!AddressInGridConstBanks(*grid_sp, core, addr)) {
        error = Status::FromErrorStringWithFormat(
            "address 0x%" PRIx64 " is not within any constant bank", addr);
        return 0;
      }
    }
    return DoReadMemory(addr, buf, size, error);
  }

  SectionSP lane_sp = gpu_thread->GetLaneSection();
  SectionSP cta_sp = gpu_thread->GetCTASection();

  // Shared memory (per-CTA): under the lane's CTA ancestor.
  SectionSP shared_sp =
      nvgpu_core::FindChildByType(*cta_sp, eSectionTypeNVGPUSharedMemory);
  if (size_t bytes = ReadFromMemorySection(shared_sp, addr, buf, size, core))
    return bytes;

  // Local memory (per-lane): a named child of the lane container.
  SectionSP local_sp =
      nvgpu_core::FindChildByType(*lane_sp, eSectionTypeNVGPULocalMemory);
  if (size_t bytes = ReadFromMemorySection(local_sp, addr, buf, size, core))
    return bytes;

  error = Status::FromErrorStringWithFormat(
      "core file does not contain address space '%s' address 0x%" PRIx64,
      info.name.c_str(), addr);
  return 0;
}

Status ProcessNVGPUCore::DoGetMemoryRegionInfo(addr_t load_addr,
                                               MemoryRegionInfo &region_info) {
  region_info.Clear();

  SectionSP region = FindGlobalMemorySection(load_addr);
  if (region) {
    region_info.GetRange().SetRangeBase(region->GetFileAddress());
    region_info.GetRange().SetByteSize(region->GetByteSize());
    region_info.SetReadable(MemoryRegionInfo::eYes);
    region_info.SetWritable(MemoryRegionInfo::eNo);
    region_info.SetExecutable(MemoryRegionInfo::eNo);
    region_info.SetMapped(MemoryRegionInfo::eYes);
    return Status();
  }

  region_info.GetRange().SetRangeBase(load_addr);
  region_info.GetRange().SetByteSize(0);
  region_info.SetReadable(MemoryRegionInfo::eNo);
  region_info.SetWritable(MemoryRegionInfo::eNo);
  region_info.SetExecutable(MemoryRegionInfo::eNo);
  region_info.SetMapped(MemoryRegionInfo::eNo);
  return Status();
}
