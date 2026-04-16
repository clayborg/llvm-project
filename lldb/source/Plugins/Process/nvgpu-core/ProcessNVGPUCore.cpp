//===-- ProcessNVGPUCore.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessNVGPUCore.h"
#include "ThreadNVGPUCore.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/CUDAAddressSpaces.h"
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

ObjectFileELF *ProcessNVGPUCore::GetCoreObjectFile() const {
  if (!m_core_module_sp)
    return nullptr;
  return llvm::dyn_cast<ObjectFileELF>(m_core_module_sp->GetObjectFile());
}

Status ProcessNVGPUCore::DoLoadCore() {
  Target &target = GetTarget();
  ArchSpec arch("nvptx64-nvidia-cuda");
  target.SetArchitecture(arch);

  target.SetIsGPUTarget(true);

  if (PlatformSP nvgpu_platform =
          target.GetDebugger().GetPlatformList().GetOrCreate("nvgpu"))
    target.SetPlatform(nvgpu_platform);

  ObjectFileELF *core = GetCoreObjectFile();
  if (!core)
    return Status::FromErrorString("core module is not an ELF object file");

  // Build section tree and parse top-level data (device table + global
  // memory regions). SM/CTA/warp/thread tables are deferred until
  // DoUpdateThreadList.
  if (llvm::Error err = m_core_data.ParseTopLevel(core))
    return Status::FromError(std::move(err));

  if (llvm::Error err = LoadCubinModules())
    return Status::FromError(std::move(err));

  // Advertise CUDA address spaces so the DWARF evaluator routes
  // address-space-qualified reads through DoReadMemory(AddressSpec).
  m_address_spaces = nvgpu::GetAddressSpaceInfos();

  SetID(1);
  return Status();
}

llvm::Error ProcessNVGPUCore::LoadCubinModules() {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOG(log, "ProcessNVGPUCore::LoadCubinModules()");

  ObjectFileELF *core = GetCoreObjectFile();
  if (!core)
    return llvm::createStringError("no core object file");

  Target &target = GetTarget();
  ModuleList loaded_modules;

  llvm::SmallVector<SectionNode *, 16> cubin_nodes =
      m_core_data.section_tree.FindAllByType(eSectionTypeCUDARelocatedImage);

  const FileSpec &core_file = core->GetFileSpec();

  for (size_t i = 0; i < cubin_nodes.size(); ++i) {
    SectionNode *node = cubin_nodes[i];
    if (!node || !node->section || !node->header)
      continue;

    uint64_t file_offset = node->header->sh_offset;
    uint64_t file_size = node->header->sh_size;
    if (file_size == 0)
      continue;

    // Point the module at the cubin's range within the core file so the
    // module system mmaps the same file -- no bulk copy of the cubin data.
    ModuleSpec module_spec(core_file);
    module_spec.SetObjectOffset(file_offset);
    module_spec.SetObjectSize(file_size);

    ModuleSP module_sp = target.GetOrCreateModule(module_spec, /*notify=*/true);
    if (module_sp) {
      bool changed = false;
      module_sp->SetLoadAddress(target, 0, /*value_is_offset=*/true, changed);
      if (changed)
        loaded_modules.AppendIfNeeded(module_sp);

      if (PlatformSP platform_sp = target.GetPlatform())
        platform_sp->RecordLoadedModule(module_sp, target);

      LLDB_LOG(log, "  loaded cubin module {0} at offset {1:x} ({2} bytes)",
               i, file_offset, file_size);
    }
  }

  target.ModulesDidLoad(loaded_modules);
  return llvm::Error::success();
}

bool ProcessNVGPUCore::DoUpdateThreadList(ThreadList &old_thread_list,
                                          ThreadList &new_thread_list) {
  Log *log = GetLog(LLDBLog::Process);
  ObjectFileELF *core = GetCoreObjectFile();
  if (!core)
    return false;

  uint32_t tid = 0;
  for (size_t dev_idx = 0; dev_idx < m_core_data.devices.size(); ++dev_idx) {
    DeviceData &dev = m_core_data.devices[dev_idx];
    llvm::ArrayRef<SMData> sms = dev.GetSMs(core);
    for (size_t sm_idx = 0; sm_idx < sms.size(); ++sm_idx) {
      const SMData &sm = sms[sm_idx];
      llvm::ArrayRef<CTAData> ctas = sm.GetCTAs(core);
      for (size_t cta_idx = 0; cta_idx < ctas.size(); ++cta_idx) {
        const CTAData &cta = ctas[cta_idx];
        llvm::ArrayRef<WarpData> warps = cta.GetWarps(core);
        for (size_t warp_idx = 0; warp_idx < warps.size(); ++warp_idx) {
          const WarpData &warp = warps[warp_idx];
          uint32_t valid_mask = warp.entry.validLanesMask;
          llvm::ArrayRef<LaneData> lanes = warp.GetLanes(core);
          for (size_t lane_idx = 0; lane_idx < lanes.size(); ++lane_idx) {
            if (!(valid_mask & (1u << lane_idx)))
              continue;

            NVGPULaneCoords coords;
            coords.dev_idx = dev_idx;
            coords.sm_idx = sm_idx;
            coords.cta_idx = cta_idx;
            coords.warp_idx = warp_idx;
            coords.lane_idx = lane_idx;

            ++tid;
            auto thread_sp =
                std::make_shared<ThreadNVGPUCore>(*this, tid, coords);
            new_thread_list.AddThread(thread_sp);
          }
        }
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

/// Try to read from a MemorySection (local or shared memory).
/// Returns the number of bytes read, or 0 if the address is not in range.
static size_t ReadFromMemorySection(const MemorySection &mem, addr_t addr,
                                    void *buf, size_t size) {
  if (mem.size == 0 || addr < mem.addr || addr >= mem.addr + mem.size)
    return 0;

  const uint64_t offset = addr - mem.addr;
  const uint64_t bytes_left = mem.size - offset;
  const size_t bytes_to_read = std::min(static_cast<uint64_t>(size), bytes_left);

  return mem.data.CopyData(offset, bytes_to_read, buf);
}

size_t ProcessNVGPUCore::DoReadMemory(addr_t addr, void *buf, size_t size,
                                      Status &error) {
  ObjectFile *core_objfile = GetCoreObjectFile();
  if (!core_objfile)
    return 0;

  const MemoryRegion *region = m_core_data.FindGlobalMemoryRegion(addr);
  if (!region) {
    error = Status::FromErrorStringWithFormat(
        "core file does not contain 0x%" PRIx64, addr);
    return 0;
  }

  const uint64_t offset = addr - region->addr;
  const uint64_t bytes_left =
      (region->size > offset) ? region->size - offset : 0;
  const size_t bytes_to_read =
      std::min(static_cast<uint64_t>(size), bytes_left);
  if (bytes_to_read == 0)
    return 0;
  return core_objfile->CopyData(region->file_offset + offset, bytes_to_read,
                                buf);
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

  ObjectFileELF *core = GetCoreObjectFile();
  if (!core)
    return 0;

  // Get the thread from the AddressSpec (set by the DWARF evaluator).
  llvm::Expected<ThreadSP> thread_or_err = addr_spec.GetThread();
  ThreadSP thread_sp;
  if (thread_or_err)
    thread_sp = *thread_or_err;
  else
    llvm::consumeError(thread_or_err.takeError());

  if (!thread_sp)
    thread_sp = GetThreadList().GetSelectedThread();

  if (!thread_sp) {
    error = Status::FromErrorString("no thread for address space read");
    return 0;
  }

  auto *gpu_thread = static_cast<ThreadNVGPUCore *>(thread_sp.get());
  const NVGPULaneCoords &coords = gpu_thread->GetCoords();

  if (coords.dev_idx >= m_core_data.devices.size()) {
    error = Status::FromErrorStringWithFormat(
        "device index %u out of range", coords.dev_idx);
    return 0;
  }
  DeviceData &dev = m_core_data.devices[coords.dev_idx];
  llvm::ArrayRef<SMData> sms = dev.GetSMs(core);
  if (coords.sm_idx >= sms.size()) {
    error = Status::FromErrorStringWithFormat(
        "SM index %u out of range", coords.sm_idx);
    return 0;
  }
  const SMData &sm = sms[coords.sm_idx];
  llvm::ArrayRef<CTAData> ctas = sm.GetCTAs(core);
  if (coords.cta_idx >= ctas.size()) {
    error = Status::FromErrorStringWithFormat(
        "CTA index %u out of range", coords.cta_idx);
    return 0;
  }
  const CTAData &cta = ctas[coords.cta_idx];

  // Shared memory (per-CTA) -- address space "shared" (8)
  size_t bytes = ReadFromMemorySection(cta.GetSharedMemory(core), addr, buf,
                                       size);
  if (bytes > 0)
    return bytes;

  llvm::ArrayRef<WarpData> warps = cta.GetWarps(core);
  if (coords.warp_idx < warps.size()) {
    const WarpData &warp = warps[coords.warp_idx];
    llvm::ArrayRef<LaneData> lanes = warp.GetLanes(core);
    if (coords.lane_idx < lanes.size()) {
      const LaneData &lane = lanes[coords.lane_idx];

      // Local memory (per-lane) -- address space "local" (6)
      bytes = ReadFromMemorySection(lane.GetLocalMemory(core), addr, buf, size);
      if (bytes > 0)
        return bytes;
    }
  }

  error = Status::FromErrorStringWithFormat(
      "core file does not contain address space '%s' address 0x%" PRIx64,
      info.name.c_str(), addr);
  return 0;
}

Status ProcessNVGPUCore::DoGetMemoryRegionInfo(addr_t load_addr,
                                               MemoryRegionInfo &region_info) {
  region_info.Clear();

  const MemoryRegion *region = m_core_data.FindGlobalMemoryRegion(load_addr);
  if (region) {
    region_info.GetRange().SetRangeBase(region->addr);
    region_info.GetRange().SetByteSize(region->size);
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
