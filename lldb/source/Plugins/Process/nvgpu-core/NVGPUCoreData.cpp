//===-- NVGPUCoreData.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVGPUCoreData.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

// ===== SectionNode =====

SectionNode *SectionNode::FindChild(SectionType type,
                                    uint32_t info) const {
  for (SectionNode *child : children) {
    if (child->type == type && child->sh_info == info)
      return child;
  }
  return nullptr;
}

// ===== SectionTree =====

llvm::Error SectionTree::Parse(ObjectFileELF *core) {
  Log *log = GetLog(LLDBLog::Process);
  SectionList *section_list = core->GetSectionList();
  size_t num_sections = core->GetNumSectionHeaders();

  uint64_t file_size = core->GetByteSize();
  m_truncated_sections.clear();
  m_nodes.resize(num_sections);
  for (size_t i = 1; i < num_sections; ++i) {
    const ObjectFileELF::ELFSectionHeaderInfo *hdr =
        core->GetSectionHeaderByIndex(i);
    if (!hdr)
      continue;

    // Skip sections whose data extends beyond the actual file size
    // (truncated corefile -- process was killed during dump). Use an
    // overflow-safe comparison: computing `sh_offset + sh_size` directly
    // could wrap uint64_t if `sh_offset` is corrupt or maliciously large.
    if (hdr->sh_size > 0 &&
        (hdr->sh_size > file_size ||
         hdr->sh_offset > file_size - hdr->sh_size)) {
      LLDB_LOG(log,
               "SectionTree: section {0} ({1}) at offset {2:x} is truncated "
               "(extends beyond file size {3:x}), skipping",
               i, hdr->section_name, hdr->sh_offset, file_size);
      m_truncated_sections.push_back(hdr->section_name.GetStringRef().str());
      continue;
    }

    auto node = std::make_unique<SectionNode>();
    node->sh_info = hdr->sh_info;
    node->sh_entsize = hdr->sh_entsize;
    node->section = section_list->FindSectionByID(i);
    node->type =
        node->section ? node->section->GetType() : eSectionTypeOther;
    m_nodes[i] = std::move(node);
  }

  for (size_t i = 1; i < num_sections; ++i) {
    if (!m_nodes[i])
      continue;
    const ObjectFileELF::ELFSectionHeaderInfo *hdr =
        core->GetSectionHeaderByIndex(i);
    uint32_t sh_link = hdr->sh_link;
    if (sh_link != 0 && sh_link < m_nodes.size() && m_nodes[sh_link]) {
      m_nodes[sh_link]->children.push_back(m_nodes[i].get());
    } else if (sh_link == 0) {
      m_roots.push_back(m_nodes[i].get());
    } else {
      LLDB_LOG(log,
               "SectionTree: section {0} has invalid sh_link={1}, skipping",
               i, sh_link);
    }
  }

  LLDB_LOG(log, "SectionTree: built tree with {0} nodes, {1} roots",
           num_sections, m_roots.size());
  return llvm::Error::success();
}

llvm::SmallVector<SectionNode *, 8>
SectionTree::GetRootsByType(SectionType type) const {
  llvm::SmallVector<SectionNode *, 8> result;
  for (SectionNode *root : m_roots) {
    if (root->type == type)
      result.push_back(root);
  }
  return result;
}

llvm::SmallVector<SectionNode *, 8>
SectionTree::GetNodesByType(SectionType type) const {
  llvm::SmallVector<SectionNode *, 8> result;
  for (const std::unique_ptr<SectionNode> &node : m_nodes) {
    if (node && node->type == type)
      result.push_back(node.get());
  }
  return result;
}

// ===== Parse Methods =====

llvm::Error DeviceData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated device table entry");
  offset_t off = 0;
  m_entry.devName = data.GetAddress(&off);
  m_entry.devType = data.GetAddress(&off);
  m_entry.smType = data.GetAddress(&off);
  m_entry.devId = data.GetU32(&off);
  m_entry.pciBusId = data.GetU32(&off);
  m_entry.pciDevId = data.GetU32(&off);
  m_entry.numSMs = data.GetU32(&off);
  m_entry.numWarpsPerSM = data.GetU32(&off);
  m_entry.numLanesPerWarp = data.GetU32(&off);
  m_entry.numRegsPerLane = data.GetU32(&off);
  m_entry.numPredicatesPrLane = data.GetU32(&off);
  m_entry.smMajor = data.GetU32(&off);
  m_entry.smMinor = data.GetU32(&off);
  m_entry.instructionSize = data.GetU32(&off);
  m_entry.status = data.GetU32(&off);
  m_entry.numUniformRegsPrWarp = data.GetU32(&off);
  m_entry.numUniformPredicatesPrWarp = data.GetU32(&off);
  m_entry.numConvergenceBarriersPrWarp = data.GetU32(&off);
  return llvm::Error::success();
}

llvm::Error SMData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated SM table entry");
  offset_t off = 0;
  m_entry.smId = data.GetU32(&off);
  m_entry.padding0 = data.GetU32(&off);
  m_entry.exception = data.GetU32(&off);
  m_entry.errorPCValid = data.GetU32(&off);
  m_entry.errorPC = data.GetAddress(&off);
  m_entry.clusterExceptionTargetBlockIdxValid = data.GetU32(&off);
  m_entry.clusterExceptionTargetBlockIdxX = data.GetU32(&off);
  m_entry.clusterExceptionTargetBlockIdxY = data.GetU32(&off);
  m_entry.clusterExceptionTargetBlockIdxZ = data.GetU32(&off);
  m_entry.exceptionString = data.GetAddress(&off);
  return llvm::Error::success();
}

llvm::Error CTAData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated CTA table entry");
  offset_t off = 0;
  m_entry.gridId64 = data.GetAddress(&off);
  m_entry.blockIdxX = data.GetU32(&off);
  m_entry.blockIdxY = data.GetU32(&off);
  m_entry.blockIdxZ = data.GetU32(&off);
  m_entry.padding0 = data.GetU32(&off);
  m_entry.clusterIdxX = data.GetU32(&off);
  m_entry.clusterIdxY = data.GetU32(&off);
  m_entry.clusterIdxZ = data.GetU32(&off);
  m_entry.padding1 = data.GetU32(&off);
  m_entry.clusterDimX = data.GetU32(&off);
  m_entry.clusterDimY = data.GetU32(&off);
  m_entry.clusterDimZ = data.GetU32(&off);
  return llvm::Error::success();
}

llvm::Error WarpData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated warp table entry");
  offset_t off = 0;
  m_entry.errorPC = data.GetAddress(&off);
  m_entry.warpId = data.GetU32(&off);
  m_entry.validLanesMask = data.GetU32(&off);
  m_entry.activeLanesMask = data.GetU32(&off);
  m_entry.isWarpBroken = data.GetU32(&off);
  m_entry.errorPCValid = data.GetU32(&off);
  m_entry.padding0 = data.GetU32(&off);
  m_entry.numRegs = data.GetU32(&off);
  m_entry.padding1 = data.GetU32(&off);
  m_entry.sharedMemSize = data.GetU32(&off);
  m_entry.padding2 = data.GetU32(&off);
  m_entry.inSyscallLanesMask = data.GetU32(&off);
  m_entry.cbuActiveLanesMask = data.GetU32(&off);
  m_entry.cbuExitedLanesMask = data.GetU32(&off);
  m_entry.cbuCollectiveLanesMask = data.GetU32(&off);
  m_entry.barrierScope = data.GetU32(&off);
  m_entry.padding3 = data.GetU32(&off);
  m_entry.additionalBarrierInfo = data.GetAddress(&off);
  return llvm::Error::success();
}

llvm::Error LaneData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated lane table entry");
  offset_t off = 0;
  m_entry.virtualPC = data.GetAddress(&off);
  m_entry.physPC = data.GetAddress(&off);
  m_entry.ln = data.GetU32(&off);
  m_entry.threadIdxX = data.GetU32(&off);
  m_entry.threadIdxY = data.GetU32(&off);
  m_entry.threadIdxZ = data.GetU32(&off);
  m_entry.exception = data.GetU32(&off);
  m_entry.callDepth = data.GetU32(&off);
  m_entry.syscallCallDepth = data.GetU32(&off);
  m_entry.ccRegister = data.GetU32(&off);
  m_entry.cbuThreadState = data.GetU32(&off);
  return llvm::Error::success();
}

// ===== Lazy Child Loading =====

/// Parse all entries from a child section node and wire each parsed entry
/// to its tree context using the on-disk row index. The row index is the
/// `sh_info` value the entry's own children will use to find back to it
/// via `SectionNode::FindChild`.
template <typename EntryT>
static std::vector<EntryT> ParseTableFromNode(ObjectFileELF *core,
                                              SectionNode *node) {
  std::vector<EntryT> entries;
  if (!node || !node->section)
    return entries;

  DataExtractor data;
  core->ReadSectionData(node->section.get(), data);
  uint64_t entry_size = node->sh_entsize;
  if (entry_size == 0 || data.GetByteSize() == 0)
    return entries;

  Log *log = GetLog(LLDBLog::Process);
  uint64_t num_entries = data.GetByteSize() / entry_size;
  entries.reserve(num_entries);
  for (uint64_t i = 0; i < num_entries; ++i) {
    uint64_t offset = i * entry_size;
    if (offset + entry_size > data.GetByteSize())
      break;
    DataExtractor entry_data(data, offset, entry_size);
    EntryT entry;
    if (llvm::Error err = entry.Parse(entry_data, entry_size)) {
      LLDB_LOG(log,
               "ParseTableFromNode: skipping entry {0} in section {1}: {2}",
               i, node->section->GetName(),
               llvm::toString(std::move(err)));
      continue;
    }
    entry.SetTreeContext(node, static_cast<uint32_t>(i));
    entries.push_back(std::move(entry));
  }
  return entries;
}

/// Lazy child-table accessor. Uses `parent_idx` to find the child
/// section by `sh_info`, then parses it via `ParseTableFromNode`.
template <typename T>
static llvm::ArrayRef<T> LazyLoadChildren(bool &loaded, std::vector<T> &cache,
                                          SectionNode *parent_node,
                                          uint32_t parent_idx,
                                          SectionType child_type,
                                          ObjectFileELF *core) {
  if (loaded)
    return cache;
  loaded = true;
  if (!parent_node)
    return cache;
  if (SectionNode *child = parent_node->FindChild(child_type, parent_idx))
    cache = ParseTableFromNode<T>(core, child);
  return cache;
}

llvm::ArrayRef<SMData> DeviceData::GetSMs(ObjectFileELF *core) const {
  return LazyLoadChildren(m_sms_loaded, m_sms, m_dev_table_node, m_dev_idx,
                          eSectionTypeCUDASmTable, core);
}

llvm::ArrayRef<CTAData> SMData::GetCTAs(ObjectFileELF *core) const {
  return LazyLoadChildren(m_ctas_loaded, m_ctas, m_sm_table_node, m_sm_idx,
                          eSectionTypeCUDACtaTable, core);
}

llvm::ArrayRef<WarpData> CTAData::GetWarps(ObjectFileELF *core) const {
  return LazyLoadChildren(m_warps_loaded, m_warps, m_cta_table_node,
                          m_cta_idx, eSectionTypeCUDAWarpTable, core);
}

llvm::ArrayRef<LaneData> WarpData::GetLanes(ObjectFileELF *core) const {
  return LazyLoadChildren(m_lanes_loaded, m_lanes, m_warp_table_node,
                          m_warp_idx, eSectionTypeCUDALaneTable, core);
}

// ===== Lazy Section Loading (zero-copy views over the corefile mmap) =====

/// Cache-population overloads for `LazyLoadSection`. A `DataExtractor` cache
/// just gets the section bytes; a `MemorySection` cache also stashes the
/// address range so memory-region lookups can match against it later.
static void PopulateCache(DataExtractor &cache, Section *section,
                          ObjectFileELF *core) {
  core->ReadSectionData(section, cache);
}

static void PopulateCache(MemorySection &cache, Section *section,
                          ObjectFileELF *core) {
  core->ReadSectionData(section, cache.data);
  cache.addr = section->GetFileAddress();
  // Use the DataExtractor's size as the authoritative byte count. CUDA
  // local-memory and shared-memory sections intentionally have SHF_ALLOC
  // cleared (see `ObjectFileELF::CreateSections`) so that LLDB's section
  // overlap detection doesn't drop all but one per lane/CTA (they all
  // share the same `sh_addr`). That also means `section->GetByteSize()`
  // returns 0 for those sections. The DataExtractor reflects what
  // ReadSectionData actually produced, which matches `sh_size`.
  cache.size = cache.data.GetByteSize();
}

/// Lazy load of a child section into either a `DataExtractor` view or a
/// `MemorySection` wrapper, dispatched via the `PopulateCache` overloads
/// above. The cache aliases the corefile mmap; nothing is copied.
template <typename T>
static const T &LazyLoadSection(bool &loaded, T &cache,
                                SectionNode *parent_node,
                                uint32_t parent_idx,
                                SectionType child_type,
                                ObjectFileELF *core) {
  if (loaded)
    return cache;
  loaded = true;
  if (!parent_node)
    return cache;
  if (SectionNode *node = parent_node->FindChild(child_type, parent_idx)) {
    if (node->section)
      PopulateCache(cache, node->section.get(), core);
  }
  return cache;
}

const DataExtractor &LaneData::GetRegisters(ObjectFileELF *core) const {
  return LazyLoadSection(m_regs_loaded, m_regs, m_lane_table_node,
                         m_lane_idx, eSectionTypeCUDARegisters, core);
}

const DataExtractor &LaneData::GetPredicates(ObjectFileELF *core) const {
  return LazyLoadSection(m_predicates_loaded, m_predicates, m_lane_table_node,
                         m_lane_idx, eSectionTypeCUDAPredicates, core);
}

const DataExtractor &
WarpData::GetUniformRegisters(ObjectFileELF *core) const {
  return LazyLoadSection(m_uniform_regs_loaded, m_uniform_regs,
                         m_warp_table_node, m_warp_idx,
                         eSectionTypeCUDAUniformRegisters, core);
}

const DataExtractor &
WarpData::GetUniformPredicates(ObjectFileELF *core) const {
  return LazyLoadSection(m_uniform_preds_loaded, m_uniform_preds,
                         m_warp_table_node, m_warp_idx,
                         eSectionTypeCUDAUniformPredicates, core);
}

const MemorySection &LaneData::GetLocalMemory(ObjectFileELF *core) const {
  return LazyLoadSection(m_local_mem_loaded, m_local_mem, m_lane_table_node,
                         m_lane_idx, eSectionTypeCUDALocalMemory, core);
}

const MemorySection &CTAData::GetSharedMemory(ObjectFileELF *core) const {
  return LazyLoadSection(m_shared_mem_loaded, m_shared_mem, m_cta_table_node,
                         m_cta_idx, eSectionTypeCUDASharedMemory, core);
}

// ===== NVGPUCoreData Top-Level =====

llvm::Error NVGPUCoreData::Parse(ObjectFileELF *core) {
  Log *log = GetLog(LLDBLog::Process);

  if (llvm::Error err = m_section_tree.Parse(core))
    return err;

  // Parse device table. ParseTableFromNode wires SetTreeContext on each
  // entry using its on-disk row index.
  llvm::SmallVector<SectionNode *, 8> dev_nodes =
      m_section_tree.GetRootsByType(eSectionTypeCUDADeviceTable);
  if (!dev_nodes.empty()) {
    m_devices = ParseTableFromNode<DeviceData>(core, dev_nodes[0]);
    LLDB_LOG(log, "NVGPUCoreData: parsed {0} devices", m_devices.size());
  }

  // Collect global memory regions (lazy reads via CopyData). CUDA global
  // and managed memory sections keep SHF_ALLOC (see
  // `ObjectFileELF::CreateSections`) so `Section::GetByteSize()` reports
  // the real sh_size for them.
  llvm::SmallVector<SectionNode *, 8> global_nodes =
      m_section_tree.GetRootsByType(eSectionTypeCUDAGlobalMemory);
  llvm::SmallVector<SectionNode *, 8> managed_nodes =
      m_section_tree.GetRootsByType(eSectionTypeCUDAManagedMemory);
  global_nodes.insert(global_nodes.end(), managed_nodes.begin(),
                      managed_nodes.end());

  for (SectionNode *node : global_nodes) {
    if (!node->section || node->section->GetByteSize() == 0)
      continue;
    MemoryRegion region;
    region.addr = node->section->GetFileAddress();
    region.size = node->section->GetByteSize();
    region.file_offset = node->section->GetFileOffset();
    m_global_memory.push_back(region);
  }
  LLDB_LOG(log, "NVGPUCoreData: found {0} global memory regions",
           m_global_memory.size());

  // Collect cubin module locations.
  llvm::SmallVector<SectionNode *, 8> cubin_nodes =
      m_section_tree.GetNodesByType(eSectionTypeCUDARelocatedImage);
  for (SectionNode *node : cubin_nodes) {
    if (!node->section)
      continue;
    uint64_t offset = node->section->GetFileOffset();
    uint64_t size = node->section->GetFileSize();
    if (size > 0)
      m_cubin_locations.push_back({offset, size});
  }
  LLDB_LOG(log, "NVGPUCoreData: found {0} cubin modules",
           m_cubin_locations.size());

  return llvm::Error::success();
}

const MemoryRegion *
NVGPUCoreData::FindGlobalMemoryRegion(uint64_t addr) const {
  for (const MemoryRegion &region : m_global_memory) {
    if (addr >= region.addr && addr < region.addr + region.size)
      return &region;
  }
  return nullptr;
}
