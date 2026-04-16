//===-- NVGPUCoreData.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVGPUCoreData.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

// ===== SectionNode =====

llvm::SmallVector<SectionNode *, 4>
SectionNode::GetChildrenByType(SectionType child_type) const {
  llvm::SmallVector<SectionNode *, 4> result;
  for (SectionNode *child : children) {
    if (child->type == child_type)
      result.push_back(child);
  }
  return result;
}

SectionNode *SectionNode::FindChild(SectionType child_type,
                                    uint32_t info) const {
  for (SectionNode *child : children) {
    if (child->type == child_type && child->sh_info == info)
      return child;
  }
  return nullptr;
}

// ===== SectionTree =====

llvm::Error SectionTree::Build(ObjectFileELF *core) {
  Log *log = GetLog(LLDBLog::Process);
  SectionList *section_list = core->GetSectionList();
  size_t num_sections = core->GetNumSectionHeaders();

  m_nodes.resize(num_sections);
  for (size_t i = 1; i < num_sections; ++i) {
    const ObjectFileELF::ELFSectionHeaderInfo *hdr =
        core->GetSectionHeaderByIndex(i);
    if (!hdr)
      continue;
    auto node = std::make_unique<SectionNode>();
    node->section_idx = i;
    node->sh_info = hdr->sh_info;
    node->header = hdr;
    node->section = section_list->FindSectionByID(i);
    node->type =
        node->section ? node->section->GetType() : eSectionTypeOther;
    m_nodes[i] = std::move(node);
  }

  for (size_t i = 1; i < num_sections; ++i) {
    if (!m_nodes[i])
      continue;
    uint32_t sh_link = m_nodes[i]->header->sh_link;
    if (sh_link != 0 && sh_link < m_nodes.size() && m_nodes[sh_link]) {
      m_nodes[i]->parent = m_nodes[sh_link].get();
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

SectionNode *SectionTree::FindRootByType(SectionType type) const {
  for (SectionNode *root : m_roots) {
    if (root->type == type)
      return root;
  }
  return nullptr;
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

llvm::SmallVector<SectionNode *, 16>
SectionTree::FindAllByType(SectionType type) const {
  llvm::SmallVector<SectionNode *, 16> result;
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
  entry.devName = data.GetAddress(&off);
  entry.devType = data.GetAddress(&off);
  entry.smType = data.GetAddress(&off);
  entry.devId = data.GetU32(&off);
  entry.pciBusId = data.GetU32(&off);
  entry.pciDevId = data.GetU32(&off);
  entry.numSMs = data.GetU32(&off);
  entry.numWarpsPerSM = data.GetU32(&off);
  entry.numLanesPerWarp = data.GetU32(&off);
  entry.numRegsPerLane = data.GetU32(&off);
  entry.numPredicatesPrLane = data.GetU32(&off);
  entry.smMajor = data.GetU32(&off);
  entry.smMinor = data.GetU32(&off);
  entry.instructionSize = data.GetU32(&off);
  entry.status = data.GetU32(&off);
  entry.numUniformRegsPrWarp = data.GetU32(&off);
  entry.numUniformPredicatesPrWarp = data.GetU32(&off);
  entry.numConvergenceBarriersPrWarp = data.GetU32(&off);
  return llvm::Error::success();
}

llvm::Error SMData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated SM table entry");
  offset_t off = 0;
  entry.smId = data.GetU32(&off);
  entry.padding0 = data.GetU32(&off);
  entry.exception = data.GetU32(&off);
  entry.errorPCValid = data.GetU32(&off);
  entry.errorPC = data.GetAddress(&off);
  entry.clusterExceptionTargetBlockIdxValid = data.GetU32(&off);
  entry.clusterExceptionTargetBlockIdxX = data.GetU32(&off);
  entry.clusterExceptionTargetBlockIdxY = data.GetU32(&off);
  entry.clusterExceptionTargetBlockIdxZ = data.GetU32(&off);
  entry.exceptionString = data.GetAddress(&off);
  return llvm::Error::success();
}

llvm::Error CTAData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated CTA table entry");
  offset_t off = 0;
  entry.gridId64 = data.GetAddress(&off);
  entry.blockIdxX = data.GetU32(&off);
  entry.blockIdxY = data.GetU32(&off);
  entry.blockIdxZ = data.GetU32(&off);
  entry.padding0 = data.GetU32(&off);
  entry.clusterIdxX = data.GetU32(&off);
  entry.clusterIdxY = data.GetU32(&off);
  entry.clusterIdxZ = data.GetU32(&off);
  entry.padding1 = data.GetU32(&off);
  entry.clusterDimX = data.GetU32(&off);
  entry.clusterDimY = data.GetU32(&off);
  entry.clusterDimZ = data.GetU32(&off);
  return llvm::Error::success();
}

llvm::Error WarpData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated warp table entry");
  offset_t off = 0;
  entry.errorPC = data.GetAddress(&off);
  entry.warpId = data.GetU32(&off);
  entry.validLanesMask = data.GetU32(&off);
  entry.activeLanesMask = data.GetU32(&off);
  entry.isWarpBroken = data.GetU32(&off);
  entry.errorPCValid = data.GetU32(&off);
  entry.padding0 = data.GetU32(&off);
  entry.numRegs = data.GetU32(&off);
  entry.padding1 = data.GetU32(&off);
  entry.sharedMemSize = data.GetU32(&off);
  entry.padding2 = data.GetU32(&off);
  entry.inSyscallLanesMask = data.GetU32(&off);
  entry.cbuActiveLanesMask = data.GetU32(&off);
  entry.cbuExitedLanesMask = data.GetU32(&off);
  entry.cbuCollectiveLanesMask = data.GetU32(&off);
  entry.barrierScope = data.GetU32(&off);
  entry.padding3 = data.GetU32(&off);
  entry.additionalBarrierInfo = data.GetAddress(&off);
  return llvm::Error::success();
}

llvm::Error LaneData::Parse(const DataExtractor &data, uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated lane table entry");
  offset_t off = 0;
  entry.virtualPC = data.GetAddress(&off);
  entry.physPC = data.GetAddress(&off);
  entry.ln = data.GetU32(&off);
  entry.threadIdxX = data.GetU32(&off);
  entry.threadIdxY = data.GetU32(&off);
  entry.threadIdxZ = data.GetU32(&off);
  entry.exception = data.GetU32(&off);
  entry.callDepth = data.GetU32(&off);
  entry.syscallCallDepth = data.GetU32(&off);
  entry.ccRegister = data.GetU32(&off);
  entry.cbuThreadState = data.GetU32(&off);
  return llvm::Error::success();
}

// ===== Lazy Child Loading =====

/// Helper to parse table entries from a section node.
template <typename EntryT>
static std::vector<EntryT> ParseTableFromNode(ObjectFileELF *core,
                                              SectionNode *node) {
  std::vector<EntryT> entries;
  if (!node || !node->section || !node->header)
    return entries;

  DataExtractor data;
  core->ReadSectionData(node->section.get(), data);
  uint64_t entry_size = node->header->sh_entsize;
  if (entry_size == 0 || data.GetByteSize() == 0)
    return entries;

  uint64_t num_entries = data.GetByteSize() / entry_size;
  entries.reserve(num_entries);
  for (uint64_t i = 0; i < num_entries; ++i) {
    uint64_t offset = i * entry_size;
    if (offset + entry_size > data.GetByteSize())
      break;
    DataExtractor entry_data(data, offset, entry_size);
    EntryT entry;
    if (llvm::Error err = entry.Parse(entry_data, entry_size)) {
      llvm::consumeError(std::move(err));
      continue;
    }
    entries.push_back(std::move(entry));
  }
  return entries;
}

llvm::ArrayRef<SMData> DeviceData::GetSMs(ObjectFileELF *core) const {
  if (m_sms_loaded)
    return m_sms;
  m_sms_loaded = true;

  if (!m_dev_table_node)
    return m_sms;

  SectionNode *sm_table =
      m_dev_table_node->FindChild(eSectionTypeCUDASmTable, m_dev_index);
  if (!sm_table)
    return m_sms;

  m_sms = ParseTableFromNode<SMData>(core, sm_table);
  for (uint32_t i = 0; i < m_sms.size(); ++i)
    m_sms[i].SetTreeContext(sm_table, i);

  return m_sms;
}

llvm::ArrayRef<CTAData> SMData::GetCTAs(ObjectFileELF *core) const {
  if (m_ctas_loaded)
    return m_ctas;
  m_ctas_loaded = true;

  if (!m_sm_table_node)
    return m_ctas;

  SectionNode *cta_table =
      m_sm_table_node->FindChild(eSectionTypeCUDACtaTable, m_sm_index);
  if (!cta_table)
    return m_ctas;

  m_ctas = ParseTableFromNode<CTAData>(core, cta_table);
  for (uint32_t i = 0; i < m_ctas.size(); ++i)
    m_ctas[i].SetTreeContext(cta_table, i);

  return m_ctas;
}

llvm::ArrayRef<WarpData> CTAData::GetWarps(ObjectFileELF *core) const {
  if (m_warps_loaded)
    return m_warps;
  m_warps_loaded = true;

  if (!m_cta_table_node)
    return m_warps;

  SectionNode *warp_table =
      m_cta_table_node->FindChild(eSectionTypeCUDAWarpTable, m_cta_index);
  if (!warp_table)
    return m_warps;

  m_warps = ParseTableFromNode<WarpData>(core, warp_table);
  for (uint32_t i = 0; i < m_warps.size(); ++i)
    m_warps[i].SetTreeContext(warp_table, i);

  return m_warps;
}

llvm::ArrayRef<LaneData> WarpData::GetLanes(ObjectFileELF *core) const {
  if (m_lanes_loaded)
    return m_lanes;
  m_lanes_loaded = true;

  if (!m_warp_table_node)
    return m_lanes;

  SectionNode *lane_table =
      m_warp_table_node->FindChild(eSectionTypeCUDALaneTable, m_warp_index);
  if (!lane_table)
    return m_lanes;

  m_lanes = ParseTableFromNode<LaneData>(core, lane_table);
  for (uint32_t i = 0; i < m_lanes.size(); ++i)
    m_lanes[i].SetTreeContext(lane_table, i);

  return m_lanes;
}

// ===== Lazy Blob Loading (zero-copy via DataExtractor) =====

const DataExtractor &LaneData::GetRegisters(ObjectFileELF *core) const {
  if (!m_regs_loaded) {
    m_regs_loaded = true;
    if (m_lane_table_node) {
      SectionNode *node = m_lane_table_node->FindChild(
          eSectionTypeCUDARegisters, m_lane_index);
      if (node && node->section)
        core->ReadSectionData(node->section.get(), m_regs);
    }
  }
  return m_regs;
}

const DataExtractor &LaneData::GetPredicates(ObjectFileELF *core) const {
  if (!m_predicates_loaded) {
    m_predicates_loaded = true;
    if (m_lane_table_node) {
      SectionNode *node = m_lane_table_node->FindChild(
          eSectionTypeCUDAPredicates, m_lane_index);
      if (node && node->section)
        core->ReadSectionData(node->section.get(), m_predicates);
    }
  }
  return m_predicates;
}

const DataExtractor &
WarpData::GetUniformRegisters(ObjectFileELF *core) const {
  if (!m_uniform_regs_loaded) {
    m_uniform_regs_loaded = true;
    if (m_warp_table_node) {
      SectionNode *node = m_warp_table_node->FindChild(
          eSectionTypeCUDAUniformRegisters, m_warp_index);
      if (node && node->section)
        core->ReadSectionData(node->section.get(), m_uniform_regs);
    }
  }
  return m_uniform_regs;
}

const DataExtractor &
WarpData::GetUniformPredicates(ObjectFileELF *core) const {
  if (!m_uniform_preds_loaded) {
    m_uniform_preds_loaded = true;
    if (m_warp_table_node) {
      SectionNode *node = m_warp_table_node->FindChild(
          eSectionTypeCUDAUniformPredicates, m_warp_index);
      if (node && node->section)
        core->ReadSectionData(node->section.get(), m_uniform_preds);
    }
  }
  return m_uniform_preds;
}

const MemorySection &LaneData::GetLocalMemory(ObjectFileELF *core) const {
  if (!m_local_mem.loaded) {
    m_local_mem.loaded = true;
    if (m_lane_table_node) {
      SectionNode *node = m_lane_table_node->FindChild(
          eSectionTypeCUDALocalMemory, m_lane_index);
      if (node && node->section && node->header) {
        core->ReadSectionData(node->section.get(), m_local_mem.data);
        m_local_mem.addr = node->header->sh_addr;
        m_local_mem.size = node->header->sh_size;
      }
    }
  }
  return m_local_mem;
}

const MemorySection &CTAData::GetSharedMemory(ObjectFileELF *core) const {
  if (!m_shared_mem.loaded) {
    m_shared_mem.loaded = true;
    if (m_cta_table_node) {
      SectionNode *node = m_cta_table_node->FindChild(
          eSectionTypeCUDASharedMemory, m_cta_index);
      if (node && node->section && node->header) {
        core->ReadSectionData(node->section.get(), m_shared_mem.data);
        m_shared_mem.addr = node->header->sh_addr;
        m_shared_mem.size = node->header->sh_size;
      }
    }
  }
  return m_shared_mem;
}

// ===== NVGPUCoreData Top-Level =====

llvm::Error NVGPUCoreData::ParseTopLevel(ObjectFileELF *core) {
  Log *log = GetLog(LLDBLog::Process);

  if (llvm::Error err = section_tree.Build(core))
    return err;

  // Parse device table
  SectionNode *dev_table_node =
      section_tree.FindRootByType(eSectionTypeCUDADeviceTable);
  if (dev_table_node) {
    devices = ParseTableFromNode<DeviceData>(core, dev_table_node);
    for (uint32_t i = 0; i < devices.size(); ++i)
      devices[i].SetTreeContext(dev_table_node, i);
    LLDB_LOG(log, "NVGPUCoreData: parsed {0} devices", devices.size());
  }

  // Collect global memory regions (lazy reads via CopyData)
  llvm::SmallVector<SectionNode *, 8> global_nodes =
      section_tree.GetRootsByType(eSectionTypeCUDAGlobalMemory);
  llvm::SmallVector<SectionNode *, 8> managed_nodes =
      section_tree.GetRootsByType(eSectionTypeCUDAManagedMemory);
  global_nodes.insert(global_nodes.end(), managed_nodes.begin(),
                      managed_nodes.end());

  for (SectionNode *node : global_nodes) {
    if (!node->header || node->header->sh_size == 0)
      continue;
    MemoryRegion region;
    region.addr = node->header->sh_addr;
    region.size = node->header->sh_size;
    region.file_offset = node->header->sh_offset;
    global_memory.push_back(region);
  }
  LLDB_LOG(log, "NVGPUCoreData: found {0} global memory regions",
           global_memory.size());

  return llvm::Error::success();
}

const MemoryRegion *
NVGPUCoreData::FindGlobalMemoryRegion(uint64_t addr) const {
  for (const MemoryRegion &region : global_memory) {
    if (addr >= region.addr && addr < region.addr + region.size)
      return &region;
  }
  return nullptr;
}
