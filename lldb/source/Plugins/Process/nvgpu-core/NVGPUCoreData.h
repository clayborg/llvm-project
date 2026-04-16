//===-- NVGPUCoreData.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// DOM-like section tree and data classes for NVIDIA GPU corefiles.
///
/// The SectionTree mirrors the GPU hardware hierarchy (device -> SM -> CTA ->
/// warp -> lane) and is built from ELF sh_link/sh_info relationships. Data
/// classes wrap table entry rows and provide lazy access to children and
/// register blobs via the tree.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H

#include "cudacoredump.h"
// cudacoredump.h conditionally defines SHT_LOUSER as a preprocessor macro
// which conflicts with the llvm::ELF::SHT_LOUSER enum value. The CUDBG_SHT_*
// enum values are already expanded by this point, so the undef is safe.
#undef SHT_LOUSER

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lldb_private {

// ===== Section Tree =====

/// A node in the section tree. Each node represents an ELF section from
/// the CUDA corefile. Nodes are linked via sh_link (child -> parent) into
/// a tree that mirrors the GPU hardware hierarchy.
struct SectionNode {
  size_t section_idx = 0;
  lldb::SectionType type = lldb::eSectionTypeOther;
  uint32_t sh_info = 0;
  lldb::SectionSP section;
  const ObjectFileELF::ELFSectionHeaderInfo *header = nullptr;

  SectionNode *parent = nullptr;
  llvm::SmallVector<SectionNode *, 8> children;

  /// Find all children of a specific section type.
  llvm::SmallVector<SectionNode *, 4>
  GetChildrenByType(lldb::SectionType child_type) const;

  /// Find a child by type and sh_info (row index in parent table).
  SectionNode *FindChild(lldb::SectionType child_type,
                         uint32_t info) const;
};

/// Tree of ELF sections from a CUDA corefile, built from sh_link/sh_info.
/// Supports DOM-like queries: "find device table", "get all children of
/// this warp table for warp index 5", etc.
class SectionTree {
public:
  /// Build tree from an ObjectFileELF. One scan of section headers.
  llvm::Error Build(ObjectFileELF *core);

  /// Get all root nodes (sections with sh_link == 0).
  llvm::ArrayRef<SectionNode *> GetRootNodes() const { return m_roots; }

  /// Find the first root node of a given type.
  SectionNode *FindRootByType(lldb::SectionType type) const;

  /// Get all root nodes of a given type.
  llvm::SmallVector<SectionNode *, 8> GetRootsByType(lldb::SectionType type) const;

  /// Find all nodes of a given type anywhere in the tree.
  llvm::SmallVector<SectionNode *, 16> FindAllByType(lldb::SectionType type) const;

private:
  std::vector<std::unique_ptr<SectionNode>> m_nodes;
  llvm::SmallVector<SectionNode *, 16> m_roots;
};

// ===== Data Classes =====

/// Coordinates identifying a single GPU lane (thread) within the hierarchy.
struct NVGPULaneCoords {
  uint32_t dev_idx;
  uint32_t sm_idx;
  uint32_t cta_idx;
  uint32_t warp_idx;
  uint32_t lane_idx;
};

/// Lazily-loaded memory section with address range for DoReadMemory lookups.
struct MemorySection {
  DataExtractor data;
  uint64_t addr = 0;
  uint64_t size = 0;
  bool loaded = false;
};

/// Wraps a CudbgThreadTableEntry row with lazy access to per-lane data.
class LaneData {
public:
  CudbgThreadTableEntry entry = {};

  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Lazy register data. Finds the CUDBG_SHT_DEV_REGS child section
  /// for this lane index and reads via ReadSectionData. Returns a
  /// DataExtractor referencing the ObjectFile's mapped data (no copy).
  const DataExtractor &GetRegisters(ObjectFileELF *core) const;

  /// Lazy predicate data.
  const DataExtractor &GetPredicates(ObjectFileELF *core) const;

  /// Lazy local memory data with address range for DoReadMemory lookups.
  const MemorySection &GetLocalMemory(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *lane_table_node, uint32_t lane_index) {
    m_lane_table_node = lane_table_node;
    m_lane_index = lane_index;
  }

private:
  SectionNode *m_lane_table_node = nullptr;
  uint32_t m_lane_index = 0;
  mutable DataExtractor m_regs;
  mutable DataExtractor m_predicates;
  mutable bool m_regs_loaded = false;
  mutable bool m_predicates_loaded = false;
  mutable MemorySection m_local_mem;
};

/// Wraps a CudbgWarpTableEntry row with lazy access to lanes and registers.
class WarpData {
public:
  CudbgWarpTableEntry entry = {};

  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Lazily load lanes from the lane table child section.
  llvm::ArrayRef<LaneData> GetLanes(ObjectFileELF *core) const;

  /// Lazy uniform register data (no copy -- references ObjectFile data).
  const DataExtractor &GetUniformRegisters(ObjectFileELF *core) const;

  /// Lazy uniform predicate data (no copy).
  const DataExtractor &GetUniformPredicates(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *warp_table_node, uint32_t warp_index) {
    m_warp_table_node = warp_table_node;
    m_warp_index = warp_index;
  }

private:
  mutable bool m_lanes_loaded = false;
  mutable std::vector<LaneData> m_lanes;
  SectionNode *m_warp_table_node = nullptr;
  uint32_t m_warp_index = 0;
  mutable DataExtractor m_uniform_regs;
  mutable DataExtractor m_uniform_preds;
  mutable bool m_uniform_regs_loaded = false;
  mutable bool m_uniform_preds_loaded = false;
};

/// Wraps a CudbgCTATableEntry row with lazy access to warps.
class CTAData {
public:
  CudbgCTATableEntry entry = {};

  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Lazily load warps from the warp table child section.
  llvm::ArrayRef<WarpData> GetWarps(ObjectFileELF *core) const;

  /// Lazy shared memory data with address range for DoReadMemory lookups.
  const MemorySection &GetSharedMemory(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *cta_table_node, uint32_t cta_index) {
    m_cta_table_node = cta_table_node;
    m_cta_index = cta_index;
  }

private:
  mutable bool m_warps_loaded = false;
  mutable std::vector<WarpData> m_warps;
  SectionNode *m_cta_table_node = nullptr;
  uint32_t m_cta_index = 0;
  mutable MemorySection m_shared_mem;
};

/// Wraps a CudbgSmTableEntry row with lazy access to CTAs.
class SMData {
public:
  CudbgSmTableEntry entry = {};

  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Lazily load CTAs from the CTA table child section.
  llvm::ArrayRef<CTAData> GetCTAs(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *sm_table_node, uint32_t sm_index) {
    m_sm_table_node = sm_table_node;
    m_sm_index = sm_index;
  }

private:
  mutable bool m_ctas_loaded = false;
  mutable std::vector<CTAData> m_ctas;
  SectionNode *m_sm_table_node = nullptr;
  uint32_t m_sm_index = 0;
};

/// Wraps a CudbgDeviceTableEntry row with lazy access to SMs/contexts/grids.
class DeviceData {
public:
  CudbgDeviceTableEntry entry = {};

  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Lazily load SMs from the SM table child section.
  llvm::ArrayRef<SMData> GetSMs(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *dev_table_node, uint32_t dev_index) {
    m_dev_table_node = dev_table_node;
    m_dev_index = dev_index;
  }

private:
  mutable bool m_sms_loaded = false;
  mutable std::vector<SMData> m_sms;
  SectionNode *m_dev_table_node = nullptr;
  uint32_t m_dev_index = 0;
};

/// Global memory region (address + file offset for lazy CopyData reads).
struct MemoryRegion {
  uint64_t addr;
  uint64_t size;
  uint64_t file_offset;
};

/// Top-level corefile data. Holds the section tree, device list, and
/// global memory regions. Tables are parsed lazily via the tree.
struct NVGPUCoreData {
  SectionTree section_tree;
  std::vector<DeviceData> devices;
  llvm::SmallVector<MemoryRegion, 16> global_memory;

  /// Parse the device table and global memory regions. SM/CTA/warp/lane
  /// tables are deferred until accessed via DeviceData::GetSMs() etc.
  llvm::Error ParseTopLevel(ObjectFileELF *core);

  /// Find a global memory region containing the given address.
  const MemoryRegion *FindGlobalMemoryRegion(uint64_t addr) const;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H
