//===-- NVGPUCoreData.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Section index and data classes for NVIDIA GPU corefiles.
///
/// The SectionTree indexes ELF sections by sh_link/sh_info parent-child
/// relationships, mirroring the GPU hardware hierarchy (device -> SM ->
/// CTA -> warp -> lane). Data classes wrap table entry rows and provide
/// lazy access to children and register/memory blobs.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H

#include "cudacoredump.h"
// cudacoredump.h conditionally defines SHT_LOUSER as a preprocessor macro
// which conflicts with the llvm::ELF::SHT_LOUSER enum value. The CUDBG_SHT_*
// enum values are already expanded by this point, so the undef is safe.
#undef SHT_LOUSER

#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <vector>

class ObjectFileELF;

namespace lldb_private {

// ===== Section Tree =====

/// A node in the section tree. Each node represents an ELF section from
/// the CUDA corefile. Nodes are linked via sh_link (child -> parent) into
/// a tree that mirrors the GPU hardware hierarchy.
struct SectionNode {
  lldb::SectionType type = lldb::eSectionTypeOther;
  uint32_t sh_info = 0;
  uint64_t sh_entsize = 0;
  lldb::SectionSP section;
  llvm::SmallVector<SectionNode *, 8> children;

  /// Find a child node matching the given section type and sh_info value.
  ///
  /// \param[in] type
  ///     The lldb::SectionType of the child to find.
  ///
  /// \param[in] info
  ///     The sh_info value (parent row index) the child should reference.
  ///
  /// \return
  ///     The matching child node, or nullptr if no child matches.
  SectionNode *FindChild(lldb::SectionType type, uint32_t info) const;
};

/// Index of ELF sections from a CUDA corefile, built from sh_link/sh_info
/// parent-child relationships. Provides lookup by section type and
/// parent-to-child navigation for the GPU hardware hierarchy.
class SectionTree {
public:
  /// Build the section index by scanning the ObjectFile's section headers
  /// and linking them via sh_link parent-child relationships. Sections
  /// whose data extends beyond the file size are skipped (truncated
  /// corefile); their names are recorded for retrieval via
  /// GetTruncatedSections.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile to scan.
  ///
  /// \return
  ///     An error only if scanning fails outright; truncated sections
  ///     are non-fatal and reported separately.
  llvm::Error Parse(ObjectFileELF *core);

  /// Get all root nodes of a given type (sh_link == 0).
  llvm::SmallVector<SectionNode *, 8>
  GetRootsByType(lldb::SectionType type) const;

  /// Get all nodes of a given type, regardless of root status.
  llvm::SmallVector<SectionNode *, 8>
  GetNodesByType(lldb::SectionType type) const;

  /// Names of sections that were truncated (beyond file size).
  llvm::ArrayRef<std::string> GetTruncatedSections() const {
    return m_truncated_sections;
  }

private:
  std::vector<std::unique_ptr<SectionNode>> m_nodes;
  llvm::SmallVector<SectionNode *, 16> m_roots;
  llvm::SmallVector<std::string, 0> m_truncated_sections;
};

// ===== Data Classes =====

/// Coordinates identifying a single GPU lane (thread) within the hierarchy.
struct NVGPULaneCoords {
  uint32_t dev_idx = 0;
  uint32_t sm_idx = 0;
  uint32_t cta_idx = 0;
  uint32_t warp_idx = 0;
  uint32_t lane_idx = 0;
};

/// Memory section with its address range, used for DoReadMemory lookups.
/// Lazy-load state lives on the owning class as a sibling `m_*_loaded` flag.
struct MemorySection {
  DataExtractor data;
  uint64_t addr = 0;
  uint64_t size = 0;
};

/// Wraps a CudbgThreadTableEntry row with lazy access to per-lane data.
class LaneData {
public:
  /// Parse a `CudbgThreadTableEntry` row from the given data slice.
  /// Reads past the slice return zero, so older corefile versions that
  /// omit later-added fields naturally default those fields to zero.
  ///
  /// \param[in] data
  ///     A DataExtractor sliced to this entry's bytes within the corefile.
  ///
  /// \param[in] entry_size
  ///     The on-disk size of the entry (CUDA driver version dependent).
  ///
  /// \return
  ///     An error if the data slice is shorter than `entry_size`; success
  ///     otherwise.
  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Get the parsed thread table row (raw binary fields from the corefile).
  const CudbgThreadTableEntry &GetEntry() const { return m_entry; }

  /// Lazy zero-copy access to this lane's register data. Finds the
  /// CUDBG_SHT_DEV_REGS child section on first call and returns a
  /// DataExtractor view aliasing the corefile mmap; subsequent calls
  /// return the cached view.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A DataExtractor view of the lane's register bytes, or an empty
  ///     view if the section doesn't exist.
  const DataExtractor &GetRegisters(ObjectFileELF *core) const;

  /// Lazy zero-copy access to this lane's predicate register data.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A DataExtractor view of the lane's predicate bytes, or an empty
  ///     view if the section doesn't exist.
  const DataExtractor &GetPredicates(ObjectFileELF *core) const;

  /// Lazy zero-copy access to this lane's local memory section.
  /// Includes the address range so `ProcessNVGPUCore::DoReadMemory` can
  /// match against it.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A `MemorySection` with the local memory data, address, and size;
  ///     a default-constructed `MemorySection` (size == 0) if the section
  ///     doesn't exist.
  const MemorySection &GetLocalMemory(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *lane_table_node, uint32_t lane_idx) {
    m_lane_table_node = lane_table_node;
    m_lane_idx = lane_idx;
  }

private:
  CudbgThreadTableEntry m_entry = {};
  SectionNode *m_lane_table_node = nullptr;
  uint32_t m_lane_idx = 0;
  mutable DataExtractor m_regs;
  mutable DataExtractor m_predicates;
  mutable MemorySection m_local_mem;
  mutable bool m_regs_loaded = false;
  mutable bool m_predicates_loaded = false;
  mutable bool m_local_mem_loaded = false;
};

/// Wraps a CudbgWarpTableEntry row with lazy access to lanes and registers.
class WarpData {
public:
  /// Parse a `CudbgWarpTableEntry` row from the given data slice.
  /// Reads past the slice return zero, so older corefile versions that
  /// omit later-added fields naturally default those fields to zero.
  ///
  /// \param[in] data
  ///     A DataExtractor sliced to this entry's bytes within the corefile.
  ///
  /// \param[in] entry_size
  ///     The on-disk size of the entry (CUDA driver version dependent).
  ///
  /// \return
  ///     An error if the data slice is shorter than `entry_size`; success
  ///     otherwise.
  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Get the parsed warp table row (raw binary fields from the corefile).
  const CudbgWarpTableEntry &GetEntry() const { return m_entry; }

  /// Lazily load this warp's lane table on first call; subsequent calls
  /// return the cached vector.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to parse the
  ///     section).
  ///
  /// \return
  ///     The lanes belonging to this warp, or an empty range if no lane
  ///     table child section exists.
  llvm::ArrayRef<LaneData> GetLanes(ObjectFileELF *core) const;

  /// Lazy zero-copy access to this warp's uniform register data.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A DataExtractor view of the warp's uniform register bytes, or
  ///     an empty view if the section doesn't exist.
  const DataExtractor &GetUniformRegisters(ObjectFileELF *core) const;

  /// Lazy zero-copy access to this warp's uniform predicate register data.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A DataExtractor view of the warp's uniform predicate bytes, or
  ///     an empty view if the section doesn't exist.
  const DataExtractor &GetUniformPredicates(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *warp_table_node, uint32_t warp_idx) {
    m_warp_table_node = warp_table_node;
    m_warp_idx = warp_idx;
  }

private:
  CudbgWarpTableEntry m_entry = {};
  SectionNode *m_warp_table_node = nullptr;
  uint32_t m_warp_idx = 0;
  mutable std::vector<LaneData> m_lanes;
  mutable DataExtractor m_uniform_regs;
  mutable DataExtractor m_uniform_preds;
  mutable bool m_lanes_loaded = false;
  mutable bool m_uniform_regs_loaded = false;
  mutable bool m_uniform_preds_loaded = false;
};

/// Wraps a CudbgCTATableEntry row with lazy access to warps.
class CTAData {
public:
  /// Parse a `CudbgCTATableEntry` row from the given data slice.
  /// Reads past the slice return zero, so older corefile versions that
  /// omit later-added fields naturally default those fields to zero.
  ///
  /// \param[in] data
  ///     A DataExtractor sliced to this entry's bytes within the corefile.
  ///
  /// \param[in] entry_size
  ///     The on-disk size of the entry (CUDA driver version dependent).
  ///
  /// \return
  ///     An error if the data slice is shorter than `entry_size`; success
  ///     otherwise.
  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Get the parsed CTA table row (raw binary fields from the corefile).
  const CudbgCTATableEntry &GetEntry() const { return m_entry; }

  /// Lazily load this CTA's warp table on first call; subsequent calls
  /// return the cached vector.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to parse the
  ///     section).
  ///
  /// \return
  ///     The warps belonging to this CTA, or an empty range if no warp
  ///     table child section exists.
  llvm::ArrayRef<WarpData> GetWarps(ObjectFileELF *core) const;

  /// Lazy zero-copy access to this CTA's shared memory section.
  /// Includes the address range so `ProcessNVGPUCore::DoReadMemory` can
  /// match against it.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to read the
  ///     section).
  ///
  /// \return
  ///     A `MemorySection` with the shared memory data, address, and
  ///     size; a default-constructed `MemorySection` (size == 0) if the
  ///     section doesn't exist.
  const MemorySection &GetSharedMemory(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *cta_table_node, uint32_t cta_idx) {
    m_cta_table_node = cta_table_node;
    m_cta_idx = cta_idx;
  }

private:
  CudbgCTATableEntry m_entry = {};
  SectionNode *m_cta_table_node = nullptr;
  uint32_t m_cta_idx = 0;
  mutable std::vector<WarpData> m_warps;
  mutable MemorySection m_shared_mem;
  mutable bool m_warps_loaded = false;
  mutable bool m_shared_mem_loaded = false;
};

/// Wraps a CudbgSmTableEntry row with lazy access to CTAs.
class SMData {
public:
  /// Parse a `CudbgSmTableEntry` row from the given data slice.
  /// Reads past the slice return zero, so older corefile versions that
  /// omit later-added fields naturally default those fields to zero.
  ///
  /// \param[in] data
  ///     A DataExtractor sliced to this entry's bytes within the corefile.
  ///
  /// \param[in] entry_size
  ///     The on-disk size of the entry (CUDA driver version dependent).
  ///
  /// \return
  ///     An error if the data slice is shorter than `entry_size`; success
  ///     otherwise.
  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Get the parsed SM table row (raw binary fields from the corefile).
  const CudbgSmTableEntry &GetEntry() const { return m_entry; }

  /// Lazily load this SM's CTA table on first call; subsequent calls
  /// return the cached vector.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to parse the
  ///     section).
  ///
  /// \return
  ///     The CTAs belonging to this SM, or an empty range if no CTA
  ///     table child section exists.
  llvm::ArrayRef<CTAData> GetCTAs(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *sm_table_node, uint32_t sm_idx) {
    m_sm_table_node = sm_table_node;
    m_sm_idx = sm_idx;
  }

private:
  CudbgSmTableEntry m_entry = {};
  SectionNode *m_sm_table_node = nullptr;
  uint32_t m_sm_idx = 0;
  mutable std::vector<CTAData> m_ctas;
  mutable bool m_ctas_loaded = false;
};

/// Wraps a CudbgDeviceTableEntry row with lazy access to SMs/contexts/grids.
class DeviceData {
public:
  /// Parse a `CudbgDeviceTableEntry` row from the given data slice.
  /// Reads past the slice return zero, so older corefile versions that
  /// omit later-added fields naturally default those fields to zero.
  ///
  /// \param[in] data
  ///     A DataExtractor sliced to this entry's bytes within the corefile.
  ///
  /// \param[in] entry_size
  ///     The on-disk size of the entry (CUDA driver version dependent).
  ///
  /// \return
  ///     An error if the data slice is shorter than `entry_size`; success
  ///     otherwise.
  llvm::Error Parse(const DataExtractor &data, uint64_t entry_size);

  /// Get the parsed device table row (raw binary fields from the corefile).
  const CudbgDeviceTableEntry &GetEntry() const { return m_entry; }

  /// Lazily load this device's SM table on first call; subsequent calls
  /// return the cached vector.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile (used on first call to parse the
  ///     section).
  ///
  /// \return
  ///     The SMs belonging to this device, or an empty range if no SM
  ///     table child section exists.
  llvm::ArrayRef<SMData> GetSMs(ObjectFileELF *core) const;

  void SetTreeContext(SectionNode *dev_table_node, uint32_t dev_idx) {
    m_dev_table_node = dev_table_node;
    m_dev_idx = dev_idx;
  }

private:
  CudbgDeviceTableEntry m_entry = {};
  SectionNode *m_dev_table_node = nullptr;
  uint32_t m_dev_idx = 0;
  mutable std::vector<SMData> m_sms;
  mutable bool m_sms_loaded = false;
};

/// Global memory region (address + file offset for lazy CopyData reads).
struct MemoryRegion {
  uint64_t addr = 0;
  uint64_t size = 0;
  uint64_t file_offset = 0;
};

/// File range for an embedded cubin module within the corefile.
struct CubinLocation {
  uint64_t file_offset = 0;
  uint64_t file_size = 0;
};

/// Top-level corefile data. Holds the section index, device list, and
/// global memory regions. Tables are parsed lazily via the tree.
class NVGPUCoreData {
public:
  /// Parse the top-level corefile data: section tree, device table,
  /// global memory regions, and cubin module locations. SM/CTA/warp/lane
  /// tables are deferred and parsed lazily via DeviceData::GetSMs() etc.
  ///
  /// \param[in] core
  ///     The CUDA corefile ObjectFile to parse.
  ///
  /// \return
  ///     An error if the section tree fails to build; success otherwise
  ///     (missing or truncated subsections are non-fatal).
  llvm::Error Parse(ObjectFileELF *core);

  /// Get the parsed device list.
  llvm::ArrayRef<DeviceData> GetDevices() const { return m_devices; }

  /// Get a device by index.
  const DeviceData &GetDevice(size_t idx) const { return m_devices[idx]; }

  /// Get the number of devices.
  size_t GetNumDevices() const { return m_devices.size(); }

  /// Get the cubin module locations within the corefile.
  llvm::ArrayRef<CubinLocation> GetCubinLocations() const {
    return m_cubin_locations;
  }

  /// Find the global memory region (if any) that contains a given GPU
  /// address.
  ///
  /// \param[in] addr
  ///     The GPU global memory address to look up.
  ///
  /// \return
  ///     The `MemoryRegion` whose `[addr, addr + size)` range contains
  ///     `addr`, or nullptr if no region matches.
  const MemoryRegion *FindGlobalMemoryRegion(uint64_t addr) const;

  /// Names of sections that were truncated (beyond file size).
  llvm::ArrayRef<std::string> GetTruncatedSections() const {
    return m_section_tree.GetTruncatedSections();
  }

private:
  SectionTree m_section_tree;
  std::vector<DeviceData> m_devices;
  llvm::SmallVector<MemoryRegion, 16> m_global_memory;
  llvm::SmallVector<CubinLocation, 16> m_cubin_locations;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_NVGPUCOREDATA_H
