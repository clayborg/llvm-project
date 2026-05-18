//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Self-decoding wrappers around the `Cudbg*TableEntry` row types from
/// `cudacoredump.h`.
///
/// Each wrapper inherits from its corresponding SDK struct (so all the row
/// fields are reachable directly, e.g. `entry.validLanesMask`) and adds a
/// `static Decode(DataExtractor, entry_size) -> llvm::Expected<Self>` method
/// that constructs a fully-populated entry from a sliced
/// DataExtractor. Reads past the slice return zero, so older corefile
/// versions that omit later-added fields naturally default those fields to
/// zero.
///
/// This header is the only place in the corefile plugin that includes
/// `cudacoredump.h`; everything else navigates the section tree via
/// `lldb_private::Section`.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_CUDBGENTRYPARSER_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_CUDBGENTRYPARSER_H

#include "cudacoredump.h"
// cudacoredump.h conditionally defines SHT_LOUSER as a preprocessor macro
// which conflicts with the llvm::ELF::SHT_LOUSER enum value. Undef it here
// since the CudbgXxxTableEntry types are already declared.
#undef SHT_LOUSER

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Section.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/ThreadName.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <string>

namespace lldb_private::nvgpu_core {

/// One row of a nvgpu-device-table.
struct DeviceEntry : CudbgDeviceTableEntry {
  static llvm::Expected<DeviceEntry>
  Decode(const DataExtractor &data, lldb::offset_t *offset_ptr,
         uint64_t entry_size);
};

/// One row of a nvgpu-sm-table.
struct SMEntry : CudbgSmTableEntry {
  static llvm::Expected<SMEntry> Decode(const DataExtractor &data,
                                        lldb::offset_t *offset_ptr,
                                        uint64_t entry_size);
};

/// One row of a nvgpu-cta-table.
struct CTAEntry : CudbgCTATableEntry {
  static llvm::Expected<CTAEntry> Decode(const DataExtractor &data,
                                         lldb::offset_t *offset_ptr,
                                         uint64_t entry_size);
};

/// One row of a nvgpu-warp-table.
struct WarpEntry : CudbgWarpTableEntry {
  static llvm::Expected<WarpEntry> Decode(const DataExtractor &data,
                                          lldb::offset_t *offset_ptr,
                                          uint64_t entry_size);

  /// True if `lane_idx` was a valid lane at dump time (i.e. had real
  /// state). `lane_idx` must be in [0, 32); guaranteed by the
  /// producer-side `num_lanes > 32` reject in
  /// `ObjectFileELF::BuildNVGPUSectionList`.
  bool IsLaneValid(uint32_t lane_idx) const {
    return validLanesMask & (1u << lane_idx);
  }

  /// True if `lane_idx` was active at the moment this warp's `errorPC`
  /// was captured. Caller is responsible for first checking
  /// `errorPCValid`. Same precondition on `lane_idx` as `IsLaneValid`.
  bool IsLaneActive(uint32_t lane_idx) const {
    return activeLanesMask & (1u << lane_idx);
  }
};

/// One row of a nvgpu-lane (thread) table.
struct LaneEntry : CudbgThreadTableEntry {
  static llvm::Expected<LaneEntry> Decode(const DataExtractor &data,
                                          lldb::offset_t *offset_ptr,
                                          uint64_t entry_size);
};

/// Read `section_sp`'s data window from `core` and decode it into an
/// `EntryT` (one of the wrapper structs above). Returns the parse error if
/// either input is missing or `EntryT::Decode` fails.
template <typename EntryT>
llvm::Expected<EntryT> ReadAndDecode(lldb::SectionSP section_sp,
                                     ObjectFileELF *core) {
  if (!section_sp || !core)
    return llvm::createStringError("missing section or core object file");
  DataExtractor data;
  core->ReadSectionData(section_sp.get(), data);
  lldb::offset_t off = 0;
  return EntryT::Decode(data, &off, section_sp->GetFileSize());
}

/// Compose a human-readable thread name from a CTA + lane row pair, of
/// the form "blockIdx(x=0 y=0 z=0) threadIdx(x=3 y=0 z=0)". Delegates to
/// `nvgpu::FormatThreadName` for the actual formatting; this overload
/// just exists so consumer code doesn't have to unpack the six SDK
/// fields by hand.
std::string FormatThreadName(const CTAEntry &cta, const LaneEntry &lane);

/// Compute the attributed exception code for `lane_idx` using the
/// standard precedence cascade:
///
///   1. Per-lane exception (`lane.exception`) is definitive when non-zero.
///      A lane that didn't execute the faulting instruction can't carry
///      an exception from it.
///   2. Otherwise, if this lane's warp caused the SM fault
///      (`warp.errorPCValid`) AND this lane was active at fault time
///      (`warp.IsLaneActive(lane_idx)`), borrow the kind from
///      `sm.exception`.
///   3. Otherwise 0 (`CUDBG_EXCEPTION_NONE`).
///
/// Returns 0 on any read/decode failure of the warp or SM rows; errors
/// are consumed internally so callers don't have to handle `Expected`.
uint32_t ComputeAttributedException(const LaneEntry &lane, uint32_t lane_idx,
                                    lldb::SectionSP warp_section_sp,
                                    lldb::SectionSP sm_section_sp,
                                    ObjectFileELF *core);

} // namespace lldb_private::nvgpu_core

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_CUDBGENTRYPARSER_H
