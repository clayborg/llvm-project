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

// Brings in cudadebugger.h (for the CUDBG_API_VERSION_* macros that gate the
// version-specific field reads below) and the single-major static_assert.
#include "lldb/Utility/NVGPU/CUDADebuggerVersion.h"

#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/NVGPU/ThreadName.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <optional>
#include <string>

namespace lldb_private::nvgpu_core {

/// Version of the driver/toolkit that produced a coredump, decoded from the
/// `CudbgMetaDataEntry` metadata section. Used for diagnostics (logging the
/// producer version and warning on a cross-major coredump).
struct ProducerInfo {
  /// Driver branch major (e.g. 575/580/615), matching the "Since CUDA
  /// driver rXXX" annotations in cudacoredump.h.
  uint32_t driver_branch = 0;
  /// CUDA driver version (e.g. 13/0) reported by the producer.
  uint32_t cuda_major = 0;
  uint32_t cuda_minor = 0;
};

/// One row of a nvgpu-device-table.
struct DeviceEntry : CudbgDeviceTableEntry {
  static llvm::Expected<DeviceEntry> Decode(const DataExtractor &data,
                                            lldb::offset_t *offset_ptr,
                                            uint64_t entry_size);
};

/// One row of a nvgpu-sm-table.
struct SMEntry : CudbgSmTableEntry {
  static llvm::Expected<SMEntry> Decode(const DataExtractor &data,
                                        lldb::offset_t *offset_ptr,
                                        uint64_t entry_size);
};

/// One row of a nvgpu-grid-table.
struct GridEntry : CudbgGridTableEntry {
  static llvm::Expected<GridEntry> Decode(const DataExtractor &data,
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

/// One row of a nvgpu-constbank-table: a single constant bank's global
/// address, size, and bank id.
struct ConstBankEntry : CudbgConstBankTableEntry {
  static llvm::Expected<ConstBankEntry> Decode(const DataExtractor &data,
                                               lldb::offset_t *offset_ptr,
                                               uint64_t entry_size);

  /// True if `addr` falls within this bank's [addr, addr + size) range.
  bool Contains(lldb::addr_t query_addr) const {
    return query_addr >= addr && query_addr - addr < size;
  }
};

/// Decode the producer version from the raw bytes of the coredump metadata
/// section. Returns std::nullopt if `data` is empty (no metadata section).
std::optional<ProducerInfo> DecodeProducerInfo(const DataExtractor &data);

/// One row of a per-lane nvgpu-backtrace section.
struct BacktraceEntry : CudbgBacktraceTableEntry {
  static llvm::Expected<BacktraceEntry> Decode(const DataExtractor &data,
                                               lldb::offset_t *offset_ptr,
                                               uint64_t entry_size);
};

/// Read `section_sp`'s data window from `core` and decode it into an
/// `EntryT` (one of the wrapper structs above). Returns the parse error if
/// either input is missing or `EntryT::Decode` fails.
template <typename EntryT>
llvm::Expected<EntryT> ReadAndDecode(lldb::SectionSP section_sp,
                                     ObjectFile *core) {
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

/// Why a corefile lane stopped. Most lanes were merely suspended when
/// the dump was taken and carry no stop reason; the fields below are
/// the ways a lane can claim one.
struct StopAttribution {
  /// `CUDBGException_t` attributed to this lane, or 0 if none.
  uint32_t attributed_exception;
  /// Lane stopped at an inline `trap;` / `__trap()`. The CUDA SDK
  /// reports this via `warp.isWarpBroken` rather than an exception
  /// code, gated by the warp's active-lane mask so unrelated lanes on
  /// the same warp don't claim the trap.
  bool at_trap;
  /// If the warp or SM row failed to decode (typically a truncated
  /// corefile), a human-readable description of the failure. Surfaced
  /// as the lane's stop reason so the user sees the corruption in
  /// `thread list` instead of the lane silently appearing healthy.
  std::string decode_error;
};

/// Compute stop attribution for `lane_idx`, or `std::nullopt` if the
/// lane has no stop reason. Exception precedence:
///
///   1. `lane.exception` is definitive when non-zero.
///   2. Otherwise, if `warp.errorPCValid` AND the lane was active at
///      fault time, borrow `sm.exception`.
///   3. Otherwise no exception.
///
/// Warp/SM read failures populate `decode_error` (and are also logged
/// via the Process log channel) so the failure surfaces to the user as
/// a stop reason rather than silently disappearing.
std::optional<StopAttribution>
ComputeStopAttribution(const LaneEntry &lane, uint32_t lane_idx,
                       lldb::SectionSP warp_section_sp,
                       lldb::SectionSP sm_section_sp, ObjectFile *core);

} // namespace lldb_private::nvgpu_core

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_CUDBGENTRYPARSER_H
