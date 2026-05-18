//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// `Section::user_id` encoding for the synthetic NVGPU corefile hierarchy
/// built by `ObjectFileELF::CreateSections` when `e_machine == EM_CUDA &&
/// GetType() == eTypeCoreFile`.
///
/// `Section::user_id` packs three fields:
///
///   * top byte (bits 56-63): the kind tag (root / device / sm / cta /
///     warp / lane / leaf), used for diagnostics and as a sanity check.
///   * middle 40 bits (bits 16-55): a kind-scoped sequence number assigned
///     by the builder so siblings *across the file* get distinct IDs and
///     `Section::user_id` stays globally unique within the section list.
///   * low 16 bits (bits 0-15): the row index in the parent table -- e.g.
///     the lane index 0..31 within the parent warp, or the SM index within
///     the parent device. `DecodeHwIdx` reads it back.
///
/// Sequence and row index are **independent**: the sequence guarantees
/// uniqueness across all siblings of a kind everywhere in the tree, while
/// the row index identifies *position within parent* (which can repeat
/// across different parents -- e.g. lane 0 of warp 0 and lane 0 of warp 1
/// share a row index but have distinct sequences).
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_NVGPUSECTIONID_H
#define LLDB_UTILITY_NVGPU_NVGPUSECTIONID_H

#include <cstdint>

namespace lldb_private::nvgpu {

/// Kind tags occupy the high byte of `Section::user_id`. Used for
/// diagnostics and as a sanity check; downstream code keys off
/// `Section::GetType()` (the LLDB-level `SectionType`) for behavior.
enum class SectionKind : uint8_t {
  Root = 0xC0,   ///< nvgpu-root container
  Device = 0xC1, ///< per-device nvgpu-device container
  Sm = 0xC2,     ///< per-SM nvgpu-sm container
  Cta = 0xC3,    ///< per-CTA nvgpu-cta container
  Warp = 0xC4,   ///< per-warp nvgpu-warp container
  Lane = 0xC5,   ///< per-lane nvgpu-lane container
  Leaf = 0xC6,   ///< all other leaves (uregs/upreds/cbarrier per warp,
                 ///< shared per CTA, regs/preds/local per lane,
                 ///< global/managed memory, cubin images)
};

/// Bit position of the kind tag in `user_id`.
constexpr unsigned kSectionKindShift = 56;

/// Width (in bits) of the row-index field at the bottom of `user_id`;
/// also the bit position at which the unique sequence number starts.
constexpr unsigned kHwIdxBits = 16;

/// Build a `Section::user_id` for any synthetic NVGPU section. See the
/// file-level comment for the bit-layout.
///
/// \param[in] kind
///     The section's kind tag (`Root`, `Device`, ..., `Leaf`).
///
/// \param[in] unique_seq
///     A monotonically-increasing sequence number scoped to `kind`,
///     assigned by the builder so all siblings of the same kind get
///     distinct user_ids regardless of where they sit in the tree.
///
/// \param[in] hw_idx
///     The row index in the parent table (lane 0..31 in a warp, sm
///     index in a device, etc.). Defaults to 0 for kinds with no
///     meaningful row index (`Root`, `Leaf`).
inline uint64_t MakeID(SectionKind kind, uint64_t unique_seq,
                       size_t hw_idx = 0) {
  return (static_cast<uint64_t>(kind) << kSectionKindShift) |
         (unique_seq << kHwIdxBits) | static_cast<uint64_t>(hw_idx);
}

/// Recover the row index (lane 0..31, sm index, etc.) from a `MakeID`-
/// tagged `user_id`. Returns the low 16 bits regardless of kind; for
/// `Root` and `Leaf` containers (which pass `hw_idx = 0`) the result is
/// always 0. Caller is responsible for ensuring the `user_id` came from
/// a kind where the row index is meaningful.
inline uint16_t DecodeHwIdx(uint64_t user_id) {
  return static_cast<uint16_t>(user_id);
}

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_NVGPUSECTIONID_H
