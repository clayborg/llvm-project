//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Plugin-private helpers for navigating the synthetic NVGPU section
/// hierarchy by type.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_SECTIONUTILS_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_SECTIONUTILS_H

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private::nvgpu_core {

/// Return all direct children of `parent` whose section type is `type`,
/// preserving insertion order. Materializes the matches into a small vector
/// so callers can use range-based for loops without iterator hassles.
///
/// \param[in] parent
///     The container section whose children to scan.
///
/// \param[in] type
///     The `SectionType` to match.
///
/// \return
///     A small vector of matching children, possibly empty.
llvm::SmallVector<lldb::SectionSP, 8>
FindChildrenByType(const Section &parent, lldb::SectionType type);

/// Return the first direct child of `parent` whose section type is `type`.
///
/// \param[in] parent
///     The container section whose children to scan.
///
/// \param[in] type
///     The `SectionType` to match.
///
/// \return
///     The first matching child, or null if none exists.
lldb::SectionSP FindChildByType(const Section &parent, lldb::SectionType type);

/// Return every descendant of `root` whose section type is `type`,
/// walked pre-order, depth-first. Iteration order at each level matches
/// the child insertion order.
///
/// Use this when only one level of the synthetic tree is interesting to
/// the caller (e.g. "every warp" or "every lane") and the intermediate
/// container nesting is just structural narration. For direct parent ->
/// child lookups prefer `FindChildrenByType` / `FindChildByType` instead.
///
/// \param[in] root
///     The root section whose subtree to walk.
///
/// \param[in] type
///     The `SectionType` to match.
///
/// \return
///     A small vector of matching descendants, possibly empty.
llvm::SmallVector<lldb::SectionSP, 16>
FindDescendantsByType(const Section &root, lldb::SectionType type);

} // namespace lldb_private::nvgpu_core

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_SECTIONUTILS_H
