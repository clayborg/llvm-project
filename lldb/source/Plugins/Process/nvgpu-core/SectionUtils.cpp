//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SectionUtils.h"

#include "lldb/Core/Section.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private::nvgpu_core {

llvm::SmallVector<SectionSP, 8> FindChildrenByType(const Section &parent,
                                                   SectionType type) {
  llvm::SmallVector<SectionSP, 8> result;
  for (const SectionSP &child : parent.GetChildren()) {
    if (child && child->GetType() == type)
      result.push_back(child);
  }
  return result;
}

SectionSP FindChildByType(const Section &parent, SectionType type) {
  for (const SectionSP &child : parent.GetChildren()) {
    if (child && child->GetType() == type)
      return child;
  }
  return nullptr;
}

llvm::SmallVector<SectionSP, 16> FindDescendantsByType(const Section &root,
                                                       SectionType type) {
  llvm::SmallVector<SectionSP, 16> result;
  auto walk = [&](const Section &node, auto &self) -> void {
    for (const SectionSP &child : node.GetChildren()) {
      if (!child)
        continue;
      if (child->GetType() == type)
        result.push_back(child);
      self(*child, self);
    }
  };
  walk(root, walk);
  return result;
}

} // namespace lldb_private::nvgpu_core
