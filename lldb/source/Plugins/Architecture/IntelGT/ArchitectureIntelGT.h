//===-- ArchitectureIntelGT.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ARCHITECTURE_INTELGT_ARCHITECTUREINTELGT_H
#define LLDB_SOURCE_PLUGINS_ARCHITECTURE_INTELGT_ARCHITECTUREINTELGT_H

#include "lldb/Core/Architecture.h"

namespace lldb_private {

/// Architecture plugin for Intel GT GPUs; supplies framedesc-based unwinding
/// because compiler-generated DWARF CFI has incomplete PC location rules.
class ArchitectureIntelGT : public Architecture {
public:
  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() { return "intelgt"; }

  ArchitectureIntelGT() = default;
  ~ArchitectureIntelGT() override = default;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void OverrideStopInfo(Thread &thread) const override {}

  /// Framedesc-based unwinding that fixes DWARF CFI's PC location rule.
  lldb::UnwindPlanSP GetArchitectureUnwindPlan(
      Thread &thread, RegisterContextUnwind *regctx,
      std::shared_ptr<const UnwindPlan> current_unwindplan) override;

private:
  static std::unique_ptr<Architecture> Create(const ArchSpec &arch);
};

} // namespace lldb_private

#endif
