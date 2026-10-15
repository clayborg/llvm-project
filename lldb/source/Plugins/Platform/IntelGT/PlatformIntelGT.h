//===-- PlatformIntelGT.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_INTELGT_PLATFORMINTELGT_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_INTELGT_PLATFORMINTELGT_H

#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private::platform_IntelGT {

class PlatformIntelGT : public Platform {
public:
  class PluginProperties : public Properties {
  public:
    PluginProperties();

    FileSpec GetLibIgaPath();
  };

  PlatformIntelGT();

  static PluginProperties &GetGlobalProperties();

  static void Initialize();

  static void Terminate();

  // lldb_private::PluginInterface functions
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static llvm::StringRef GetPluginNameStatic(bool is_host) {
    return is_host ? Platform::GetHostPlatformName() : "intelgt";
  }

  static llvm::StringRef GetPluginDescriptionStatic(bool is_host);

  llvm::StringRef GetPluginName() override {
    return GetPluginNameStatic(IsHost());
  }

  // lldb_private::Platform functions
  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic(IsHost());
  }

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &error) override;

  void GetStatus(Stream &strm) override;

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  void CalculateTrapHandlerSymbolNames() override;

  lldb::UnwindPlanSP GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                              ConstString name) override;

  CompilerType GetSiginfoType(const llvm::Triple &triple) override;

  std::vector<ArchSpec> m_supported_architectures;

private:
  static void DebuggerInitialize(Debugger &debugger);
};

} // namespace lldb_private::platform_IntelGT

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_INTELGT_PLATFORMINTELGT_H
