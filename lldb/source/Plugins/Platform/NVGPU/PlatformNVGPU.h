//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_NVGPU_PLATFORMNVGPU_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_NVGPU_PLATFORMNVGPU_H

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Platform.h"

namespace lldb_private::platform_NVGPU {

/// Single mapping entry for a PC range
struct PTXPieceToSassEntry {
  lldb::addr_t pc_start;
  lldb::addr_t pc_end;
  lldb::addr_t pc_extended_end;
  std::string reg_name;
  // PTX location can ba mapped to the maximum of two location
  uint64_t locations = 0;
};

typedef std::map<uint64_t, std::list<PTXPieceToSassEntry>> PTXPRegMap;

class PlatformNVGPU : public Platform {
public:
  class PluginProperties : public Properties {
  public:
    PluginProperties();

    FileSpec GetNvdisasmPath();
  };

  PlatformNVGPU();

  static PluginProperties &GetGlobalProperties();

  static void Initialize();

  static void Terminate();

  // lldb_private::PluginInterface functions
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static llvm::StringRef GetPluginNameStatic(bool is_host) { return "nvgpu"; }

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

  std::optional<llvm::Error> ReadVirtualRegister(RegisterContext *reg_ctx,
                                                 lldb::RegisterKind reg_kind,
                                                 lldb::regnum64_t reg_num,
                                                 Value &value) override;

  void RecordLoadedModule(const lldb::ModuleSP &module_sp,
                          Target &target) override;

private:
  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  uint64_t FindRegisterLocations(const lldb::ModuleSP &module_sp,
                                 lldb::addr_t pc, uint64_t reg_num);

  llvm::Error LocationToValue(RegisterContext *reg_ctx,
                              lldb::RegisterKind reg_kind, uint32_t location,
                              Value &value);

  std::vector<ArchSpec> m_supported_architectures;

  std::map<lldb::ModuleSP, PTXPRegMap> m_entries;
};

} // namespace lldb_private::platform_NVGPU

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_NVGPU_PLATFORMNVGPU_H
