//===-- PlatformIntelGT.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformIntelGT.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"

#include "llvm/TargetParser/Triple.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_IntelGT;

LLDB_PLUGIN_DEFINE(PlatformIntelGT)

namespace {
#define LLDB_PROPERTIES_platformintelgtuser
#include "PlatformIntelGTUserProperties.inc"

enum {
#define LLDB_PROPERTIES_platformintelgtuser
#include "PlatformIntelGTUserPropertiesEnum.inc"
};
} // namespace

PlatformIntelGT::PluginProperties::PluginProperties() {
  m_collection_sp = std::make_shared<OptionValueProperties>(
      PlatformIntelGT::GetPluginNameStatic(/*is_host=*/false));
  m_collection_sp->Initialize(g_platformintelgtuser_properties);
}

FileSpec PlatformIntelGT::PluginProperties::GetLibIgaPath() {
  return GetPropertyAtIndexAs<FileSpec>(ePropertyLibIgaPath, {});
}

PlatformIntelGT::PluginProperties &PlatformIntelGT::GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

static uint32_t g_initialize_count = 0;

PlatformSP PlatformIntelGT::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (!create && arch) {
    // Intel GT uses the spirv64 triple.
    create = arch->GetTriple().isIntelGPU();
  }
  if (create)
    return PlatformSP(new PlatformIntelGT());
  return PlatformSP();
}

llvm::StringRef PlatformIntelGT::GetPluginDescriptionStatic(bool is_host) {
  return "IntelGT GPU platform plug-in.";
}

void PlatformIntelGT::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformIntelGT::GetPluginNameStatic(false),
        PlatformIntelGT::GetPluginDescriptionStatic(false),
        PlatformIntelGT::CreateInstance, PlatformIntelGT::DebuggerInitialize);
  }
}

void PlatformIntelGT::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(
          debugger, GetPluginNameStatic(/*is_host=*/false))) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        "Properties for the IntelGT platform plugin.",
        /*is_global_property=*/true);
  }
}

void PlatformIntelGT::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformIntelGT::CreateInstance);

  Platform::Terminate();
}

PlatformIntelGT::PlatformIntelGT() : Platform(/*is_host=*/false) {
  // Intel GT uses the spirv64-unknown-unknown triple.
  m_supported_architectures =
      CreateArchList({llvm::Triple::spirv64}, llvm::Triple::UnknownOS);
}

std::vector<ArchSpec>
PlatformIntelGT::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

void PlatformIntelGT::GetStatus(Stream &strm) { Platform::GetStatus(strm); }

void PlatformIntelGT::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformIntelGT::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                          ConstString name) {
  return {};
}

CompilerType PlatformIntelGT::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}

lldb::ProcessSP PlatformIntelGT::Attach(ProcessAttachInfo &attach_info,
                                        Debugger &debugger, Target *target,
                                        Status &error) {
  error = Status::FromErrorString("PlatformIntelGT::Attach() not implemented");
  return lldb::ProcessSP();
}
