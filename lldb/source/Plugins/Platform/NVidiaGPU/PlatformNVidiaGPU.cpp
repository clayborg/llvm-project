//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformNVidiaGPU.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"

#include "llvm/TargetParser/Triple.h"


using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_NVidiaGPU;

LLDB_PLUGIN_DEFINE(PlatformNVidiaGPU)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformNVidiaGPU::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (!create && arch)
    create = arch->GetTriple().isNVPTX();
  if (create)
    return PlatformSP(new PlatformNVidiaGPU());
  return PlatformSP();
}

llvm::StringRef PlatformNVidiaGPU::GetPluginDescriptionStatic(bool is_host) {
  return "NVidiaGPU platform plug-in.";
}

void PlatformNVidiaGPU::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformNVidiaGPU::GetPluginNameStatic(false),
        PlatformNVidiaGPU::GetPluginDescriptionStatic(false),
        PlatformNVidiaGPU::CreateInstance, nullptr);
  }
}

void PlatformNVidiaGPU::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformNVidiaGPU::CreateInstance);

  Platform::Terminate();
}

PlatformNVidiaGPU::PlatformNVidiaGPU() : Platform(/*is_host=*/false) {
  m_supported_architectures =
        CreateArchList({llvm::Triple::nvptx, llvm::Triple::nvptx64}, 
                       llvm::Triple::UnknownOS);
}

std::vector<ArchSpec>
PlatformNVidiaGPU::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

void PlatformNVidiaGPU::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);
}

void PlatformNVidiaGPU::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformNVidiaGPU::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                      ConstString name) {
  return {};
}

CompilerType PlatformNVidiaGPU::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}

lldb::ProcessSP PlatformNVidiaGPU::Attach(ProcessAttachInfo &attach_info,
                                       Debugger &debugger,
                                       Target *target,
                                       Status &error) {
  error = Status::FromErrorString("PlatformNVidiaGPU::Attach() not implemented");
  return lldb::ProcessSP();
}
