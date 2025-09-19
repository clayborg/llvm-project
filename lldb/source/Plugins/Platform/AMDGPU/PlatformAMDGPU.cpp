//===-- PlatformAMDGPU.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformAMDGPU.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"

#include "llvm/TargetParser/Triple.h"


using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_AMDGPU;

LLDB_PLUGIN_DEFINE(PlatformAMDGPU)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformAMDGPU::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (!create && arch)
    create = arch->GetTriple().isAMDGPU();
  if (create)
    return PlatformSP(new PlatformAMDGPU());
  return PlatformSP();
}

llvm::StringRef PlatformAMDGPU::GetPluginDescriptionStatic(bool is_host) {
  return "AMD GPU platform plug-in.";
}

void PlatformAMDGPU::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformAMDGPU::GetPluginNameStatic(false),
        PlatformAMDGPU::GetPluginDescriptionStatic(false),
        PlatformAMDGPU::CreateInstance, nullptr);
  }
}

void PlatformAMDGPU::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformAMDGPU::CreateInstance);

  Platform::Terminate();
}

PlatformAMDGPU::PlatformAMDGPU() : Platform(/*is_host=*/false) {
  m_supported_architectures =
        CreateArchList({llvm::Triple::r600, llvm::Triple::amdgcn}, 
                       llvm::Triple::AMDHSA);
}

std::vector<ArchSpec>
PlatformAMDGPU::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

void PlatformAMDGPU::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);
}

void PlatformAMDGPU::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformAMDGPU::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                      ConstString name) {
  return {};
}

CompilerType PlatformAMDGPU::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}

lldb::ProcessSP PlatformAMDGPU::Attach(ProcessAttachInfo &attach_info,
                                       Debugger &debugger,
                                       Target *target,
                                       Status &error) {
  error = Status::FromErrorString("PlatformAMDGPU::Attach() not implemented");
  return lldb::ProcessSP();
}
