//===-- PlatformNVGPU.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformNVGPU.h"
#include "lldb/Core/PluginManager.h"

#include "llvm/TargetParser/Triple.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_NVGPU;

LLDB_PLUGIN_DEFINE(PlatformNVGPU)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformNVGPU::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (!create && arch)
    create = arch->GetTriple().isNVPTX();
  if (create)
    return PlatformSP(new PlatformNVGPU());
  return PlatformSP();
}

llvm::StringRef PlatformNVGPU::GetPluginDescriptionStatic(bool is_host) {
  return "NVGPU platform plug-in.";
}

void PlatformNVGPU::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformNVGPU::GetPluginNameStatic(false),
        PlatformNVGPU::GetPluginDescriptionStatic(false),
        PlatformNVGPU::CreateInstance, nullptr);
  }
}

void PlatformNVGPU::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformNVGPU::CreateInstance);

  Platform::Terminate();
}

PlatformNVGPU::PlatformNVGPU() : Platform(/*is_host=*/false) {
  m_supported_architectures = CreateArchList(
      {llvm::Triple::nvptx, llvm::Triple::nvptx64}, llvm::Triple::UnknownOS);
}

std::vector<ArchSpec>
PlatformNVGPU::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  return m_supported_architectures;
}

void PlatformNVGPU::GetStatus(Stream &strm) { Platform::GetStatus(strm); }

void PlatformNVGPU::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformNVGPU::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                        ConstString name) {
  return {};
}

CompilerType PlatformNVGPU::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}

lldb::ProcessSP PlatformNVGPU::Attach(ProcessAttachInfo &attach_info,
                                      Debugger &debugger, Target *target,
                                      Status &error) {
  error = Status::FromErrorString("PlatformNVGPU::Attach() not implemented");
  return lldb::ProcessSP();
}
