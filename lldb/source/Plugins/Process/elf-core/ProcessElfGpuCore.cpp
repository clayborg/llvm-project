//===-- ProcessElfGpuCore.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include <memory>

#include "ProcessElfGpuCore.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb_private;
using namespace lldb;

llvm::Expected<lldb::TargetSP>
ProcessElfGpuCore::CreateGpuTarget(lldb_private::Debugger &debugger) {

  // Create target for GPU
  TargetSP gpu_target_sp;
  llvm::StringRef exe_path;
  llvm::StringRef triple;
  OptionGroupPlatform *platform_options = nullptr;

  Status error(debugger.GetTargetList().CreateTarget(
      debugger, exe_path, triple, eLoadDependentsNo, platform_options,
      gpu_target_sp));

  if (error.Fail())
    return error.ToError();

  if (!gpu_target_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create GPU target");

  return gpu_target_sp;
}

llvm::Expected<std::shared_ptr<ProcessElfGpuCore>>
ProcessElfGpuCore::LoadGpuCore(std::shared_ptr<ProcessElfCore> cpu_core_process,
                               const FileSpec &core_file) {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(
      log,
      "ProcessElfGpuCore::LoadGpuCore() - Looking for GPU data in core file");

  lldb_private::Debugger &debugger =
      cpu_core_process->GetTarget().GetDebugger();

  // First, try embedded core plugins (simpler callback mechanism)
  // These don't require a GPU target upfront
  std::shared_ptr<ProcessElfGpuCore> gpu_process_sp;
  for (uint32_t idx = 0; GetEmbeddedCoreCreateCallbackAtIndex(idx) != nullptr;
       ++idx) {
    auto create_callback = GetEmbeddedCoreCreateCallbackAtIndex(idx);
    llvm::StringRef plugin_name = GetEmbeddedCorePluginNameAtIndex(idx);

    LLDB_LOGF(log, "LoadGpuCore() - Trying embedded core plugin: %s",
              plugin_name.str().c_str());

    gpu_process_sp =
        create_callback(cpu_core_process, debugger.GetListener(), &core_file);

    if (gpu_process_sp) {
      LLDB_LOGF(log, "LoadGpuCore() - Embedded plugin %s created process",
                plugin_name.str().c_str());
      break;
    }
  }

  if (!gpu_process_sp) {
    // No GPU plugin found - this is NOT an error, just means no GPU data
    LLDB_LOGF(log,
              "ProcessElfGpuCore::LoadGpuCore() - No GPU data found in core "
              "(this is OK, core may be CPU-only)");
    return nullptr; // Return nullptr to indicate "no GPU" (graceful)
  }

  // Get the plugin name for the GPU target association
  llvm::StringRef plugin_name = gpu_process_sp->GetPluginName();

  // Load the GPU core
  Status error = gpu_process_sp->LoadCore();
  if (error.Fail()) {
    // This IS an error - GPU plugin accepted but failed to load
    LLDB_LOGF(log,
              "ProcessElfGpuCore::LoadGpuCore() - GPU plugin %s failed to "
              "load core: %s",
              plugin_name.str().c_str(), error.AsCString());
    return error.ToError();
  }

  // Success!
  LLDB_LOGF(log,
            "ProcessElfGpuCore::LoadGpuCore() - Successfully loaded GPU core "
            "with plugin %s",
            plugin_name.str().c_str());
  return gpu_process_sp;
}

// Plugin management for embedded core plugins
struct EmbeddedCorePluginInstance {
  EmbeddedCorePluginInstance(
      llvm::StringRef name, llvm::StringRef description,
      ProcessElfGpuCore::ELFEmbeddedCoreCreateInstance create_callback)
      : name(name), description(description), create_callback(create_callback) {
  }

  std::string name;
  std::string description;
  ProcessElfGpuCore::ELFEmbeddedCoreCreateInstance create_callback;
};

static std::vector<EmbeddedCorePluginInstance> &
GetEmbeddedCorePluginInstances() {
  static std::vector<EmbeddedCorePluginInstance> g_instances;
  return g_instances;
}

void ProcessElfGpuCore::RegisterEmbeddedCorePlugin(
    llvm::StringRef name, llvm::StringRef description,
    ELFEmbeddedCoreCreateInstance create_callback) {
  if (create_callback) {
    auto &instances = GetEmbeddedCorePluginInstances();
    instances.emplace_back(name, description, create_callback);
  }
}

bool ProcessElfGpuCore::UnregisterEmbeddedCorePlugin(
    ELFEmbeddedCoreCreateInstance create_callback) {
  if (!create_callback)
    return false;

  auto &instances = GetEmbeddedCorePluginInstances();
  auto pos = instances.begin();
  auto end = instances.end();
  for (; pos != end; ++pos) {
    if (pos->create_callback == create_callback) {
      instances.erase(pos);
      return true;
    }
  }
  return false;
}

ProcessElfGpuCore::ELFEmbeddedCoreCreateInstance
ProcessElfGpuCore::GetEmbeddedCoreCreateCallbackAtIndex(uint32_t idx) {
  auto &instances = GetEmbeddedCorePluginInstances();
  if (idx < instances.size())
    return instances[idx].create_callback;
  return nullptr;
}

llvm::StringRef
ProcessElfGpuCore::GetEmbeddedCorePluginNameAtIndex(uint32_t idx) {
  auto &instances = GetEmbeddedCorePluginInstances();
  if (idx < instances.size())
    return instances[idx].name;
  return llvm::StringRef();
}
