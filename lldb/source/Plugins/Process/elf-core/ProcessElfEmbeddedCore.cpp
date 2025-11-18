//===-- ProcessElfEmbeddedCore.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include <memory>

#include "ProcessElfEmbeddedCore.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb_private;
using namespace lldb;

llvm::Expected<lldb::TargetSP> ProcessElfEmbeddedCore::CreateEmbeddedCoreTarget(
    lldb_private::Debugger &debugger) {

  // Create target for embedded core
  TargetSP target_sp;
  llvm::StringRef exe_path;
  llvm::StringRef triple;
  OptionGroupPlatform *platform_options = nullptr;

  Status error(debugger.GetTargetList().CreateTarget(
      debugger, exe_path, triple, eLoadDependentsNo, platform_options,
      target_sp));

  if (error.Fail())
    return error.ToError();

  if (!target_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create embedded core target");

  return target_sp;
}

void ProcessElfEmbeddedCore::LoadEmbeddedCoreFiles(
    std::shared_ptr<ProcessElfCore> cpu_core_process,
    const FileSpec &core_file) {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log, "ProcessElfEmbeddedCore::LoadEmbeddedCoreFiles() - Looking "
                 "for embedded core data in core file");

  lldb_private::Debugger &debugger =
      cpu_core_process->GetTarget().GetDebugger();

  // Try all embedded core plugins (simpler callback mechanism)
  // These don't require a GPU target upfront
  for (uint32_t idx = 0; GetEmbeddedCoreCreateCallbackAtIndex(idx) != nullptr;
       ++idx) {
    auto create_callback = GetEmbeddedCoreCreateCallbackAtIndex(idx);
    llvm::StringRef plugin_name = GetEmbeddedCorePluginNameAtIndex(idx);

    LLDB_LOGF(log, "LoadEmbeddedCoreFiles() - Trying embedded core plugin: %s",
              plugin_name.str().c_str());

    create_callback(cpu_core_process, debugger.GetListener(), &core_file);
  }
}

// Plugin management for embedded core plugins
struct EmbeddedCorePluginInstance {
  EmbeddedCorePluginInstance(
      llvm::StringRef name, llvm::StringRef description,
      ProcessElfEmbeddedCore::ELFEmbeddedCoreCreateInstance create_callback)
      : name(name), description(description), create_callback(create_callback) {
  }

  std::string name;
  std::string description;
  ProcessElfEmbeddedCore::ELFEmbeddedCoreCreateInstance create_callback;
};

static std::vector<EmbeddedCorePluginInstance> &
GetEmbeddedCorePluginInstances() {
  static std::vector<EmbeddedCorePluginInstance> g_instances;
  return g_instances;
}

void ProcessElfEmbeddedCore::RegisterEmbeddedCorePlugin(
    llvm::StringRef name, llvm::StringRef description,
    ELFEmbeddedCoreCreateInstance create_callback) {
  if (create_callback) {
    auto &instances = GetEmbeddedCorePluginInstances();
    instances.emplace_back(name, description, create_callback);
  }
}

bool ProcessElfEmbeddedCore::UnregisterEmbeddedCorePlugin(
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

ProcessElfEmbeddedCore::ELFEmbeddedCoreCreateInstance
ProcessElfEmbeddedCore::GetEmbeddedCoreCreateCallbackAtIndex(uint32_t idx) {
  auto &instances = GetEmbeddedCorePluginInstances();
  if (idx < instances.size())
    return instances[idx].create_callback;
  return nullptr;
}

llvm::StringRef
ProcessElfEmbeddedCore::GetEmbeddedCorePluginNameAtIndex(uint32_t idx) {
  auto &instances = GetEmbeddedCorePluginInstances();
  if (idx < instances.size())
    return instances[idx].name;
  return llvm::StringRef();
}
