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

#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Threading.h"

using namespace lldb_private;
using namespace lldb;

lldb::ProcessSP ProcessElfGpuCore::CreateInstance(lldb::TargetSP target_sp,
                                                  lldb::ListenerSP listener_sp,
                                                  const FileSpec *crash_file,
                                                  bool can_connect) {
  return ProcessElfCore::CreateInstance(target_sp, listener_sp, crash_file,
                                        can_connect);
}

lldb::ProcessSP ProcessElfGpuCore::FindGpuPlugin(
    lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
    const FileSpec *crash_file_path,
    std::shared_ptr<ProcessElfCore> cpu_core_process) {
  Log *log = GetLog(LLDBLog::Process);

  ProcessSP process_sp;
  ProcessCreateInstance create_callback = nullptr;

  // Iterate through all registered GPU Process plugins
  for (uint32_t idx = 0;
       (create_callback =
            PluginManager::GetGpuProcessCreateCallbackAtIndex(idx)) != nullptr;
       ++idx) {

    llvm::StringRef plugin_name =
        PluginManager::GetGpuProcessPluginNameAtIndex(idx);
    LLDB_LOGF(log, "FindGpuPlugin() - Trying GPU plugin: %s",
              plugin_name.str().c_str());

    // Try to create GPU process with this plugin
    process_sp =
        create_callback(target_sp, listener_sp, crash_file_path, false);

    if (process_sp) {
      // Set the CPU process BEFORE calling CanDebug so it has access to it.
      // TODO: if we can extend ProcessCreateInstance to accept cpu process
      // parameter this can be removed.
      std::shared_ptr<ProcessElfGpuCore> gpu_process_sp =
          std::static_pointer_cast<ProcessElfGpuCore>(process_sp);
      if (gpu_process_sp) {
        gpu_process_sp->SetCpuProcess(cpu_core_process);
      }

      // Now CanDebug can access the CPU process if needed
      if (process_sp->CanDebug(target_sp, false)) {
        LLDB_LOGF(log, "FindGpuPlugin() - Plugin %s accepted core",
                  plugin_name.str().c_str());
        break;
      } else {
        process_sp.reset();
      }
    }

    LLDB_LOGF(log, "FindGpuPlugin() - Plugin %s declined core",
              plugin_name.str().c_str());
  }

  return process_sp;
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

  // Use FindGpuPlugin to discover which GPU plugin can handle this core file
  // This also automatically sets the CPU process on the GPU process
  lldb::ProcessSP process_sp = ProcessElfGpuCore::FindGpuPlugin(
      gpu_target_sp, debugger.GetListener(), &core_file,
      cpu_core_process);

  if (!process_sp) {
    // No GPU plugin found - this is NOT an error, just means no GPU data
    LLDB_LOGF(log,
              "ProcessElfGpuCore::LoadGpuCore() - No GPU data found in core "
              "(this is OK, core may be CPU-only)");
    // Clean up the GPU target we created since we won't be using it
    debugger.GetTargetList().DeleteTarget(gpu_target_sp);
    return nullptr; // Return nullptr to indicate "no GPU" (graceful)
  }

  // Set the process on the GPU target (this is critical!)
  gpu_target_sp->SetProcessSP(process_sp);

  // Cast to ProcessElfGpuCore
  std::shared_ptr<ProcessElfGpuCore> gpu_process_sp =
      std::static_pointer_cast<ProcessElfGpuCore>(process_sp);

  if (!gpu_process_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "GPU process is not a ProcessElfGpuCore");

  // Get the plugin name for the GPU target association
  llvm::StringRef plugin_name = gpu_process_sp->GetPluginName();

  // Set up the CPU-GPU connection
  cpu_core_process->GetTarget().SetGPUPluginTarget(plugin_name, gpu_target_sp);

  // Load the GPU core
  error = gpu_process_sp->LoadCore();
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
