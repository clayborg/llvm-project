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

#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Threading.h"

#include "ProcessAmdGpuCore.h"
#include "ProcessElfGpuCore.h"

using namespace lldb_private;
using namespace lldb;

lldb::ProcessSP ProcessElfGpuCore::CreateInstance(lldb::TargetSP target_sp,
                                                  lldb::ListenerSP listener_sp,
                                                  const FileSpec *crash_file,
                                                  bool can_connect) {
  return ProcessElfCore::CreateInstance(target_sp, listener_sp, crash_file,
                                        can_connect);
}

llvm::Expected<std::shared_ptr<ProcessElfGpuCore>>
ProcessElfGpuCore::LoadGpuCore(ProcessElfCore *cpu_core_process,
                               const FileSpec &core_file) {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOG(log, "ProcessElfGpuCore::LoadGpuCore()");

  lldb_private::Debugger &debugger =
      cpu_core_process->GetTarget().GetDebugger();
  TargetSP gpu_target_sp;
  llvm::StringRef exe_path;
  llvm::StringRef triple;
  OptionGroupPlatform *platform_options = nullptr;
  // Create an empty target for our GPU.
  Status error(debugger.GetTargetList().CreateTarget(
      debugger, exe_path, triple, eLoadDependentsNo, platform_options,
      gpu_target_sp));
  if (error.Fail())
    return error.ToError();
  if (!gpu_target_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create target");
  cpu_core_process->GetTarget().SetGPUPluginTarget("amdgpu-core",
                                                   gpu_target_sp);
  // lldb::ProcessSP process_sp = std::make_shared<ProcessAmdGpuCore>(
  //     gpu_target_sp, cpu_core_process, debugger.GetListener(), core_file);

  ObjectFile *core_objfile = cpu_core_process->GetObjectFile();
  if (!core_objfile || core_objfile->GetType() != ObjectFile::eTypeCoreFile)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid core file");

  lldb::ProcessSP process_sp =
      gpu_target_sp->CreateProcess(debugger.GetListener(), "amdgpu-core",
                                   &core_objfile->GetFileSpec(), false);
  if (!process_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create GPU process");
  std::shared_ptr<ProcessElfGpuCore> gpu_process_sp =
      std::static_pointer_cast<ProcessElfGpuCore>(process_sp);
  if (!gpu_process_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create GPU process");
  // TODO: make this part of the plugin interface.
  gpu_process_sp->m_cpu_core_process = cpu_core_process;
  error = gpu_process_sp->LoadCore();
  if (error.Fail())
    return error.ToError();

  return gpu_process_sp;
}
