//===-- AmdGpuSymbolLoader.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/AmdGpuSymbolLoader.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UUID.h"

using namespace lldb;
using namespace lldb_private;

bool lldb_private::LoadAmdGpuCodeObjectSymbols(Target &target,
                                               const ModuleSP &module_sp) {
  if (!module_sp || module_sp->GetSymbolFileFileSpec())
    return false;

  if (!ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return false;

  const UUID &uuid = module_sp->GetUUID();
  if (!uuid.IsValid())
    return false;

  ModuleSpec module_spec(module_sp->GetFileSpec(), uuid);
  module_spec.GetArchitecture() = module_sp->GetArchitecture();
  module_spec.SetObjectOffset(module_sp->GetObjectOffset());
  module_spec.SetObjectSize(module_sp->GetObjectSize());

  FileSpec symbol_file_spec;
  if (PlatformSP platform_sp = target.GetPlatform()) {
    ModuleSP located_module_sp;
    bool did_create = false;
    platform_sp->CallLocateModuleCallbackIfSet(module_spec, located_module_sp,
                                               symbol_file_spec, &did_create);
    if (symbol_file_spec)
      module_spec.GetSymbolFileSpec() = symbol_file_spec;
  }

  StatisticsMap symbol_locator_map;
  if (!module_spec.GetSymbolFileSpec()) {
    FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();
    module_spec.GetSymbolFileSpec() = PluginManager::LocateExecutableSymbolFile(
        module_spec, search_paths, symbol_locator_map);
    module_sp->GetSymbolLocatorStatistics().merge(symbol_locator_map);
  }

  Status error;
  if (!module_spec.GetSymbolFileSpec()) {
    PluginManager::DownloadObjectAndSymbolFile(module_spec, error,
                                               /*force_lookup=*/true,
                                               /*copy_executable=*/false);
  }

  const FileSpec &resolved_symbol_file_spec = module_spec.GetSymbolFileSpec();
  if (!resolved_symbol_file_spec)
    return false;

  module_sp->SetSymbolFileFileSpec(resolved_symbol_file_spec);
  SymbolFile *symbol_file = module_sp->GetSymbolFile(/*can_create=*/true);
  if (!symbol_file) {
    LLDB_LOG(GetLog(LLDBLog::Symbols),
             "Failed to load AMDGPU code-object symbol file {0}",
             resolved_symbol_file_spec);
    return false;
  }

  ModuleList module_list;
  module_list.Append(module_sp);
  target.SymbolsDidLoad(module_list);

  Status scripting_error;
  StreamString scripting_feedback;
  module_sp->LoadScriptingResourceInTarget(&target, scripting_error,
                                           scripting_feedback);
  return true;
}
