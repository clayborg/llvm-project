//===-- DynamicLoaderProcessModuleList.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Main header include
#include "DynamicLoaderProcessModuleList.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(DynamicLoaderProcessModuleList,
                       DynamicLoaderProcessModuleList)

void DynamicLoaderProcessModuleList::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderProcessModuleList::Terminate() {}

llvm::StringRef DynamicLoaderProcessModuleList::GetPluginDescriptionStatic() {
  return "Dynamic loader that loads modules from process-provided module list";
}

DynamicLoader *DynamicLoaderProcessModuleList::CreateInstance(Process *process,
                                                              bool force) {
  // Only used when requested by name from a Process that provides
  // LoadedModuleInfoList through GetLoadedModuleList().
  if (force)
    return new DynamicLoaderProcessModuleList(process);
  return nullptr;
}

DynamicLoaderProcessModuleList::DynamicLoaderProcessModuleList(Process *process)
    : DynamicLoader(process) {}

DynamicLoaderProcessModuleList::~DynamicLoaderProcessModuleList() {}

static std::optional<ModuleSP>
LoadModule(Process *process,
           const LoadedModuleInfoList::LoadedModuleInfo &mod_info) {
  Log *log = GetLog(LLDBLog::DynamicLoader);

  std::string name;
  addr_t base_addr;
  if (!mod_info.get_name(name) || !mod_info.get_base(base_addr))
    return std::nullopt;

  std::shared_ptr<DataBufferHeap> data_sp;

  // Read the object file from memory if available.
  lldb::addr_t native_mem_addr, native_mem_size;
  if (mod_info.get_native_memory_address(native_mem_addr) &&
      mod_info.get_native_memory_size(native_mem_size)) {
    LLDB_LOG(log, "Reading \"{0}\" from memory at {1:x}", name,
             native_mem_addr);
    TargetSP cpu_target_sp = process->GetTarget().GetNativeTargetForGPU();
    if (cpu_target_sp) {
      ProcessSP cpu_process_sp = cpu_target_sp->GetProcessSP();
      if (cpu_process_sp) {
        data_sp = std::make_shared<DataBufferHeap>(native_mem_size, 0);
        Status error;
        const size_t bytes_read =
            cpu_process_sp->ReadMemory(native_mem_addr, data_sp->GetBytes(),
                                       data_sp->GetByteSize(), error);
        if (bytes_read != native_mem_size) {
          LLDB_LOG(log, "Failed to read \"{0}\" from memory at {1:x}: {2}",
                   name, native_mem_addr, error);
          data_sp.reset();
        }
      } else {
        LLDB_LOG(log, "Invalid CPU process for \"{0}\" from memory at {1:x}",
                 name, native_mem_addr);
      }
    } else {
      LLDB_LOG(log, "Invalid CPU target for \"{0}\" from memory at {1:x}", name,
               native_mem_addr);
    }
  }

  Target &target = process->GetTarget();
  UUID uuid;
  // Create a module specification from the info we got.
  ModuleSpec module_spec(FileSpec(name), uuid, data_sp);

  // Set file offset and size if available
  uint64_t file_offset, file_size;
  if (mod_info.get_file_offset(file_offset))
    module_spec.SetObjectOffset(file_offset);
  if (mod_info.get_file_size(file_size))
    module_spec.SetObjectSize(file_size);

  // Get or create the module from the module spec.
  ModuleSP module_sp = target.GetOrCreateModule(module_spec,
                                                /*notify=*/true);
  if (module_sp) {
    LLDB_LOG(log, "Created module for \"{0}\": {1:x}", name, module_sp.get());
    bool changed = false;
    LLDB_LOG(log, "Setting load address for module \"{0}\" to {1:x}", name,
             base_addr);

    module_sp->SetLoadAddress(target, base_addr,
                              /*value_is_offset=*/true, changed);
    if (changed) {
      LLDB_LOG(log, "Module \"{0}\" was loaded, notifying target", name);
      return module_sp;
    }
  }
  return std::nullopt;
}

void DynamicLoaderProcessModuleList::DidAttach() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderProcessModuleList::%s() pid %" PRIu64,
            __FUNCTION__,
            m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);

  llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
      m_process->GetLoadedModuleList();
  if (!module_info_list_ep) {
    LLDB_LOGF(log,
              "DynamicLoaderProcessModuleList::%s fail to get module list "
              "from GetLoadedModuleList().",
              __FUNCTION__);
    llvm::consumeError(module_info_list_ep.takeError());
    return;
  }

  const LoadedModuleInfoList &module_info_list = *module_info_list_ep;
  ModuleList module_list;
  for (const LoadedModuleInfoList::LoadedModuleInfo &mod_info :
       module_info_list.m_list) {
    std::optional<ModuleSP> module_sp = LoadModule(m_process, mod_info);
    if (module_sp)
      module_list.AppendIfNeeded(*module_sp);
  }
  m_process->GetTarget().ModulesDidLoad(module_list);
}
