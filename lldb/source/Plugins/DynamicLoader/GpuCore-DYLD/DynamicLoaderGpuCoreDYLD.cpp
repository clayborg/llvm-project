//===-- DynamicLoaderGpuCoreDYLD.cpp--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Main header include
#include "DynamicLoaderGpuCoreDYLD.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "Plugins/ObjectFile/Placeholder/ObjectFilePlaceholder.h"
#include <mutex>
#include <regex>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(DynamicLoaderGpuCoreDYLD, DynamicLoaderGpuCoreDYLD)

void DynamicLoaderGpuCoreDYLD::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderGpuCoreDYLD::Terminate() {}

llvm::StringRef DynamicLoaderGpuCoreDYLD::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in for GPU coredumps";
}

DynamicLoader *DynamicLoaderGpuCoreDYLD::CreateInstance(Process *process,
                                                        bool force) {
  // This plug-in is only used when it is requested by name from
  // ProcessELFCore. ProcessELFCore will look to see if the core
  // file contains a NT_FILE ELF note, and ask for this plug-in
  // by name if it does.
  if (force)
    return new DynamicLoaderGpuCoreDYLD(process);
  return nullptr;
}

DynamicLoaderGpuCoreDYLD::DynamicLoaderGpuCoreDYLD(Process *process)
    : DynamicLoader(process) {}

DynamicLoaderGpuCoreDYLD::~DynamicLoaderGpuCoreDYLD() {}

void DynamicLoaderGpuCoreDYLD::SetLoadedModule(const ModuleSP &module_sp,
                                               addr_t link_map_addr) {
  llvm::sys::ScopedWriter lock(m_loaded_modules_rw_mutex);
  m_loaded_modules[module_sp] = link_map_addr;
}

std::optional<lldb::addr_t>
DynamicLoaderGpuCoreDYLD::GetLoadedModuleLinkAddr(const ModuleSP &module_sp) {
  llvm::sys::ScopedReader lock(m_loaded_modules_rw_mutex);
  auto it = m_loaded_modules.find(module_sp);
  if (it != m_loaded_modules.end())
    return it->second;
  return std::nullopt;
}

struct GPUDynamicLoaderLibraryInfo {
  /// The path to the shared library object file on disk.
  std::string pathname;
  /// The address where the object file is loaded. If this member has a value
  /// the object file is loaded at an address and all sections should be slid to
  /// match this base address. If this member doesn't have a value, then
  /// individual section's load address must be specified individually if
  /// \a loaded_sections has a value. If this doesn't have a value and the
  /// \a loaded_Section doesn't have a value, this library will be unloaded.
  lldb::addr_t load_address;

  /// If this library is only available as an in memory image of an object file
  /// in the native process, then this address holds the address from which the
  /// image can be read.
  std::optional<lldb::addr_t> native_memory_address;
  /// If this library is only available as an in memory image of an object file
  /// in the native process, then this size of the in memory image that starts
  /// at \a native_memory_address.
  std::optional<lldb::addr_t> native_memory_size;
  /// If the library exists inside of a file at an offset, \a file_offset will
  /// have a value that is the offset in bytes from the start of the file
  /// specified by \a pathname.
  std::optional<uint64_t> file_offset;
  /// If the library exists inside of a file at an offset, \a file_size will
  /// have a value that indicates the size in bytes of the object file.
  std::optional<uint64_t> file_size;
};

// This function will parse the shared library string that AMDs GPU driver
// sends to the debugger. The format is one of:
//  file://<path>#offset=<file-offset>&size=<file-size>
//  memory://<name>#offset=<image-addr>&size=<image-size>
static std::optional<GPUDynamicLoaderLibraryInfo>
ParseLibraryInfo(llvm::StringRef lib_spec, lldb::addr_t load_address) {
  auto get_offset_and_size = [](llvm::StringRef &values,
                                std::optional<uint64_t> &offset,
                                std::optional<uint64_t> &size) {
    offset = std::nullopt;
    size = std::nullopt;
    llvm::StringRef value;
    uint64_t uint_value;
    std::tie(value, values) = values.split('&');
    while (!value.empty()) {
      if (value.consume_front("offset=")) {
        if (!value.getAsInteger(0, uint_value))
          offset = uint_value;
      } else if (value.consume_front("size=")) {
        if (!value.getAsInteger(0, uint_value))
          size = uint_value;
      }
      std::tie(value, values) = values.split('&');
    }
  };
  GPUDynamicLoaderLibraryInfo lib_info;
  if (lib_spec.consume_front("file://")) {
    llvm::StringRef path, values;
    std::tie(path, values) = lib_spec.split('#');
    if (path.empty())
      return std::nullopt;
    lib_info.pathname = path.str();
    get_offset_and_size(values, lib_info.file_offset, lib_info.file_size);
  } else if (lib_spec.consume_front("memory://")) {
    llvm::StringRef name, values;
    std::tie(name, values) = lib_spec.split('#');
    if (name.empty())
      return std::nullopt;
    lib_info.pathname = name.str();
    get_offset_and_size(values, lib_info.native_memory_address,
                        lib_info.native_memory_size);
    // We must have a valid address and size for memory objects.
    if (!(lib_info.native_memory_address.has_value() &&
          lib_info.native_memory_size.has_value()))
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  return lib_info;
}

std::optional<ModuleSP> LoadModule(Process *gpu_process, const GPUDynamicLoaderLibraryInfo& info) {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  std::shared_ptr<DataBufferHeap> data_sp;
  // Read the object file from memory if requested.
  if (info.native_memory_address && info.native_memory_size) {
    LLDB_LOG(log, "Reading \"{0}\" from memory at {1:x}", info.pathname,
              *info.native_memory_address);
    TargetSP cpu_target_sp = gpu_process->GetTarget().GetNativeTargetForGPU();
    if (cpu_target_sp) {
      ProcessSP cpu_process_sp = cpu_target_sp->GetProcessSP();
      if (cpu_process_sp) {
        data_sp =
            std::make_shared<DataBufferHeap>(*info.native_memory_size, 0);
        Status error;
        const size_t bytes_read = cpu_process_sp->ReadMemory(
            *info.native_memory_address, data_sp->GetBytes(),
            data_sp->GetByteSize(), error);
        if (bytes_read != *info.native_memory_size) {
          LLDB_LOG(log, "Failed to read \"{0}\" from memory at {1:x}: {2}",
                    info.pathname, *info.native_memory_address, error);
          data_sp.reset();
        }
      } else {
        LLDB_LOG(log, "Invalid CPU process for \"{0}\" from memory at {1:x}",
                  info.pathname, *info.native_memory_address);
      }
    } else {
      LLDB_LOG(log, "Invalid CPU target for \"{0}\" from memory at {1:x}",
                info.pathname, *info.native_memory_address);
    }
  }
  Target &target = gpu_process->GetTarget();
  // Extract the UUID if available.
  UUID uuid;
  // Create a module specification from the info we got.
  ModuleSpec module_spec(FileSpec(info.pathname), uuid, data_sp);
  if (info.file_offset)
    module_spec.SetObjectOffset(*info.file_offset);
  if (info.file_size)
    module_spec.SetObjectSize(*info.file_size);

  ModuleList loaded_module_list;  
  // Get or create the module from the module spec.
  ModuleSP module_sp = target.GetOrCreateModule(module_spec,
                                                /*notify=*/true);
  if (module_sp) {
    LLDB_LOG(log, "Created module for \"{0}\": {1:x}", info.pathname,
              module_sp.get());
    bool changed = false;
    if (info.load_address) {
      LLDB_LOG(log, "Setting load address for module \"{0}\" to {1:x}",
                info.pathname, info.load_address);

      module_sp->SetLoadAddress(target, info.load_address,
                                /*value_is_offset=*/true, changed);
    }
    if (changed) {
      LLDB_LOG(log, "Module \"{0}\" was loaded, notifying target",
                info.pathname);
      return module_sp;
    }
  }
  return std::nullopt;
}

void DynamicLoaderGpuCoreDYLD::DidAttach() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderGpuCoreDYLD::%s() pid %" PRIu64, __FUNCTION__,
            m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);

  llvm::Expected<LoadedModuleInfoList> module_info_list_ep =
      m_process->GetLoadedModuleList();
  if (!module_info_list_ep) {
    LLDB_LOGF(log,
              "DynamicLoaderGpuCoreDYLD::%s fail to get module list "
              "from GetLoadedModuleList().",
              __FUNCTION__);
    llvm::consumeError(module_info_list_ep.takeError());
    return;
  }

  const LoadedModuleInfoList &module_info_list = *module_info_list_ep;
  ModuleList module_list;
  for (const LoadedModuleInfoList::LoadedModuleInfo &mod_info :
       module_info_list.m_list) {
    addr_t base_addr;
    std::string name;
    if (!mod_info.get_base(base_addr) || !mod_info.get_name(name))
      continue;

    std::optional<GPUDynamicLoaderLibraryInfo> lib_info = ParseLibraryInfo(name, base_addr);
    if (!lib_info)
      continue;

    std::optional<ModuleSP> module_sp = LoadModule(m_process, *lib_info);
    if (!module_sp)
        continue;
    module_list.AppendIfNeeded(*module_sp);
  }
  m_process->GetTarget().ModulesDidLoad(module_list);
}
