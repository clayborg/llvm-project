//===-- DynamicLoaderGpuCoreDYLD.h ----------------------------------------*-
// C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERGPUCOREDYLD_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERGPUCOREDYLD_H

#include "lldb/Core/LoadedModuleInfoList.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/DynamicLoader.h"
#include "llvm/Support/RWMutex.h"

/**
 * Dynamic loader for GPU coredump process.
 */
class DynamicLoaderGpuCoreDYLD : public lldb_private::DynamicLoader {
public:
  DynamicLoaderGpuCoreDYLD(lldb_private::Process *process);

  ~DynamicLoaderGpuCoreDYLD() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "gpucore-dyld"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  // DynamicLoader protocol

  void DidAttach() override;

  void DidLaunch() override {
    llvm_unreachable("DynamicLoaderGpuCoreDYLD::DidLaunch shouldn't be called");
  }

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override {
    llvm_unreachable("DynamicLoaderGpuCoreDYLD::"
                     "GetStepThroughTrampolinePlan shouldn't be called");
  }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb_private::Status CanLoadImage() override {
    return lldb_private::Status();
  }

private:
  DynamicLoaderGpuCoreDYLD(const DynamicLoaderGpuCoreDYLD &) = delete;
  const DynamicLoaderGpuCoreDYLD &
  operator=(const DynamicLoaderGpuCoreDYLD &) = delete;

  // Structure to hold module information
  struct ModuleInfo {
    std::string name;
    lldb::addr_t base_addr;
    lldb::addr_t module_size;
    lldb::addr_t link_map_addr;
  };

  // TODO: merge with DynamicLoaderPOSIXDYLD::m_loaded_modules
  // The same as DynamicLoaderPOSIXDYLD::m_loaded_modules to track all loaded
  // module's link map addresses. It is used by TLS to get DTV data structure.
  /// This may be accessed in a multi-threaded context. Use the accessor methods
  /// to access `m_loaded_modules` safely.
  std::map<lldb::ModuleWP, lldb::addr_t, std::owner_less<lldb::ModuleWP>>
      m_loaded_modules;
  mutable llvm::sys::RWMutex m_loaded_modules_rw_mutex;

  void SetLoadedModule(const lldb::ModuleSP &module_sp,
                       lldb::addr_t link_map_addr);
  std::optional<lldb::addr_t>
  GetLoadedModuleLinkAddr(const lldb::ModuleSP &module_sp);
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERGPUCOREDYLD_H
