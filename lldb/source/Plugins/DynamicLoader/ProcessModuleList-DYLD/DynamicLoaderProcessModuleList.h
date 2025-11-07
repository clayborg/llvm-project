//===-- DynamicLoaderProcessModuleList.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERPROCESSMODULELIST_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERPROCESSMODULELIST_H

#include "lldb/Core/LoadedModuleInfoList.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/DynamicLoader.h"
#include "llvm/Support/RWMutex.h"

/// \class DynamicLoaderProcessModuleList
///
/// A generic dynamic loader that loads modules from a process-provided
/// LoadedModuleInfoList. The process implements GetLoadedModuleList() to
/// provide module information, and this loader creates and loads the
/// corresponding modules.
class DynamicLoaderProcessModuleList : public lldb_private::DynamicLoader {
public:
  DynamicLoaderProcessModuleList(lldb_private::Process *process);

  ~DynamicLoaderProcessModuleList() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "process-module-list"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  // DynamicLoader protocol

  void DidAttach() override;

  void DidLaunch() override {
    llvm_unreachable(
        "DynamicLoaderProcessModuleList::DidLaunch shouldn't be called");
  }

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override {
    llvm_unreachable("DynamicLoaderProcessModuleList::"
                     "GetStepThroughTrampolinePlan shouldn't be called");
  }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb_private::Status CanLoadImage() override {
    return lldb_private::Status();
  }

private:
  DynamicLoaderProcessModuleList(const DynamicLoaderProcessModuleList &) =
      delete;
  const DynamicLoaderProcessModuleList &
  operator=(const DynamicLoaderProcessModuleList &) = delete;

  // Structure to hold module information
  struct ModuleInfo {
    std::string name;
    lldb::addr_t base_addr;
    lldb::addr_t module_size;
    lldb::addr_t link_map_addr;
  };
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_GPUCORE_DYLD_DYNAMICLOADERPROCESSMODULELIST_H
