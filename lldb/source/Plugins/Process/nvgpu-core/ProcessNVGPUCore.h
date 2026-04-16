//===-- ProcessNVGPUCore.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Process plugin for NVIDIA GPU core files.
///
/// This plugin handles standalone CUDA core files (ET_CORE + EM_CUDA) that
/// contain GPU execution state. It builds a DOM-like SectionTree from the
/// ELF section hierarchy and lazily parses table entries and register data
/// on demand.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H

#include "NVGPUCoreData.h"
#include "lldb/Target/PostMortemProcess.h"

class ObjectFileELF;

class ProcessNVGPUCore : public lldb_private::PostMortemProcess {
public:
  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "nvgpu-core"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                const lldb_private::FileSpec *crash_file, bool can_connect);

  ProcessNVGPUCore(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                   const lldb_private::FileSpec &core_file);

  ~ProcessNVGPUCore() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  lldb_private::Status DoLoadCore() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override { return nullptr; }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb_private::Status DoDestroy() override { return lldb_private::Status(); }

  void RefreshStateAfterStop() override {}

  lldb_private::Status WillResume() override {
    return lldb_private::Status::FromErrorStringWithFormatv(
        "error: {0} does not support resuming processes", GetPluginName());
  }

  bool WarnBeforeDetach() const override { return false; }

  bool DoUpdateThreadList(lldb_private::ThreadList &old_thread_list,
                          lldb_private::ThreadList &new_thread_list) override;

  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    lldb_private::Status &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      lldb_private::Status &error) override;

  size_t DoReadMemory(const lldb_private::AddressSpec &addr_spec,
                      const lldb_private::AddressSpaceInfo &info, void *buf,
                      size_t size, lldb_private::Status &error) override;

  lldb_private::NVGPUCoreData &GetCoreData() { return m_core_data; }
  const lldb_private::NVGPUCoreData &GetCoreData() const {
    return m_core_data;
  }

  ObjectFileELF *GetCoreObjectFile() const;

protected:
  lldb_private::Status
  DoGetMemoryRegionInfo(lldb::addr_t load_addr,
                        lldb_private::MemoryRegionInfo &region_info) override;

private:
  llvm::Error LoadCubinModules();

  lldb::ModuleSP m_core_module_sp;
  lldb_private::NVGPUCoreData m_core_data;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H
