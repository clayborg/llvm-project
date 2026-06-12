//===----------------------------------------------------------------------===//
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
/// This plugin handles standalone NVGPU corefiles (ET_CORE + EM_CUDA) that
/// contain GPU execution state. It walks the synthetic GPU hierarchy that
/// `ObjectFileELF::CreateSections` builds for NVGPU corefiles
/// (`nvgpucore` root -> `dev0` -> `sm0` -> `cta0` -> `warp0` -> `lane0` ->
/// per-lane leaves, plus a sibling `dev0` -> `grid0` branch) and creates one
/// `ThreadNVGPUCore` per active GPU lane.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H

#include "CudbgEntryParser.h"

#include "lldb/Target/PostMortemProcess.h"
#include "lldb/lldb-forward.h"

#include <optional>

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

  void RefreshStateAfterStop() override {
    // Prefer an exception thread, then a trap thread; otherwise fall
    // back to the first thread we created.
    if (m_exception_tid != LLDB_INVALID_THREAD_ID)
      GetThreadList().SetSelectedThreadByID(m_exception_tid);
    else if (m_stop_tid != LLDB_INVALID_THREAD_ID)
      GetThreadList().SetSelectedThreadByID(m_stop_tid);
    else {
      if (lldb::ThreadSP first = GetThreadList().GetThreadAtIndex(0))
        GetThreadList().SetSelectedThreadByID(first->GetID());
    }
  }

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

  lldb_private::ObjectFile *GetCoreObjectFile() const;

protected:
  lldb_private::Status
  DoGetMemoryRegionInfo(lldb::addr_t load_addr,
                        lldb_private::MemoryRegionInfo &region_info) override;

private:
  llvm::Error LoadCubinModules();

  /// Decode the coredump metadata section (if present) into `m_producer`, log
  /// the producer driver/CUDA version, and warn the user if it is missing or
  /// from a different CUDA major release.
  void LoadProducerInfo(const lldb_private::SectionList &sections);

  /// Raw bytes of the `.cudbg.meta` metadata section (producer driver / CUDA
  /// version), or std::nullopt if the coredump has no such section.
  std::optional<lldb_private::DataExtractor>
  GetNVGPUMetadata(const lldb_private::SectionList &sections);

  /// Find the nvgpu-global-memory or nvgpu-managed-memory leaf section
  /// under the nvgpucore root that contains the given GPU virtual address.
  ///
  /// \param[in] addr
  ///     The GPU virtual address to locate.
  ///
  /// \return
  ///     The containing memory leaf section, or null if no such section
  ///     exists.
  lldb::SectionSP FindGlobalMemorySection(lldb::addr_t addr) const;

  lldb::ModuleSP m_core_module_sp;
  lldb::SectionSP m_root_sp;
  /// Driver/toolkit version that produced this coredump (from the metadata
  /// section), or std::nullopt if the coredump has no decodable metadata.
  std::optional<lldb_private::nvgpu_core::ProducerInfo> m_producer;
  /// First thread with an attributed CUDA exception.
  lldb::tid_t m_exception_tid = LLDB_INVALID_THREAD_ID;
  /// First thread stopped on an inline `trap;` / `__trap()`.
  lldb::tid_t m_stop_tid = LLDB_INVALID_THREAD_ID;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_PROCESSNVGPUCORE_H
