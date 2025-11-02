//===-- ProcessElfGpuCore.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the base class for GPU core file debugging plugins.
///
/// OVERVIEW
/// --------
/// This architecture supports debugging GPU cores in two scenarios:
///
/// 1. **Hybrid CPU+GPU Core Files** (e.g., AMD ROCm applications)
///    - A single core file contains both CPU and GPU state
///    - The CPU process (ProcessElfCore) loads first
///    - GPU plugin(s) detect and load GPU-specific data from the same file
///    - GPU processes are linked to the CPU process for unified debugging
///
/// 2. **Pure GPU Core Files** (e.g., standalone GPU dumps)
///    - Core file contains only GPU state (no CPU process)
///    - Regular process plugin loads as the primary/only process
///    - No CPU process linking required
///
/// SCENARIO 1: GPU-ONLY CORE FILES (Simple Case)
/// ----------------------------------------------
///
/// If your GPU vendor ONLY supports standalone GPU cores (no hybrid support):
///
/// **Step 1: Create Your Plugin Class**
///
/// \code
/// class ProcessVendorGpuCore : public ProcessElfGpuCore {
/// public:
///   static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
///                                         lldb::ListenerSP listener_sp,
///                                         const FileSpec *crash_file,
///                                         bool can_connect);
///
///   bool CanDebug(lldb::TargetSP target_sp,
///                 bool plugin_specified_by_name) override;
///
///   llvm::StringRef GetPluginName() override {
///     return "vendor-gpu-core";
///   }
///
/// protected:
///   Status DoLoadCore() override;
/// };
/// \endcode
///
/// **Step 2: Register as Regular Process Plugin**
///
/// For GPU-only cores, register with RegisterPlugin() (NOT
/// RegisterGpuProcessPlugin):
///
/// \code
/// void ProcessVendorGpuCore::Initialize() {
///   PluginManager::RegisterPlugin(
///       GetPluginNameStatic(),
///       GetPluginDescriptionStatic(),
///       CreateInstance);
/// }
/// \endcode
///
/// **Step 3: Implement CanDebug()**
///
/// \code
/// bool ProcessVendorGpuCore::CanDebug(lldb::TargetSP target_sp,
///                                     bool plugin_specified_by_name) {
///   // Check if this is a pure GPU core file
///   if (ObjectFile *obj = /* get object file from target */) {
///     return IsVendorGpuCore(obj);
///   }
///   return false;
/// }
/// \endcode
///
/// **Step 4: Implement DoLoadCore()**
///
/// \code
/// Status ProcessVendorGpuCore::DoLoadCore() {
///   Status error = ProcessElfCore::DoLoadCore();
///   if (error.Fail())
///     return error;
///
///   // Load GPU-specific state (threads, memory, registers, etc.)
///   error = LoadGpuThreads();
///   if (error.Fail())
///     return error;
///
///   return error;
/// }
/// \endcode
///
/// SCENARIO 2: HYBRID CPU+GPU SUPPORT (Advanced Case)
/// ---------------------------------------------------
///
/// If your GPU vendor supports BOTH pure GPU cores AND hybrid CPU+GPU cores
/// (e.g., AMD ROCm), you need to register BOTH plugin types:
///
/// **Step 1: Create Your Plugin Class (same as above)**
///
/// \code
/// class ProcessVendorGpuCore : public ProcessElfGpuCore {
///   // Same as GPU-only scenario
/// };
/// \endcode
///
/// **Step 2: Register BOTH Plugin Types**
///
/// \code
/// void ProcessVendorGpuCore::Initialize() {
///   // Register as regular process plugin for GPU-only cores
///   PluginManager::RegisterPlugin(
///       GetPluginNameStatic(),
///       GetPluginDescriptionStatic(),
///       CreateInstance);
///
///   // ALSO register as GPU process plugin for hybrid cores
///   PluginManager::RegisterGpuProcessPlugin(
///       GetPluginNameStatic(),
///       GetPluginDescriptionStatic(),
///       CreateInstance);
/// }
/// \endcode
///
/// **Step 3: Implement Smart CanDebug()**
///
/// Your CanDebug() must handle BOTH scenarios:
///
/// \code
/// bool ProcessVendorGpuCore::CanDebug(lldb::TargetSP target_sp,
///                                     bool plugin_specified_by_name) {
///   // Scenario 1: Called via GPU plugin system (hybrid CPU+GPU)
///   // GetCpuProcess() will be non-null because FindGpuPlugin() set it
///   if (auto cpu_process = GetCpuProcess()) {
///     ObjectFile *obj = cpu_process->GetObjectFile();
///     // Check if CPU core file has our GPU data
///     return HasVendorGpuData(obj);
///   }
///
///   // Scenario 2: Called via regular plugin system (pure GPU core)
///   // This is the FIRST time we're seeing the core file
///   if (ObjectFile *obj = /* get object file from target */) {
///     // Check if this is a pure GPU core
///     if (IsVendorPureGpuCore(obj)) {
///       // Return TRUE to load as regular process (GPU-only)
///       return true;
///     }
///
///     // Check if this is actually a hybrid core with CPU data
///     if (HasCpuData(obj)) {
///       // Return FALSE - yield to ProcessElfCore
///       // ProcessElfCore will load the CPU, then call us via GPU plugin
///       return false;
///     }
///   }
///
///   return false;
/// }
/// \endcode
///
/// **Step 4: Implement DoLoadCore() (same as GPU-only)**
///
/// \code
/// Status ProcessVendorGpuCore::DoLoadCore() {
///   Status error = ProcessElfCore::DoLoadCore();
///   if (error.Fail())
///     return error;
///
///   // Load GPU-specific state
///   error = LoadGpuThreads();
///   return error;
/// }
/// \endcode
///
/// HOW IT WORKS: Hybrid CPU+GPU Flow
/// ----------------------------------
///
/// When user loads a hybrid core: `target create --core hybrid.core`
///
/// 1. LLDB calls CanDebug() on all regular process plugins
/// 2. Your CanDebug() sees no CPU process (GetCpuProcess() == nullptr)
/// 3. Your CanDebug() detects hybrid core with CPU data
/// 4. Your CanDebug() returns FALSE (yields to ProcessElfCore)
/// 5. ProcessElfCore::CanDebug() returns TRUE and loads
/// 6. ProcessElfCore::DoLoadCore() calls ProcessElfGpuCore::LoadGpuCore()
/// 7. LoadGpuCore() calls FindGpuPlugin() to iterate GPU plugins
/// 8. FindGpuPlugin() calls your CreateInstance() and sets CPU process
/// 9. Your CanDebug() is called again, this time GetCpuProcess() != nullptr
/// 10. Your CanDebug() returns TRUE (accepts GPU data)
/// 11. LoadGpuCore() registers your GPU process with GPU target
/// 12. Both CPU and GPU targets/processes exist side-by-side
///
/// HOW IT WORKS: Pure GPU Flow
/// ----------------------------
///
/// When user loads a pure GPU core: `target create --core gpu.core`
///
/// 1. LLDB calls CanDebug() on all regular process plugins
/// 2. Your CanDebug() sees no CPU process (GetCpuProcess() == nullptr)
/// 3. Your CanDebug() detects pure GPU core (no CPU data)
/// 4. Your CanDebug() returns TRUE
/// 5. Your plugin loads as the primary/only process
/// 6. DoLoadCore() loads GPU state
/// 7. GetCpuProcess() returns nullptr (no CPU coordination)
///
/// COMPLETE EXAMPLE: AMD ROCm Plugin (Hybrid Support)
/// ---------------------------------------------------
///
/// \code
/// // ProcessAmdGpuCore supports BOTH pure GPU and hybrid CPU+GPU
///
/// void ProcessAmdGpuCore::Initialize() {
///   // Register for pure GPU cores
///   PluginManager::RegisterPlugin(..., CreateInstance);
///   // Register for hybrid cores
///   PluginManager::RegisterGpuProcessPlugin(..., CreateInstance);
/// }
///
/// bool ProcessAmdGpuCore::CanDebug(lldb::TargetSP target_sp,
///                                  bool plugin_specified_by_name) {
///   // Hybrid scenario: called via GPU plugin
///   if (auto cpu_process = GetCpuProcess()) {
///     ObjectFile *obj = cpu_process->GetObjectFile();
///     return HasAmdGpuNotes(obj);
///   }
///
///   // Pure GPU or hybrid detection
///   if (ObjectFile *obj = /* from target */) {
///     // Pure AMD GPU core?
///     if (IsAmdPureGpuCore(obj))
///       return true;  // Accept as regular process
///
///     // Hybrid core with CPU?
///     if (HasCpuThreads(obj))
///       return false; // Yield to ProcessElfCore
///   }
///
///   return false;
/// }
///
/// Status ProcessAmdGpuCore::DoLoadCore() {
///   Status error = ProcessElfCore::DoLoadCore();
///   if (error.Fail())
///     return error;
///
///   // Load AMD GPU threads, memory maps, registers
///   error = LoadAmdGpuState();
///   return error;
/// }
/// \endcode
///
/// KEY POINTS
/// ----------
/// - Override DoLoadCore()
/// - GPU-only vendors: Use RegisterPlugin() only
/// - Hybrid vendors: Use BOTH RegisterPlugin() AND RegisterGpuProcessPlugin()
/// - CanDebug() logic differs based on GetCpuProcess() != nullptr
/// - Hybrid cores: yield to ProcessElfCore first, then engage via GPU plugin
/// - Pure GPU cores: accept directly via regular process plugin
/// - GetCpuProcess() is nullptr for pure GPU, non-null for hybrid
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H

#include "ProcessElfCore.h"

class ProcessElfGpuCore : public lldb_private::PostMortemProcess {
public:
  // Constructors and Destructors
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  // Constructors and Destructors
  ProcessElfGpuCore(lldb::TargetSP target_sp,
                    std::shared_ptr<ProcessElfCore> cpu_core_process,
                    lldb::ListenerSP listener_sp,
                    const lldb_private::FileSpec &core_file)
      : PostMortemProcess(target_sp, listener_sp, core_file),
        m_cpu_core_process(cpu_core_process) {}

  static llvm::Expected<std::shared_ptr<ProcessElfGpuCore>>
  LoadGpuCore(std::shared_ptr<ProcessElfCore> cpu_core_process,
              const lldb_private::FileSpec &core_file);

  std::shared_ptr<ProcessElfCore> GetCpuProcess() {
    return m_cpu_core_process.lock();
  }

  lldb_private::Status DoDestroy() override { return lldb_private::Status(); }

  void RefreshStateAfterStop() override {}

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      lldb_private::Status &error) override {
    error.FromErrorString("not implemented");
    return 0;
  }

private:
  /// Find a GPU plugin that can handle the given core file similar to
  /// Process::FindPlugin().
  /// Automatically sets the CPU process on the GPU
  /// process before calling CanDebug(), so CanDebug() has access to the CPU
  /// process.
  static lldb::ProcessSP
  FindGpuPlugin(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                const lldb_private::FileSpec *crash_file_path,
                std::shared_ptr<ProcessElfCore> cpu_core_process);

  /// Set the CPU process for this GPU process. This establishes the link
  /// between the GPU and CPU processes for merged core file debugging.
  void SetCpuProcess(std::shared_ptr<ProcessElfCore> cpu_core_process) {
    m_cpu_core_process = cpu_core_process;
  }

protected:
  std::weak_ptr<ProcessElfCore> m_cpu_core_process;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
