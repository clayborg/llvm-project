//===-- ProcessElfGpuCore.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the base class for companion/embedded GPU core file
/// debugging.
///
/// OVERVIEW
/// --------
/// This architecture supports debugging companion cores (e.g., GPU, DSP, or
/// other accelerators) that are embedded within a primary CPU core file.
///
/// **Use Case: Hybrid CPU+Companion Core Files**
///    - A single core file contains both CPU and companion device state
///    - The CPU process (ProcessElfCore) loads first
///    - Companion core plugin(s) detect and extract device-specific data
///    - Companion cores are presented as separate processes for debugging
///
/// CREATING A COMPANION CORE PLUGIN
/// ---------------------------------
///
/// **Step 1: Create Your Plugin Class**
///
/// \code
/// class ProcessVendorCompanionCore : public ProcessElfGpuCore {
/// public:
///   // Factory function for embedded core plugin registration
///   static std::shared_ptr<ProcessElfGpuCore> CreateInstance(
///       std::shared_ptr<ProcessElfCore> cpu_core_process,
///       lldb::ListenerSP listener_sp,
///       const lldb_private::FileSpec *crash_file);
///
///   llvm::StringRef GetPluginName() override {
///     return "vendor-companion-core";
///   }
///
/// protected:
///   Status LoadCore() override;
/// };
/// \endcode
///
/// **Step 2: Register as Embedded Core Plugin**
///
/// Use RegisterEmbeddedCorePlugin() to register your companion core plugin:
///
/// \code
/// void ProcessVendorCompanionCore::Initialize() {
///   ProcessElfGpuCore::RegisterEmbeddedCorePlugin(
///       GetPluginNameStatic(),
///       GetPluginDescriptionStatic(),
///       CreateInstance);
/// }
/// \endcode
///
/// **Step 3: Implement CreateInstance()**
///
/// Your factory function should check if the CPU core contains your companion
/// device data. Return nullptr if not found (allows other plugins to try).
///
/// \code
/// std::shared_ptr<ProcessElfGpuCore>
/// ProcessVendorCompanionCore::CreateInstance(
///     std::shared_ptr<ProcessElfCore> cpu_core_process,
///     lldb::ListenerSP listener_sp,
///     const lldb_private::FileSpec *crash_file) {
///   
///   // Check if CPU core file contains vendor-specific companion data
///   ObjectFile *obj = cpu_core_process->GetTarget().GetExecutableModulePointer()
///                         ->GetObjectFile();
///   if (!HasVendorCompanionData(obj))
///     return nullptr;  // Not our companion core, let other plugins try
///
///   // Found our data! Create the companion target and process
///   lldb::TargetSP target_sp;
///   Status error = CreateGpuTarget(
///       cpu_core_process->GetTarget().GetDebugger(), target_sp);
///   if (error.Fail())
///     return nullptr;
///
///   return std::make_shared<ProcessVendorCompanionCore>(
///       target_sp, cpu_core_process, listener_sp, *crash_file);
/// }
/// \endcode
///
/// **Step 4: Implement DoLoadCore()**
///
/// \code
/// Status ProcessVendorCompanionCore::DoLoadCore() {
///   Status error;
///   ...
///   return error;
/// }
/// \endcode
///
/// HOW IT WORKS: Companion Core Loading Flow
/// ------------------------------------------
///
/// When user loads a core with companion data: `target create --core app.core`
///
/// 1. LLDB calls CanDebug() on all regular process plugins
/// 2. ProcessElfCore::CanDebug() returns TRUE and loads the CPU core
/// 3. ProcessElfCore::DoLoadCore() calls ProcessElfGpuCore::LoadGpuCore()
/// 4. LoadGpuCore() iterates through registered embedded core plugins
/// 5. Each plugin's CreateInstance() is called with the CPU process
/// 6. Plugin checks if its companion data exists in the core file
/// 7. If found, plugin returns a new companion process instance
/// 8. If not found, plugin returns nullptr (next plugin tries)
/// 9. LoadGpuCore() calls LoadCore() on the companion process
/// 10. Companion process extracts its device-specific data
/// 11. CPU and companion processes coexist for unified debugging
///
/// KEY POINTS
/// ----------
/// - Use RegisterEmbeddedCorePlugin() to register companion core plugins
/// - CreateInstance() checks if companion data exists, returns nullptr if not
/// - Multiple plugins can coexist; first match wins
/// - GetCpuProcess() provides access to the CPU core for shared data access
/// - Companion cores are automatically linked to the CPU process
/// - DoLoadCore() extracts device-specific state from the CPU core file
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H

#include "ProcessElfCore.h"

class ProcessElfGpuCore : public lldb_private::PostMortemProcess {
public:
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

  // Plugin code
  typedef std::shared_ptr<ProcessElfGpuCore> (*ELFEmbeddedCoreCreateInstance)(
      std::shared_ptr<ProcessElfCore> cpu_core_process,
      lldb::ListenerSP listener_sp, const lldb_private::FileSpec *crash_file);
  static void
  RegisterEmbeddedCorePlugin(llvm::StringRef name, llvm::StringRef description,
                             ELFEmbeddedCoreCreateInstance create_instance);
  static bool
  UnregisterEmbeddedCorePlugin(ELFEmbeddedCoreCreateInstance create_callback);
  static ELFEmbeddedCoreCreateInstance
  GetEmbeddedCoreCreateCallbackAtIndex(uint32_t idx);
  static llvm::StringRef GetEmbeddedCorePluginNameAtIndex(uint32_t idx);

protected:
  /// Create a GPU target for companion core debugging
  static llvm::Expected<lldb::TargetSP>
  CreateGpuTarget(lldb_private::Debugger &debugger);

protected:
  std::weak_ptr<ProcessElfCore> m_cpu_core_process;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
