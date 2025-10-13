//===-- ProcessAmdGpuCore.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Notes about Linux Process core dumps:
//  1) Linux core dump is stored as ELF file.
//  2) The ELF file's PT_NOTE and PT_LOAD segments describes the program's
//     address space and thread contexts.
//  3) PT_NOTE segment contains note entries which describes a thread context.
//  4) PT_LOAD segment describes a valid contiguous range of process address
//     space.
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSAMDGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSAMDGPUCORE_H

#include <amd-dbgapi/amd-dbgapi.h>
#include "Plugins/Process/elf-core/ProcessElfCore.h"

class ProcessAmdGpuCore : public ProcessElfCore {
public:
  // Constructors and Destructors
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "amdgpu-core"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  // Constructors and Destructors
  ProcessAmdGpuCore(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec &core_file);

  ~ProcessAmdGpuCore() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  // Creating a new process, or attaching to an existing one
  lldb_private::Status DoLoadCore() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:

  bool DoUpdateThreadList(lldb_private::ThreadList &old_thread_list,
                          lldb_private::ThreadList &new_thread_list) override;

private:
  bool initRocm();  

  amd_dbgapi_architecture_id_t m_architecture_id = AMD_DBGAPI_ARCHITECTURE_NONE;
  amd_dbgapi_process_id_t m_gpu_pid = AMD_DBGAPI_PROCESS_NONE;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSAMDGPUCORE_H
