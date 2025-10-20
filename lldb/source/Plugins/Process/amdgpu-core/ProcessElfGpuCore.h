//===-- ProcessAmdGpuCore.h -----------------------------------*- C++ -*-===//
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

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H

#include "../elf-core/ProcessElfCore.h"

class ProcessElfGpuCore : public ProcessElfCore {
public:
  // Constructors and Destructors
  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  // Constructors and Destructors
  ProcessElfGpuCore(lldb::TargetSP target_sp, ProcessElfCore* cpu_core_process,
                    lldb::ListenerSP listener_sp,
                    const lldb_private::FileSpec &core_file): ProcessElfCore(target_sp, listener_sp, core_file),
      m_cpu_core_process(cpu_core_process) {}

  static llvm::Expected<std::shared_ptr<ProcessElfGpuCore>>
  LoadGpuCore(ProcessElfCore *cpu_core_process,
              const lldb_private::FileSpec &core_file);

  ProcessElfCore *GetCpuProcess() { return m_cpu_core_process; }
protected:
  ProcessElfCore *m_cpu_core_process;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_PROCESSELFGPUCORE_H
