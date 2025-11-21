//===-- ThreadAMDGPU.h --------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_THREADAMDGPU_H
#define LLDB_TOOLS_LLDB_SERVER_THREADAMDGPU_H

#include "RegisterContextAmdGpu.h"
#include "lldb/Target/Thread.h"
#include "lldb/lldb-private-forward.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <string>

namespace lldb_private {

class NativeProcessLinux;

class ThreadAMDGPU : public Thread {
public:
  ThreadAMDGPU(Process &process, amd_dbgapi_architecture_id_t architecture_id,
               lldb::tid_t tid,
               std::optional<amd_dbgapi_wave_id_t> wave_id = std::nullopt)
      : Thread(process, tid), m_architecture_id(architecture_id),
        m_wave_id(wave_id) {}

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  const char *GetName() override;

  void SetName(const char *name) override {
    if (name && name[0])
      m_thread_name.assign(name);
    else
      m_thread_name.clear();
  }

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  GetSiginfo(size_t max_size) const override;

  std::optional<amd_dbgapi_wave_id_t> GetWaveId() const { return m_wave_id; }

  amd_dbgapi_architecture_id_t GetArchitectureId() const {
    return m_architecture_id;
  }

protected:
  bool CalculateStopInfo() override;

private:
  // Member Variables
  std::string m_thread_name;
  amd_dbgapi_architecture_id_t m_architecture_id;
  std::optional<amd_dbgapi_wave_id_t> m_wave_id;
};
} // namespace lldb_private

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_THREADAMDGPU_H
