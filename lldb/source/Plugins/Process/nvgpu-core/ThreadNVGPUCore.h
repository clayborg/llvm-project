//===-- ThreadNVGPUCore.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H

#include "NVGPUCoreData.h"
#include "lldb/Target/Thread.h"

class ProcessNVGPUCore;

namespace lldb_private {

class ThreadNVGPUCore : public Thread {
public:
  ThreadNVGPUCore(Process &process, lldb::tid_t tid,
                  const NVGPULaneCoords &coords);

  ~ThreadNVGPUCore() override;

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

  const char *GetName() override;

  const NVGPULaneCoords &GetCoords() const { return m_coords; }

protected:
  bool CalculateStopInfo() override;

private:
  NVGPULaneCoords m_coords;
  std::string m_name;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_NVGPU_CORE_THREADNVGPUCORE_H
