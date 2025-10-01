//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_STRUCTUREDDATA_NVGPUMONITOR_STRUCTUREDDATANVGPUMONITOR_H
#define LLDB_SOURCE_PLUGINS_STRUCTUREDDATA_NVGPUMONITOR_STRUCTUREDDATANVGPUMONITOR_H

#include "lldb/Target/StructuredDataPlugin.h"

namespace lldb_private {

class StructuredDataNVGPUMonitor : public StructuredDataPlugin {

public:
  // Public static API

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetStaticPluginName() { return "nvgpu-monitor"; }

  llvm::StringRef GetPluginName() override { return GetStaticPluginName(); }

  bool SupportsStructuredDataType(llvm::StringRef type_name) override;

  void HandleArrivalOfStructuredData(
      Process &process, llvm::StringRef type_name,
      const StructuredData::ObjectSP &object_sp) override;

  Status GetDescription(const StructuredData::ObjectSP &object_sp,
                        lldb_private::Stream &stream) override;

private:
  StructuredDataNVGPUMonitor(const lldb::ProcessWP &process_wp);

  static lldb::StructuredDataPluginSP CreateInstance(Process &process);

  static void DebuggerInitialize(Debugger &debugger);

  static bool InitCompletionHookCallback(void *baton,
                                         StoppointCallbackContext *context,
                                         lldb::user_id_t break_id,
                                         lldb::user_id_t break_loc_id);

  static Status FilterLaunchInfo(ProcessLaunchInfo &launch_info,
                                 Target *target);
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_STRUCTUREDDATA_NVGPUMONITOR_STRUCTUREDDATANVGPUMONITOR_H
