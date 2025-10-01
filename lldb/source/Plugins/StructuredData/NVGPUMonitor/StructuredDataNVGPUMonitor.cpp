//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructuredDataNVGPUMonitor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(StructuredDataNVGPUMonitor)

void StructuredDataNVGPUMonitor::Initialize() {
  PluginManager::RegisterPlugin(GetStaticPluginName(),
                                "NVIDIA GPU Monitor support", &CreateInstance,
                                &DebuggerInitialize, &FilterLaunchInfo);
}

void StructuredDataNVGPUMonitor::Terminate() {
  PluginManager::UnregisterPlugin(&CreateInstance);
}

bool StructuredDataNVGPUMonitor::SupportsStructuredDataType(
    llvm::StringRef type_name) {
  return type_name == GetStaticPluginName();
}

void StructuredDataNVGPUMonitor::HandleArrivalOfStructuredData(
    Process &process, llvm::StringRef type_name,
    const StructuredData::ObjectSP &object_sp) {
  if (StructuredData::Dictionary *dictionary = object_sp->GetAsDictionary()) {
    llvm::StringRef subtype;
    if (dictionary->GetValueForKeyAsString("subtype", subtype)) {
      if (subtype == "log") {
        llvm::StringRef message;
        dictionary->GetValueForKeyAsString("message", message);
        llvm::errs() << "NVIDIA GPU: " << message << "\n";
        return;
      }
    }
  }

  // If we didn't handle the message correctly, we dump it in JSON format.
  StreamString json_stream;
  if (object_sp)
    object_sp->Dump(json_stream);
  else
    json_stream.PutCString("<null>");

  llvm::errs() << json_stream.GetData() << "\n";
}

Status StructuredDataNVGPUMonitor::GetDescription(
    const StructuredData::ObjectSP &object_sp, Stream &stream) {
  return Status();
}

Status
StructuredDataNVGPUMonitor::FilterLaunchInfo(ProcessLaunchInfo &launch_info,
                                             Target *target) {
  return Status();
}

bool StructuredDataNVGPUMonitor::InitCompletionHookCallback(
    void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id,
    lldb::user_id_t break_loc_id) {
  return true;
}

StructuredDataPluginSP
StructuredDataNVGPUMonitor::CreateInstance(Process &process) {
  auto process_wp = ProcessWP(process.shared_from_this());
  return StructuredDataPluginSP(new StructuredDataNVGPUMonitor(process_wp));
}

void StructuredDataNVGPUMonitor::DebuggerInitialize(Debugger &debugger) {
  StructuredDataPlugin::InitializeBasePluginForDebugger(debugger);
}

StructuredDataNVGPUMonitor::StructuredDataNVGPUMonitor(
    const ProcessWP &process_wp)
    : StructuredDataPlugin(process_wp) {}
