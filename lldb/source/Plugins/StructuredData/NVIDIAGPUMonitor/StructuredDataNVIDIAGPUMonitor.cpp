//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructuredDataNVIDIAGPUMonitor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(StructuredDataNVIDIAGPUMonitor)

void StructuredDataNVIDIAGPUMonitor::Initialize() {
  PluginManager::RegisterPlugin(GetStaticPluginName(),
                                "NVIDIA GPU Monitor support", &CreateInstance,
                                &DebuggerInitialize, &FilterLaunchInfo);
}

void StructuredDataNVIDIAGPUMonitor::Terminate() {
  PluginManager::UnregisterPlugin(&CreateInstance);
}

bool StructuredDataNVIDIAGPUMonitor::SupportsStructuredDataType(
    llvm::StringRef type_name) {
  return type_name == GetStaticPluginName();
}

void StructuredDataNVIDIAGPUMonitor::HandleArrivalOfStructuredData(
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

Status StructuredDataNVIDIAGPUMonitor::GetDescription(
    const StructuredData::ObjectSP &object_sp, Stream &stream) {
  return Status();
}

Status
StructuredDataNVIDIAGPUMonitor::FilterLaunchInfo(ProcessLaunchInfo &launch_info,
                                                 Target *target) {
  return Status();
}

bool StructuredDataNVIDIAGPUMonitor::InitCompletionHookCallback(
    void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id,
    lldb::user_id_t break_loc_id) {
  return true;
}

StructuredDataPluginSP
StructuredDataNVIDIAGPUMonitor::CreateInstance(Process &process) {
  auto process_wp = ProcessWP(process.shared_from_this());
  return StructuredDataPluginSP(new StructuredDataNVIDIAGPUMonitor(process_wp));
}

void StructuredDataNVIDIAGPUMonitor::DebuggerInitialize(Debugger &debugger) {
  StructuredDataPlugin::InitializeBasePluginForDebugger(debugger);
}

StructuredDataNVIDIAGPUMonitor::StructuredDataNVIDIAGPUMonitor(
    const ProcessWP &process_wp)
    : StructuredDataPlugin(process_wp) {}
