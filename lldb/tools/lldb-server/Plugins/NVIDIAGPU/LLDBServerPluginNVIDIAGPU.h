//===-- LLDBServerPluginNVIDIAGPU.h ---------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVIDIAGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVIDIAGPU_H

#include "CUDADebuggerAPI.h"
#include "MainLoopEventNotifier.h"
#include "NVIDIAGPU.h"
#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

class TCPSocket;

namespace lldb_server {

/// The LLDB server plugin for NVIDIA GPUs.
///
/// This effectively orchestrates the initialization of the NVIDIA debugger API
/// and the interaction between the CPU process, the GPU process and the
/// debugger API.
class LLDBServerPluginNVIDIAGPU
    : public lldb_private::lldb_server::LLDBServerPlugin {
public:
  LLDBServerPluginNVIDIAGPU(
      lldb_private::lldb_server::LLDBServerPlugin::GDBServer &native_process,
      MainLoop &main_loop);

  llvm::StringRef GetPluginName() override;

  GPUActions GetInitializeActions() override;

  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  std::optional<GPUActions> NativeProcessIsStopping() override;

private:
  /// Create a connection to the GPU process that the client can use.
  llvm::Expected<GPUPluginConnectionInfo> CreateConnection();

  /// Function used to execute the main loop of the GPU process in an
  /// independent thread.
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  /// Process debugger API events.
  void OnDebuggerAPIEvent();

  Status m_main_loop_status;
  std::optional<CUDADebuggerAPI> m_cuda_api;
  NVIDIAGPU *m_gpu = nullptr;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVIDIAGPU_H
