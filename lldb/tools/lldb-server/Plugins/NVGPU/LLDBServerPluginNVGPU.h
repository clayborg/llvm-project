//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H

#include "CUDADebuggerAPI.h"
#include "MainLoopEventNotifier.h"
#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "ProcessNVGPU.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

namespace lldb_private::lldb_server {

/// LLDB server plugin for NVIDIA GPU debugging support.
///
/// This effectively orchestrates the initialization of the NVIDIA debugger API
/// and the interaction between the CPU process, the GPU process and the
/// debugger API.
class LLDBServerPluginNVGPU : public LLDBServerPlugin {
public:
  /// Constructor for the NVIDIA GPU server plugin.
  ///
  /// \param[in] native_process
  ///     Reference to the GDB server managing the native process.
  ///
  /// \param[in] main_loop
  ///     Reference to the main event loop for handling asynchronous events.
  LLDBServerPluginNVGPU(LLDBServerPlugin::GDBServer &native_process,
                        MainLoop &main_loop);

  /// Get the name identifier for this plugin.
  ///
  /// \return
  ///     String reference containing the plugin name.
  llvm::StringRef GetPluginName() override;

  /// Get the initialization actions required for this plugin.
  ///
  /// \return
  ///     GPUActions structure containing the initialization steps.
  GPUActions GetInitializeActions() override;

  /// Handle breakpoint hit events from the GPU.
  ///
  /// Processes breakpoint events and determines the appropriate response
  /// action for the debugger.
  ///
  /// \param[in] args
  ///     Arguments containing details about the breakpoint hit.
  ///
  /// \return
  ///     Expected response indicating the action to take, or error if
  ///     the breakpoint could not be processed.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  /// Handle notification that the native process is stopping.
  ///
  /// \return
  ///     Optional GPUActions if specific actions need to be taken during
  ///     the stop process, or nullopt if no actions are required.
  std::optional<GPUActions> NativeProcessIsStopping() override;

  void NativeProcessDidExit(const WaitStatus &exit_status) override;

private:
  /// Create a connection to the GPU process that the client can use.
  ///
  /// Establishes a communication channel between the debugger client and
  /// the GPU debugging infrastructure.
  ///
  /// \param[in] before_accept
  ///     Called once the socket is listening and before the thread that
  ///     accepts the client starts. That thread cannot be stopped, and it
  ///     ends lldb-server if no client connects, so a step that can fail
  ///     after the socket exists belongs here. If it returns an error, no
  ///     thread is started and the socket is closed.
  ///
  /// \return
  ///     Expected connection info on success, or error on failure.
  llvm::Expected<GPUPluginConnectionInfo>
  CreateConnection(llvm::function_ref<llvm::Error()> before_accept);

  /// Function used to execute the main loop of the GPU process in an
  /// independent thread.
  ///
  /// This method runs the GPU process event handling loop in a separate
  /// thread to avoid blocking the main debugger execution.
  ///
  /// \param[in] listen_socket_up
  ///     Unique pointer to TCP socket for listening to client connections.
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  /// Process debugger API events.
  ///
  /// Handles events from the CUDA debugger API, processing them and
  /// taking appropriate action based on event type.
  void OnDebuggerAPIEvent();

  /// Stop using the debugger API after it reports an internal error, a call
  /// into it fails, or the device state it reports cannot be decoded, and stop
  /// the GPU process with \a description. lldb also gets \a description as an
  /// error if it is waiting for a stop reply, and the native process stops too
  /// unless NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1.
  ///
  /// \param[in] description
  ///     What went wrong, which the stop shows.
  void ReportDebuggerAPIFault(std::string description);

  /// Drain the event queue until empty, then acknowledge once, returning how
  /// many events were handled. One notification can cover several events, so
  /// draining one at a time would strand the rest.
  llvm::Expected<int> DrainSyncEvents();

  /// Finalize and drop the debugger API along with everything holding a
  /// pointer into it, to undo an InitializeAPIAndConnect that failed partway.
  void ReleaseDebuggerAPI();

  /// Bring the debugger API up from a breakpoint hit, which runs on the native
  /// server's thread, the only one that may use \a native_process.
  llvm::Expected<GPUActions>
  InitializeAPIAndConnect(const GPUPluginBreakpointHitArgs &args,
                          llvm::StringRef libcuda_library_name,
                          NativeProcessProtocol &native_process);

  Status m_main_loop_status;
  std::optional<CUDADebuggerAPI> m_cuda_api;
  ProcessNVGPU *m_gpu = nullptr;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Set once the debugger API is up. Only breakpoint hits use it, and they
  /// arrive on the native server's thread.
  bool m_api_initialized = false;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
