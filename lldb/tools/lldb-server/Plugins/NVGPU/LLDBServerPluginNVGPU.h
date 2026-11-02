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

#include <atomic>
#include <thread>

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

  /// Stop the GPU MainLoop thread and finalize the debugger API before any
  /// member is destroyed, since that thread and the driver's event callback
  /// both use this object.
  ~LLDBServerPluginNVGPU() override;

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
  ///     accepts the client starts, which waits for a client until the
  ///     plug-in shuts down. A step that can fail after the socket exists
  ///     belongs here. If it returns an error, no thread is started and the
  ///     socket is closed.
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

  /// \return
  ///     The GPU process, or null once the GPU server has forgotten it, which
  ///     it does on the GPU MainLoop thread when the GPU process exits or lldb
  ///     detaches from it. Other threads may call this only before that thread
  ///     starts or after it ends.
  ProcessNVGPU *GetGPUProcess() const;

  /// Stop using the debugger API after it reports an internal error, a call
  /// into it fails, or the device state it reports cannot be decoded, and stop
  /// the GPU process with \a description. lldb also gets \a description as an
  /// error if it is waiting for a stop reply, and the native process stops too
  /// unless NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1.
  ///
  /// \param[in] description
  ///     What went wrong, which the stop shows.
  void ReportDebuggerAPIFault(std::string description);

  /// Drain the event queue until it is empty or a stop is reported, then
  /// acknowledge once, returning how many events were handled. One
  /// notification can cover several events, so draining one at a time would
  /// strand the rest. While the GPU process is stopped, nothing is drained,
  /// for the reasons in the class comment of ProcessNVGPU.
  llvm::Expected<int> DrainSyncEvents();

  /// Finalize and drop the debugger API along with everything holding a
  /// pointer into it, on destruction, and to undo an InitializeAPIAndConnect
  /// that failed partway.
  void ReleaseDebuggerAPI();

  /// Let go of a debugger API that lldb never connected to serve, without
  /// finalizing it. The driver keeps the application waiting until the events
  /// it raised are acknowledged, and a finalize meanwhile can crash it, as can
  /// one just after an acknowledgment lets the application go on. The
  /// application goes on by itself once lldb-server exits.
  void ReleaseUnservedDebuggerAPI();

  /// Take from the GPU process, if the GPU server still has it, everything
  /// that points into the debugger API or back at this object: the API
  /// table, the event drain notifier and the fault handler.
  void UnhookGPUProcess();

  /// Bring the debugger API up from a breakpoint hit, which runs on the native
  /// server's thread, the only one that may use \a native_process.
  llvm::Expected<GPUActions>
  InitializeAPIAndConnect(const GPUPluginBreakpointHitArgs &args,
                          llvm::StringRef libcuda_library_name,
                          NativeProcessProtocol &native_process);

  Status m_main_loop_status;

  /// Runs AcceptAndMainLoopThread. Only used on the native server's thread,
  /// which creates the connection and destroys this object.
  std::thread m_gpu_thread;

  /// Set by the destructor so that the GPU thread stops waiting for the client
  /// to connect.
  std::atomic<bool> m_shutting_down = false;

  std::optional<CUDADebuggerAPI> m_cuda_api;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Set once the debugger API is up. Only breakpoint hits use it, and they
  /// arrive on the native server's thread.
  bool m_api_initialized = false;

  /// Set on the native server's thread when the native process exits, before
  /// the GPU MainLoop thread reports the exit to lldb. From then on, the
  /// debugger API's events are left alone: acknowledging the events of an
  /// application that has exited can crash the debugger library.
  std::atomic<bool> m_native_process_exited = false;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
