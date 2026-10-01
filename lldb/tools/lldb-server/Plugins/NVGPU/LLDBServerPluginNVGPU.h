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
#include "llvm/ADT/StringMap.h"

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

  std::optional<GPUActions> GPUProcessIsStopping() override;

  void NativeProcessDidExit(const WaitStatus &exit_status) override;

private:
  /// Create a connection to the GPU process that the client can use.
  ///
  /// Establishes a communication channel between the debugger client and
  /// the GPU debugging infrastructure.
  ///
  /// \return
  ///     Expected connection info on success, or error on failure.
  llvm::Expected<GPUPluginConnectionInfo> CreateConnection();

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

  // ProcessNVGPU::Detach delegates to the private DetachCleanup.
  friend class ProcessNVGPU;

  /// Where the GPU stands between bringing the debugger API up and releasing
  /// it. The launch path stays at eNone until a detach.
  enum class AttachState {
    eNone,
    /// The driver has finished a late attach.
    eComplete,
    eDetaching,
  };

  void HandleInternalError(CUDBGResult error_type);
  void OnAttachComplete();

  /// Drain the event queue until empty, then acknowledge once, returning how
  /// many events were handled. One notification can cover several events, so
  /// draining one at a time would strand the rest.
  llvm::Expected<int> DrainSyncEventsOnce();

  /// Run \a work on the native (CPU) MainLoop thread and block until done, so
  /// that it can use the host server's process actions, which only that
  /// thread may use. On Linux, for example, ptrace accepts requests only from
  /// the thread that attached, so writing inferior memory from the GPU
  /// MainLoop thread fails with ESRCH, while reads go through process_vm_readv
  /// and do not, which makes this easy to miss. That thread is also the one
  /// that destroys the process when the inferior exits. The caller must not
  /// hold m_attach_mutex.
  llvm::Error RunOnNativeMainLoop(std::function<llvm::Error()> work);

  /// Have the host server stop the native process if it is running, which it
  /// then reports to the client, and return without waiting for the stop,
  /// which no caller needs. The base class's HaltNativeProcessIfNeeded instead
  /// polls the process from the calling thread, which reads freed memory if
  /// the inferior exits meanwhile.
  void HaltNativeProcess();

  /// Detach from the GPU when the client sends "D": tear down device
  /// breakpoints, have the driver clean up, and release the debugger API. If
  /// the driver needs the application running to clean up, the client keeps
  /// it running until this returns.
  void DetachCleanup();

  /// Drain events until CUDBG_EVENT_DETACH_COMPLETE, the API faults, the
  /// inferior exits, or kDetachMaxIterations is reached. We occupy the GPU
  /// MainLoop thread the notifier would dispatch on, so the event has to be
  /// drained here rather than awaited.
  void DrainUntilDetachComplete();

  /// Finalize and drop the debugger API along with everything holding a
  /// pointer into it, on detach and to undo an InitializeAPIAndConnect that
  /// failed partway.
  void ReleaseDebuggerAPI();

  /// Send a monitor log line to the client. Only valid while the GPU is
  /// running: the client is then waiting for a stop reply and will not mistake
  /// it for a query response.
  void SendMonitorLogToClient(llvm::StringRef message);

  llvm::Expected<GPUActions>
  InitializeAPIAndConnect(SymbolAddressProvider get_symbol_address,
                          llvm::StringRef libcuda_library_name,
                          bool is_late_attach);

  llvm::Error
  FinishLateAttachIpcHandshake(SymbolAddressProvider get_symbol_address);

  Status m_main_loop_status;
  std::optional<CUDADebuggerAPI> m_cuda_api;
  ProcessNVGPU *m_gpu = nullptr;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Set while the GPU stop that completes a late attach is being reported, so
  /// its stop reply asks the client to stop the CPU. Only used on the GPU
  /// MainLoop thread, which reports GPU stops.
  bool m_stop_native_with_gpu = false;

  /// Guards everything below, and is taken from both the native server thread
  /// and the GPU main loop thread. Must NOT be held across the host server's
  /// process actions, which block.
  std::mutex m_attach_mutex;
  AttachState m_attach_state = AttachState::eNone;

  /// libcuda symbol addresses the client resolved for the breakpoint that
  /// brought the API up. Detach uses them too.
  llvm::StringMap<uint64_t> m_libcuda_symbols;

  /// Not derivable from m_attach_state, which stays eNone on the launch path.
  bool m_api_initialized = false;

  /// Set by CUDBG_EVENT_INTERNAL_ERROR; acking or resuming on a poisoned API
  /// can wedge or crash the session.
  bool m_api_faulted = false;
  bool m_native_process_exited = false;

  static constexpr int kDetachMaxIterations = 100;
  static constexpr unsigned kNativeWorkTimeoutMs = 5000;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
