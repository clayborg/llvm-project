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

#include <chrono>

namespace lldb_private::lldb_server {

/// LLDB server plugin for NVIDIA GPU debugging support.
///
/// This effectively orchestrates the initialization of the NVIDIA debugger API
/// and the interaction between the CPU process, the GPU process and the
/// debugger API.
class LLDBServerPluginNVGPU : public LLDBServerPlugin {
public:
  LLDBServerPluginNVGPU(LLDBServerPlugin::GDBServer &native_process,
                        MainLoop &main_loop);

  llvm::StringRef GetPluginName() override;

  /// Set the breakpoints that drive initialization: the cuInit-style ones for
  /// launch, plus the attach-procedure-finished one when \a args says we are
  /// attaching.
  GPUActions GetInitializeActions(const GPUPluginInitializeArgs &args) override;

  /// Bring the debugger API up. Both initialization breakpoints land here; the
  /// debug engine is present by the time either fires.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  std::optional<GPUActions> NativeProcessIsStopping() override;

  void NativeProcessDidExit(const WaitStatus &exit_status) override;

private:
  // ProcessNVGPU::Detach delegates to the private DetachCleanup.
  friend class ProcessNVGPU;

  /// Phases of the late attach handshake. The launch path stays at eNone.
  enum class AttachState {
    /// Not attaching, or launch-based initialization.
    eNone,
    /// Probing the running process for a usable safe-attach handler, retrying
    /// on each native stop until the handshake can be initiated.
    eProbing,
    /// The safe attach has been initiated (or CUDA came up via the launch-style
    /// breakpoint during an attach); waiting for the driver.
    eInjected,
    /// Attach is complete; device state has been refreshed.
    eComplete,
    /// A detach is in progress. CUDBG_EVENT_DETACH_COMPLETE returns it to
    /// eNone, which is how the event-drain loop tells a deliberate detach apart
    /// from an active attach.
    eDetaching,
  };

  /// Create a connection to the GPU process that the client can use.
  llvm::Expected<GPUPluginConnectionInfo> CreateConnection();

  /// Run the GPU process's main loop on its own thread, so it does not block
  /// the native one.
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  /// Async event-notifier callback; drives one drain pass.
  void OnDebuggerAPIEvent();

  /// Drain the synchronous event queue until it reports empty, then acknowledge
  /// once. A single notification can cover several queued events, so draining
  /// one at a time would strand the rest.
  ///
  /// \return
  ///     The number of events handled in this pass.
  int DrainSyncEventsOnce();

  /// Handle CUDBG_EVENT_INTERNAL_ERROR: poison the API, force a clean GPU stop
  /// and surface a structured error to the client without aborting lldb-server.
  void HandleInternalError(CUDBGResult error_type);

  /// Run \a work on the native (CPU) MainLoop thread and block until it
  /// finishes.
  ///
  /// Linux accepts ptrace requests only from the thread that attached, which
  /// for lldb-server is the native MainLoop thread. Writing inferior memory or
  /// resuming the process therefore fails with ESRCH when issued from the GPU
  /// MainLoop thread, where the detach path runs. Memory *reads* go through
  /// process_vm_readv and have no such restriction, which makes the failure
  /// easy to miss.
  ///
  /// The caller must not hold m_attach_mutex; the native thread may need it.
  llvm::Error RunOnNativeMainLoop(std::function<llvm::Error()> work,
                                  std::chrono::milliseconds timeout);

  /// Tear down device breakpoints, let the driver clean up, reset its handshake
  /// globals and finalize the API. Driven from ProcessNVGPU::Detach on the GPU
  /// MainLoop thread, so it drains events inline rather than via the notifier.
  ///
  /// \return
  ///     Error::success() on a clean detach, or an error describing what went
  ///     wrong (the session is torn down either way).
  llvm::Error DetachCleanup();

  /// Re-probe for a usable safe-attach handler on a timer, re-arming until the
  /// state leaves eProbing, so attach progresses even if the inferior produces
  /// no further native stops.
  void ScheduleAttachProbe();

  /// Initialize the debugger API, wire up the event notifier and create the
  /// reverse connection. Shared by the launch and late attach paths, which
  /// differ only in how the driver's IPC flag is sequenced (see
  /// FinishLateAttachIpcHandshake).
  llvm::Expected<GPUActions>
  InitializeAPIAndConnect(SymbolAddressProvider get_symbol_address,
                          llvm::StringRef libcuda_library_name,
                          bool is_late_attach);

  /// Finish the late attach handshake once the API is up and the event callback
  /// is registered: publish the IPC flag, then use
  /// CUDBG_RESUME_FOR_ATTACH_DETACH to decide whether the driver will replay
  /// pre-existing state and end with CUDBG_EVENT_ATTACH_COMPLETE, or whether
  /// there is nothing to replay and the attach completes here.
  llvm::Error
  FinishLateAttachIpcHandshake(SymbolAddressProvider get_symbol_address);

  /// Resolve the driver handshake symbols and initiate the safe attach, all
  /// server-side: a gdb-remote round-trip during attach stop processing would
  /// corrupt the in-flight continue cycle. Called on native stops while
  /// attaching; a no-op once the handshake has been initiated, and it retries
  /// on later stops while libcuda is not resolvable yet.
  void TryInitiateAttachServerSide();

  /// Handle CUDBG_EVENT_ATTACH_COMPLETE: refresh device state so CUDA threads
  /// are enumerated and reported to the client.
  void OnAttachComplete();

  /// Schedule the one-shot timer that warns if the post-injection handshake
  /// never completes.
  void ScheduleInjectedPhaseWatchdog();

  Status m_main_loop_status;
  std::optional<CUDADebuggerAPI> m_cuda_api;
  ProcessNVGPU *m_gpu = nullptr;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Guards everything below. Taken from both the native server thread and the
  /// GPU main loop thread. Must NOT be held across blocking ptrace/procfs/FD
  /// work.
  std::mutex m_attach_mutex;
  AttachState m_attach_state = AttachState::eNone;

  /// Deadline for the current attach phase. Probing falls back to launch-style
  /// initialization when it expires; a stuck injected phase is logged.
  std::chrono::steady_clock::time_point m_attach_deadline{};

  /// True once the API is live. Not derivable from m_attach_state, which
  /// reaches eInjected when the magic byte is written -- before the API is
  /// brought up at the report-finished breakpoint.
  bool m_api_initialized = false;

  /// Set by CUDBG_EVENT_INTERNAL_ERROR. The API is then poisoned: acking or
  /// resuming on it can wedge or crash the session.
  bool m_api_faulted = false;

  /// Set by CUDBG_EVENT_DETACH_COMPLETE so the detach drain loop can stop.
  bool m_detach_complete = false;

  static constexpr unsigned kAttachProbeTimeoutSeconds = 30;
  static constexpr unsigned kAttachInjectTimeoutSeconds = 60;
  static constexpr unsigned kAttachProbeIntervalSeconds = 1;

  /// Bound on the detach drain loop, mirroring cuda-gdb's.
  static constexpr int kDetachMaxIterations = 100;

  /// How long to wait for ptrace-bound work on the native MainLoop thread.
  static constexpr unsigned kNativeWorkTimeoutMs = 5000;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
