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
  /// \param[in] args
  ///     Initialization context, including whether the native process is being
  ///     attached to versus launched.
  ///
  /// \return
  ///     GPUActions structure containing the initialization steps.
  GPUActions GetInitializeActions(const GPUPluginInitializeArgs &args) override;

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
  // ProcessNVGPU::Detach delegates the late-attach detach cleanup back to this
  // plugin via the private DetachCleanup entry point.
  friend class ProcessNVGPU;

  /// Phases of the late attach handshake. When attaching to an already-running
  /// CUDA process the driver must inject the debug engine at a safe point, so
  /// initialization is asynchronous and driven by this state machine. The
  /// launch path leaves this at eNone.
  enum class AttachState {
    /// Not attaching, or launch-based initialization. The cuInit-style
    /// breakpoint drives initialization.
    eNone,
    /// Attaching: we are probing the running process to discover whether CUDA
    /// is active and the safe-attach handler is available. Symbol resolution is
    /// retried on each native stop until the handshake can be initiated. A
    /// native stop reads this state under m_attach_mutex, then runs the
    /// (unlocked) safe-attach work -- resolving the driver symbols and writing
    /// the magic byte -- and on success commits eInjected. The native callbacks
    /// that drive this all run serially on the single LLGS MainLoop thread, so
    /// no second stop can run concurrently and the unlocked work is not
    /// re-entered; the GPU event/watchdog threads do not exist yet during this
    /// window (they are created later, in InitializeAPIAndConnect at the
    /// report-finished breakpoint).
    eProbing,
    /// The debugger API has been (or is being) initialized for this attach --
    /// either we wrote the magic byte to the attach-procedure FD and are
    /// waiting for the driver to call CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED,
    /// or CUDA initialized via the launch-style breakpoint during attach. In
    /// either case probing has stopped and we are waiting for
    /// CUDBG_EVENT_ATTACH_COMPLETE.
    eInjected,
    /// Attach is complete; device state has been refreshed.
    eComplete,
    /// A detach is in progress: breakpoints have been (or are being) torn down
    /// and the driver is being asked to clean up. The DETACH_COMPLETE event
    /// returns the state to eNone. Entered by DetachCleanup so the event-drain
    /// loop and CancelInProgressAttach can tell a deliberate detach apart from
    /// an active attach.
    eDetaching,
  };

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
  /// taking appropriate action based on event type. This is the callback wired
  /// to the async event notifier; it simply drives one drain pass.
  void OnDebuggerAPIEvent();

  /// Drain and handle one pass of the synchronous CUDA debugger API event
  /// queue, then acknowledge the events that were consumed.
  ///
  /// A single notification can correspond to several queued events, so this
  /// loops getNextEvent until the queue reports empty (or the
  /// CUDBG_EVENT_INVALID sentinel) and acknowledges once. It is shared by the
  /// async-notifier path (OnDebuggerAPIEvent) and the synchronous detach loop
  /// (DetachCleanup), which both run on the GPU MainLoop thread. It sets the
  /// m_detach_complete and m_api_faulted latches as those events are observed,
  /// and short-circuits (without further getNextEvent/acknowledge calls) once
  /// the API has been poisoned by an internal error.
  ///
  /// \return
  ///     The number of events handled in this pass.
  int DrainSyncEventsOnce();

  /// Handle CUDBG_EVENT_INTERNAL_ERROR: poison the API, force a clean GPU stop
  /// and surface a structured error to the client without aborting lldb-server.
  ///
  /// \param[in] error_type
  ///     The CUDBGResult carried by the internal-error event.
  void HandleInternalError(CUDBGResult error_type);

  /// Run \a work on the native (CPU) MainLoop thread and block until it
  /// finishes.
  ///
  /// Linux accepts ptrace requests only from the thread that attached to the
  /// tracee, which for lldb-server is the native MainLoop thread. Anything that
  /// reaches PTRACE_POKEDATA or PTRACE_CONT -- writing inferior memory, or
  /// resuming the process -- therefore fails with ESRCH when issued from the
  /// GPU MainLoop thread, where the detach path runs. (Memory *reads* happen to
  /// work because they go through process_vm_readv, which has no such
  /// restriction, so the failure is easy to miss.)
  ///
  /// The caller must not hold m_attach_mutex, since the native thread may need
  /// it. The shared state is kept alive independently of this call, so a
  /// timeout cannot leave the callback writing to freed memory.
  ///
  /// \param[in] work
  ///     The work to run on the native MainLoop thread.
  /// \param[in] timeout
  ///     How long to wait before giving up on the work being run.
  ///
  /// \return
  ///     The error returned by \a work, or an error if it could not be
  ///     scheduled or did not run within \a timeout.
  llvm::Error RunOnNativeMainLoop(std::function<llvm::Error()> work,
                                  std::chrono::milliseconds timeout);

  /// Plugin-owned detach cleanup: tear down device breakpoints, reset the
  /// driver handshake flags, optionally resume the application so the driver can
  /// finish its cleanup (draining the resulting events inline), then clear the
  /// attach state. Driven from ProcessNVGPU::Detach. Runs on the GPU MainLoop
  /// thread, so it drains events inline rather than waiting on the notifier.
  ///
  /// \return
  ///     Error::success() on a clean detach, or an error describing what went
  ///     wrong (the session is still torn down on a best-effort basis).
  llvm::Error DetachCleanup();

  /// Cancel an attach that is still in progress (from the interrupt path or the
  /// probe watchdog). Idempotent against OnAttachComplete via the state machine.
  void CancelInProgressAttach();

  /// Schedule a one-shot CPU-MainLoop timer that re-probes the running process
  /// for a usable safe-attach handler while still eProbing, re-arming itself
  /// until the state leaves eProbing. This makes late attach progress even when
  /// the inferior produces no further native stops.
  void ScheduleAttachProbe();

  /// \return the probing-phase timeout in seconds, overridable at runtime via
  /// the NVGPU_ATTACH_PROBE_TIMEOUT_SECONDS environment variable.
  unsigned GetAttachProbeTimeoutSeconds() const;

  /// Initialize the CUDA debugger API using the given symbol address resolver
  /// and libcuda library, wire up the event notifier, and create a reverse
  /// connection for the client. Shared by the launch (breakpoint) path and the
  /// attach (report-finished breakpoint) path.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  /// \param[in] libcuda_library_name
  ///     The libcuda library to load the debugger API from.
  /// \param[in] is_late_attach
  ///     True when this initialization is the late attach path (the
  ///     report-finished breakpoint), false for launch-style initialization.
  ///     Controls how the driver's IPC "client ready" flag is sequenced; see
  ///     FinishLateAttachIpcHandshake.
  ///
  /// \return
  ///     GPUActions carrying the connection info on success, or an error.
  llvm::Expected<GPUActions>
  InitializeAPIAndConnect(SymbolAddressProvider get_symbol_address,
                          llvm::StringRef libcuda_library_name,
                          bool is_late_attach);

  /// Perform the post-injection half of the late attach handshake, once the
  /// debugger API is up and the new-event callback is registered.
  ///
  /// Publishes the driver's IPC flag (see CUDADebuggerAPI::SetIpcFlag), then
  /// reads CUDBG_RESUME_FOR_ATTACH_DETACH to decide how the attach finishes:
  ///
  ///   - set: the driver has pre-existing contexts and modules to replay and
  ///     needs the application to keep running to do it. The attach finishes
  ///     when CUDBG_EVENT_ATTACH_COMPLETE arrives.
  ///   - clear: there is nothing to replay, so no CUDBG_EVENT_ATTACH_COMPLETE
  ///     will ever arrive and this completes the attach itself.
  ///
  /// \param[in] get_symbol_address
  ///     Resolver for native-process symbol load addresses.
  ///
  /// \return
  ///     Error::success() once the correct branch has been taken, or an error
  ///     if the IPC flag could not be written or the resume flag not read.
  llvm::Error
  FinishLateAttachIpcHandshake(SymbolAddressProvider get_symbol_address);

  /// Drive the late attach handshake entirely server-side: resolve the driver
  /// handshake symbols in the inferior (no gdb-remote round-trip), probe whether
  /// the safe-attach handler is available, and if so initiate the safe
  /// injection. Called on native stops while attaching; safe to call repeatedly
  /// (a no-op once the handshake has been initiated, and it retries on later
  /// stops if libcuda is not resolvable yet).
  void TryInitiateAttachServerSide();

  /// Handle CUDBG_EVENT_ATTACH_COMPLETE: refresh device state so CUDA threads
  /// are enumerated and reported to the client.
  void OnAttachComplete();

  /// Surface a stuck post-injection handshake. The caller must hold
  /// m_attach_mutex. If we are still in eInjected past m_attach_deadline, log
  /// the wedge warning. Invoked from the one-shot MainLoop watchdog timer (the
  /// single injected-phase deadline surface), which fires once, so no dedup
  /// latch is needed.
  void CheckInjectedPhaseTimeoutLocked();

  /// Schedule a one-shot MainLoop timer that checks the injected-phase deadline.
  /// This is the sole injected-phase deadline surface: it runs on the GPU
  /// MainLoop precisely while waiting for CUDBG_EVENT_ATTACH_COMPLETE. Safe to
  /// call from any thread (MainLoopBase::AddCallback is thread-safe).
  void ScheduleInjectedPhaseWatchdog();

  Status m_main_loop_status;
  std::optional<CUDADebuggerAPI> m_cuda_api;
  ProcessNVGPU *m_gpu = nullptr;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Guards the late attach state machine, which is touched from both the CPU
  /// server thread (GetInitializeActions / NativeProcessIsStopping /
  /// TryInitiateAttachServerSide) and the GPU main loop thread
  /// (OnDebuggerAPIEvent). It must NOT be held across blocking ptrace/procfs/FD
  /// work; TryInitiateAttachServerSide only holds it to read and commit state
  /// transitions.
  std::mutex m_attach_mutex;
  AttachState m_attach_state = AttachState::eNone;

  /// Deadline for the current attach phase (probing or injected). If the phase
  /// has not progressed by this time we surface an actionable error instead of
  /// hanging silently: probing falls back to launch-style initialization, and a
  /// stuck post-injection handshake is logged. Guarded by m_attach_mutex.
  std::chrono::steady_clock::time_point m_attach_deadline{};

  /// True once the debugger API has been initialized and connected. Guards
  /// BreakpointWasHit against initializing a second, live API (which would
  /// finalize/leak the first) if an initialization breakpoint fires again.
  /// Guarded by m_attach_mutex.
  ///
  /// m_attach_state is the source of truth for the attach phase; this bool is
  /// deliberately separate and is NOT derivable from the enum: the late-attach
  /// path enters eInjected when the magic byte is written, BEFORE the API is
  /// actually brought up (that happens later, when the report-finished
  /// breakpoint fires). m_cuda_api.has_value() is also not a safe substitute
  /// because m_cuda_api is assigned unlocked in InitializeAPIAndConnect. This
  /// flag is the one explicit, lock-published "the API object is live" guard.
  bool m_api_initialized = false;

  /// Set once the debugger API has reported a CUDBG_EVENT_INTERNAL_ERROR. The
  /// API is then considered poisoned: the event-drain loop and detach path stop
  /// calling into it (acking/resuming on a faulted API can wedge or crash the
  /// session). Guarded by m_attach_mutex.
  bool m_api_faulted = false;

  /// Latch set by the CUDBG_EVENT_DETACH_COMPLETE handler so the synchronous
  /// detach drain loop (DetachCleanup) can tell when the driver has finished its
  /// cleanup. Reset at the start of each detach. Guarded by m_attach_mutex.
  bool m_detach_complete = false;

  /// Bounds on how long the late attach phases may run before we stop waiting.
  static constexpr unsigned kAttachProbeTimeoutSeconds = 30;
  static constexpr unsigned kAttachInjectTimeoutSeconds = 60;

  /// How often the probe watchdog re-checks the running process for a usable
  /// safe-attach handler while eProbing (gap 9 probe cadence).
  static constexpr unsigned kAttachProbeIntervalSeconds = 1;

  /// Maximum number of inline drain iterations the detach cleanup loop performs
  /// while waiting for CUDBG_EVENT_DETACH_COMPLETE before giving up. Mirrors
  /// cuda-gdb's remote sync detach loop bound.
  static constexpr int kDetachMaxIterations = 100;

  /// How long the detach path waits for a piece of ptrace-bound work to run on
  /// the native MainLoop thread (see RunOnNativeMainLoop).
  static constexpr unsigned kNativeWorkTimeoutMs = 5000;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
