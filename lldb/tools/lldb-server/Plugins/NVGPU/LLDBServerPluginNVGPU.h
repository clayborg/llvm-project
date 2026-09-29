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
  GPUActions GetInitializeActions(const GPUPluginInitializeArgs &args) override;
  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;
  std::optional<GPUActions> NativeProcessIsStopping() override;
  void NativeProcessDidExit(const WaitStatus &exit_status) override;
  std::vector<std::string> GetSymbolsToLookUp() override;
  void SymbolLookedUp(llvm::StringRef name,
                      std::optional<uint64_t> value) override;
  bool ShouldResumeToFinishAttach() override;

private:
  // ProcessNVGPU::Detach delegates to the private DetachCleanup.
  friend class ProcessNVGPU;

  /// Phases of the late attach handshake. The launch path stays at eNone.
  enum class AttachState {
    eNone,
    eProbing,
    eInjected,
    eComplete,
    eDetaching,
  };

  llvm::Expected<GPUPluginConnectionInfo> CreateConnection();
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);
  void OnDebuggerAPIEvent();
  void HandleInternalError(CUDBGResult error_type);
  void ScheduleInjectedPhaseWatchdog();
  void TryInitiateAttachServerSide();
  void SetAttachStateIfProbing(AttachState state);
  void OnAttachComplete();

  /// Drain the event queue until empty, then acknowledge once, returning how
  /// many events were handled. One notification can cover several events, so
  /// draining one at a time would strand the rest.
  llvm::Expected<int> DrainSyncEventsOnce();

  /// Run \a work on the native (CPU) MainLoop thread and block until done.
  ///
  /// Linux accepts ptrace requests only from the thread that attached, which
  /// here is the native MainLoop thread, so writing inferior memory or resuming
  /// the process from the GPU MainLoop thread fails with ESRCH. Reads go
  /// through process_vm_readv and have no such restriction, which makes this
  /// easy to miss. \a work gets the process as looked up on that thread, which
  /// is also the one that destroys it when the inferior exits. The caller must
  /// not hold m_attach_mutex.
  llvm::Error
  RunOnNativeMainLoop(std::function<llvm::Error(NativeProcessProtocol &)> work);

  void DetachCleanup();

  /// Step 3 of DetachCleanup: ask the driver to clean up, resume the devices
  /// and the application so it can, and drain until it reports completion.
  /// Best effort; every failure is logged and detach continues.
  void ResumeForDriverCleanup(CUDBGAPI api, uint32_t resume_flags);

  /// Drain events until CUDBG_EVENT_DETACH_COMPLETE, the API faults, the
  /// inferior exits, or kDetachMaxIterations is reached. We occupy the GPU
  /// MainLoop thread the notifier would dispatch on, so the event has to be
  /// drained here rather than awaited.
  void DrainUntilDetachComplete();

  /// Step 5 of DetachCleanup: clear the driver's handshake globals so a later
  /// debugger re-negotiates. Only valid once the driver has finished its own
  /// cleanup, which needs the IPC flag still set.
  void ResetDriverHandshakeFlags(const llvm::StringMap<uint64_t> &symbols);

  /// Step 6 of DetachCleanup: finalize and drop the debugger API along with
  /// everything holding a pointer into it.
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

  /// Guards everything below, and is taken from both the native server thread
  /// and the GPU main loop thread. Must NOT be held across blocking
  /// ptrace/procfs/FD work.
  std::mutex m_attach_mutex;
  AttachState m_attach_state = AttachState::eNone;
  std::chrono::steady_clock::time_point m_attach_deadline{};

  /// libcuda symbol addresses the client has resolved, from qSymbol for the
  /// safe attach handshake and from the initialization breakpoints. Detach
  /// uses them too.
  llvm::StringMap<uint64_t> m_libcuda_symbols;

  /// Not derivable from m_attach_state, which reaches eInjected when the magic
  /// byte is written -- before the API is brought up.
  bool m_api_initialized = false;

  /// Set by CUDBG_EVENT_INTERNAL_ERROR; acking or resuming on a poisoned API
  /// can wedge or crash the session.
  bool m_api_faulted = false;
  bool m_native_process_exited = false;

  /// The client resumed the process for us and is waiting for the stop that
  /// ends its attach, which we owe it once ours has finished or timed out.
  bool m_client_waiting_for_attach = false;

  static constexpr unsigned kAttachProbeTimeoutSeconds = 30;
  static constexpr unsigned kAttachInjectTimeoutSeconds = 60;
  static constexpr unsigned kAttachWaitTimeoutSeconds = 10;
  static constexpr int kDetachMaxIterations = 100;
  static constexpr unsigned kNativeWorkTimeoutMs = 5000;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
