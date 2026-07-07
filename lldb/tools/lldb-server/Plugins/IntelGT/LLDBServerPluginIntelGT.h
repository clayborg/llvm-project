//===-- LLDBServerPluginIntelGT.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LLDBSERVERPLUGININTELGT_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LLDBSERVERPLUGININTELGT_H

#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "ProcessIntelGT.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringRef.h"

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace lldb_private {

class TCPSocket;

namespace lldb_server {

/// Thin IOObject wrapper over a plain file descriptor; used to register the
/// Level Zero wakeup pipe with the main loop.
class GPUIOObjectIntelGT : public IOObject {
public:
  explicit GPUIOObjectIntelGT(int fd) : IOObject(eFDTypeSocket), m_fd(fd) {}

  Status Read(void * /*buf*/, size_t & /*num_bytes*/) override {
    return Status();
  }
  Status Write(const void * /*buf*/, size_t & /*num_bytes*/) override {
    return Status();
  }
  bool IsValid() const override { return m_fd >= 0; }
  Status Close() override { return Status(); }
  WaitableHandle GetWaitableHandle() override { return m_fd; }

private:
  int m_fd = -1;
};

/// LLDBServerPlugin that enables hybrid CPU+GPU debugging for Intel GPUs via
/// the Level Zero EU debugger API. All Level Zero calls must run on the
/// ptrace thread (BreakpointWasHit / NativeProcessIsStopping).
class LLDBServerPluginIntelGT : public LLDBServerPlugin {
public:
  LLDBServerPluginIntelGT(LLDBServerPlugin::GDBServer &native_process,
                          MainLoop &main_loop);
  ~LLDBServerPluginIntelGT() override;

  // ----- LLDBServerPlugin interface -----------------------------------------

  llvm::StringRef GetPluginName() override { return "intelgt"; }

  std::optional<LLDBSettings> GetLLDBSettings() override {
    LLDBSettings s;
    s.dyld_plugin_name = "gdb-remote-gpu";
    s.gpu_plugin_name = "intelgt";
    // Route GetGPUDynamicLoaderLibraryInfos to the GPU server, not the CPU.
    s.send_dyld_packet_to_gpu = true;
    return s;
  }

  GPUActions GetInitializeActions() override;

  std::optional<GPUActions> NativeProcessIsStopping() override;

  void NativeProcessDidExit(const WaitStatus &exit_status) override;

  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  std::optional<GPUDynamicLoaderResponse>
  GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) override;

  // ----- Accessors ----------------------------------------------------------

  NativeProcessProtocol *GetNativeProcess() {
    return m_native_process.GetCurrentProcess();
  }

  ProcessIntelGT *GetGPUProcess() {
    return static_cast<ProcessIntelGT *>(m_gdb_server->GetCurrentProcess());
  }

  /// Re-arm ZE event polling from ProcessIntelGT::Resume().
  void TriggerNotifier();

  /// Deliver wait_for_gpu_process_to_stop=true on the next CPU stop.
  void SetPendingGPUStop() { m_pending_gpu_stop = true; }

  /// Wait for this many EU threads to stop before reporting to LLDB.
  void SetExpectedStoppedThreadCount(size_t count) {
    m_expected_stopped_count = count;
  }

private:
  /// Initialise the Level Zero runtime and enumerate Intel GPU devices.
  Status InitializeLevelZero();

  /// Attach to all discovered devices and create ProcessIntelGT.
  Status AttachToDevices();

  /// Attempt to attach to a single device (or sub-device).
  bool AttachToDevice(ze_device_handle_t device, uint32_t device_index,
                      uint64_t tid_base, uint64_t max_threads,
                      DeviceSession &session_out);

  /// Create the GPU process and fake-launch it.
  Status CreateGpuProcess();

  /// Set up the wakeup pipe and register its read end with the main loop.
  Status InstallNotifierOnMainLoop();

  /// Create the reverse TCP connection and return connection info.
  std::optional<GPUPluginConnectionInfo> CreateConnection();

  /// Drain all pending ZE events from all device sessions (non-blocking);
  /// returns true if any GPU thread stopped or modules changed.
  bool DrainZeEvents(GPUActions &actions);

  /// Re-arm the notifier pipe so the main loop wakes up again.
  void RearmNotifier();

  /// Dedicated thread that blocks on zetDebugReadEvent(50ms) per session,
  /// queues events, and wakes the main loop via TriggerNotifier.
  void StartZeEventPollThread();
  void StopZeEventPollThread();

  struct ZeQueuedEvent {
    uint32_t device_index;
    zet_debug_session_handle_t session;
    zet_debug_event_t event;
  };
  std::vector<ZeQueuedEvent> m_ze_event_queue;
  std::mutex m_ze_event_queue_mutex;

  std::thread m_ze_poll_thread;
  std::atomic<bool> m_ze_poll_stop{false};
  /// Set when MODULE_LOAD arrives; cleared by NativeProcessIsStopping.
  bool m_pending_library_load = false;
  /// Set when THREAD_STOPPED fires; cleared by NativeProcessIsStopping.
  bool m_pending_gpu_stop = false;
  /// Set when CPU is halted for GPU stop delivery; cleared after auto-resume.
  bool m_cpu_halted_for_gpu_stop = false;
  /// Set when MODULE_LOAD should trigger auto-continue after loading.
  bool m_auto_continue_after_library_load = false;
  /// Set when GPU should auto-resume after LLDB client connects.
  bool m_auto_resume_after_attach = false;
  /// Number of EU threads DrainZeEvents will wait for before halting.
  size_t m_expected_stopped_count = 0;

  /// Stale members from removed background-attach thread.
  std::thread m_attach_thread;
  std::atomic<bool> m_attach_done{false};
  std::atomic<bool> m_attach_failed{false};
  std::optional<GPUPluginConnectionInfo> m_pending_connect_info;
  bool m_pending_load_libraries = false;

  /// State machine.
  enum class ZeState {
    Uninitialized,
    Initialized,
    Attached,
    RuntimeLoaded,
    Detached,
    Error,
  };
  ZeState m_ze_state = ZeState::Uninitialized;

  /// Discovered device sessions (one per attached device or sub-device).
  std::vector<DeviceSession> m_device_sessions;

  /// Notifier pipe; read end registered on the main loop.
  static constexpr int INVALID_FD = -1;
  int m_notifier_fd[2] = {INVALID_FD, INVALID_FD};

  std::shared_ptr<GPUIOObjectIntelGT> m_gpu_event_io_obj_sp;
  MainLoopBase::ReadHandleUP m_gpu_event_read_up;
  std::vector<MainLoopBase::ReadHandleUP> m_read_handles;
  std::unique_ptr<TCPSocket> m_listen_socket;

  /// Whether the trigger breakpoint was already hit (one-shot).
  bool m_zemodulecreate_hit = false;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LLDBSERVERPLUGININTELGT_H
