//===-- DAPSessionManager.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
#define LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
#include "lldb/lldb-types.h"
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace lldb_dap {

// Forward declarations
struct DAP; 

class ManagedEventThread {
public:
  // Constructor declaration
  ManagedEventThread(lldb::SBBroadcaster broadcaster, std::thread t);

  ~ManagedEventThread();

  ManagedEventThread(const ManagedEventThread &) = delete;
  ManagedEventThread &operator=(const ManagedEventThread &) = delete;

private:
  lldb::SBBroadcaster m_broadcaster;
  std::thread m_event_thread;
};

/// Global DAP session manager
class DAPSessionManager {
public:
  /// Get the singleton instance of the DAP session manager
  static DAPSessionManager &GetInstance();

  /// Register a DAP session
  void RegisterSession(lldb::IOObjectSP io, DAP *dap);

  /// Unregister a DAP session
  void UnregisterSession(lldb::IOObjectSP io);

  /// Get all active DAP sessions
  std::vector<DAP *> GetActiveSessions();

  /// Disconnect all active sessions
  void DisconnectAllSessions();

  /// Wait for all sessions to finish disconnecting
  void WaitForAllSessionsToDisconnect();

  /// Set the shared debugger instance for a specific target index
  void SetSharedDebugger(uint32_t target_idx, lldb::SBDebugger debugger);

  /// Get the shared debugger instance for a specific target index
  std::optional<lldb::SBDebugger> GetSharedDebugger(uint32_t target_idx);

  /// Get or create event thread for a specific debugger
  std::shared_ptr<ManagedEventThread>
  GetEventThreadForDebugger(lldb::SBDebugger debugger, DAP *requesting_dap);

  /// Find the DAP instance that owns the given target
  DAP *FindDAPForTarget(lldb::SBTarget target);

  /// Clean up shared resources when the last session exits
  void CleanupSharedResources();

  /// Clean up expired event threads from the collection
  void ReleaseExpiredEventThreads();

private:
  DAPSessionManager() = default;
  ~DAPSessionManager() = default;

  // Non-copyable and non-movable
  DAPSessionManager(const DAPSessionManager &) = delete;
  DAPSessionManager &operator=(const DAPSessionManager &) = delete;
  DAPSessionManager(DAPSessionManager &&) = delete;
  DAPSessionManager &operator=(DAPSessionManager &&) = delete;

  std::mutex m_sessions_mutex;
  std::condition_variable m_sessions_condition;
  std::map<lldb::IOObjectSP, DAP *> m_active_sessions;

  /// Optional map from target index to shared debugger set when the native
  /// process spawns a new GPU target
  std::map<uint32_t, lldb::SBDebugger> m_target_to_debugger_map;

  /// Map from debugger ID to its event thread used for when
  /// multiple DAP sessions are using the same debugger instance.
  std::map<lldb::user_id_t, std::weak_ptr<ManagedEventThread>> m_debugger_event_threads;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_DAPSESSIONMANAGER_H
