//===-- ThreadIntelGT.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_THREADINTELGT_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_THREADINTELGT_H

#include "EUThreadIntelGT.h"
#include "RegisterContextIntelGT.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/lldb-enumerations.h"

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <memory>
#include <string>

namespace lldb_private {
namespace lldb_server {

class ProcessIntelGT;

/// One NativeThreadProtocol per SIMD lane within an EU hardware thread;
/// siblings share the EUThreadIntelGT parent, each lane keeps its own stop
/// reason, and the shadow thread (TID=1) has no EU parent.
class ThreadIntelGT : public NativeThreadProtocol {
  friend class ProcessIntelGT;

public:
  /// Construct a lane thread from an EU thread parent.
  ThreadIntelGT(ProcessIntelGT &process, lldb::tid_t tid,
                std::shared_ptr<EUThreadIntelGT> eu_thread, uint32_t lane_id);

  /// Factory: create the permanent shadow thread (TID = 1).
  static std::unique_ptr<ThreadIntelGT>
  CreateShadowThread(ProcessIntelGT &process);

  // ----- NativeThreadProtocol interface ------------------------------------

  std::string GetName() override;
  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  RegisterContextIntelGT &GetRegisterContext() override {
    return m_reg_context;
  }

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override {
    return Status::FromErrorString("Watchpoints not supported on Intel GPU");
  }

  Status RemoveWatchpoint(lldb::addr_t addr) override {
    return Status::FromErrorString("Watchpoints not supported on Intel GPU");
  }

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override {
    return Status::FromErrorString(
        "Hardware breakpoints not supported on Intel GPU");
  }

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override {
    return Status::FromErrorString(
        "Hardware breakpoints not supported on Intel GPU");
  }

  // ----- IntelGT-specific helpers ------------------------------------------

  bool IsShadowThread() const { return GetID() == INTELGT_SHADOW_THREAD_ID; }

  ze_device_thread_t GetZeThread() const;
  uint32_t GetDeviceIndex() const;
  uint32_t GetLaneID() const { return m_lane_id; }
  EUThreadIntelGT *GetEUThread() const { return m_eu_thread.get(); }

  void SetState(lldb::StateType state) { m_state = state; }

  /// Set the per-lane stop reason.  Does NOT propagate to the EU thread.
  void SetStopReason(lldb::StopReason reason, uint32_t signo = 0);
  void SetStopDescription(std::string desc) {
    m_stop_description = std::move(desc);
  }

  ProcessIntelGT &GetProcess();

  static constexpr lldb::tid_t INTELGT_SHADOW_THREAD_ID = 1;

private:
  /// Private constructor for the shadow thread (no EU thread, no lane).
  ThreadIntelGT(ProcessIntelGT &process);

  lldb::StateType m_state = lldb::eStateStopped;
  std::shared_ptr<EUThreadIntelGT> m_eu_thread; ///< null for shadow thread.
  uint32_t m_lane_id = 0;
  RegisterContextIntelGT m_reg_context;

  /// Per-lane stop reason; siblings share it so the user can select and step
  /// them.
  ThreadStopInfo m_stop_info{};
  std::string m_stop_description;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_THREADINTELGT_H
