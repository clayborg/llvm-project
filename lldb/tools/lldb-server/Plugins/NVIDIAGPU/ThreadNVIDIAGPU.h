//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H
#define LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H

#include "RegisterContextNVIDIAGPU.h"
#include "forward-declarations.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/lldb-private-forward.h"
#include <string>

namespace lldb_private::lldb_server {
/// This class represents a HW thread in a GPU.
class ThreadNVIDIAGPU : public NativeThreadProtocol {
public:
  ThreadNVIDIAGPU(NVIDIAGPU &gpu, const ThreadState *thread_state);

  ThreadNVIDIAGPU(NVIDIAGPU &gpu, const ThreadState *thread_state,
                  lldb::tid_t tid);

  std::string GetName() override;

  lldb::StateType GetState() override;

  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) override;

  RegisterContextNVIDIAGPU &GetRegisterContext() override {
    return m_reg_context;
  }

  Status SetWatchpoint(lldb::addr_t addr, size_t size, uint32_t watch_flags,
                       bool hardware) override {
    return Status::FromErrorString("unimplemented");
  }

  Status RemoveWatchpoint(lldb::addr_t addr) override {
    return Status::FromErrorString("unimplemented");
  }

  Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override {
    return Status::FromErrorString("unimplemented");
  }

  Status RemoveHardwareBreakpoint(lldb::addr_t addr) override {
    return Status::FromErrorString("unimplemented");
  }

  /// \return the process that this thread belongs to.
  NVIDIAGPU &GetGPU();

  /// \return the process that this thread belongs to.
  const NVIDIAGPU &GetGPU() const;

  /// \return the ThreadState object associated with this thread. It might be
  /// null.
  const ThreadState *GetThreadState() const { return m_thread_state; }

  /// Change the state of this thread and update its stop info accordingly.
  void SetStoppedByDynamicLoader();

  /// Change the state of this thread and update its stop info accordingly.
  void SetStoppedByException(const ExceptionInfo &exception_info);

  /// Set the thread to stopped state by a signal.
  void SetStoppedBySignal(int signo);

  /// Set the thread to stopped state.
  void SetStopped(lldb::StopReason reason = lldb::eStopReasonNone,
                  std::optional<llvm::StringRef> description = std::nullopt,
                  uint32_t signo = 0);

  /// Set the thread to stopped state by initialization.
  void SetStoppedByInitialization();

  /// Set the thread to running state.
  void SetRunning();

private:
  friend class NVIDIAGPU;

  /// Set the physical coordinates and thread index of the thread within its block.
  void SetThreadState(const ThreadState *thread_state) {
    m_thread_state = thread_state;
  }

  /// The current state of the thread.
  lldb::StateType m_state;

  /// The stop info of the thread.
  ThreadStopInfo m_stop_info;

  /// The textual description of the thread's stop reason.
  std::string m_stop_description;

  /// The register context of the thread.
  RegisterContextNVIDIAGPU m_reg_context;

  /// The raw state of this thread. It is null during initialization, as there
  /// are no threads yet.
  const ThreadState *m_thread_state = nullptr;
};
} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H
