//===-- ThreadNVIDIAGPU.h ----------------------------------- -*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H
#define LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H

#include "RegisterContextNVIDIAGPU.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/lldb-private-forward.h"
#include <string>

namespace lldb_private::lldb_server {
class NVIDIAGPU;

class NativeProcessLinux;

/// This class represents a HW thread in a GPU.
class ThreadNVIDIAGPU : public NativeThreadProtocol {
public:
  // This struct represents the physical coordinates of a HW thread in a GPU.
  struct PhysicalCoords {
    int64_t dev_id = -1;
    int64_t sm_id = -1;
    int64_t warp_id = -1;
    int64_t lane_id = -1;

    bool IsValid() const;

    std::string AsThreadName() const;
  };

  ThreadNVIDIAGPU(NVIDIAGPU &gpu, lldb::tid_t tid,
                  PhysicalCoords physical_coords);

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

  /// \return the physical coordinates of this thread.
  PhysicalCoords GetPhysicalCoords() const { return m_physical_coords; }

  /// Change the state of this thread and update its stop info accordingly.
  void SetStoppedByDynamicLoader();

  /// Change the state of this thread and update its stop info accordingly.
  void SetStoppedByException();

  /// Set the thread to stopped state by a signal.
  void SetStoppedBySignal(int signo);

  /// Set the thread to stopped state.
  void SetStopped();

  /// Set the thread to stopped state by a threadless state.
  void SetStoppedByThreadlessState();

  /// Set the thread to running state.
  void SetRunning();

private:
  friend class NVIDIAGPU;

  /// Set the physical coordinates of this thread.
  ///
  /// \param physical_coords the physical coordinates of this thread.
  void SetPhysicalCoords(const PhysicalCoords &physical_coords) {
    m_physical_coords = physical_coords;
  }

  lldb::StateType m_state;
  ThreadStopInfo m_stop_info;
  RegisterContextNVIDIAGPU m_reg_context;
  std::string m_stop_description;
  PhysicalCoords m_physical_coords;
};
} // namespace lldb_private::lldb_server

#endif // #ifndef LLDB_TOOLS_LLDB_SERVER_THREADNVIDIAGPU_H
