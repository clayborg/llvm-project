//===-- ThreadNVIDIAGPU.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadNVIDIAGPU.h"
#include "NVIDIAGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"

using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_server;

ThreadNVIDIAGPU::ThreadNVIDIAGPU(NVIDIAGPU &gpu, lldb::tid_t tid,
                                 PhysicalCoords physical_coords)
    : NativeThreadProtocol(gpu, tid), m_state(lldb::eStateInvalid),
      m_stop_info(), m_reg_context(*this), m_physical_coords(physical_coords) {}

std::string ThreadNVIDIAGPU::GetName() { return "NVIDIA GPU Thread"; }

lldb::StateType ThreadNVIDIAGPU::GetState() { return m_state; }

bool ThreadNVIDIAGPU::GetStopReason(ThreadStopInfo &stop_info,
                                    std::string &description) {
  stop_info = m_stop_info;
  description = m_stop_description;
  return true;
}

NVIDIAGPU &ThreadNVIDIAGPU::GetGPU() {
  return static_cast<NVIDIAGPU &>(m_process);
}

const NVIDIAGPU &ThreadNVIDIAGPU::GetGPU() const {
  return static_cast<const NVIDIAGPU &>(m_process);
}

void ThreadNVIDIAGPU::SetStoppedBySignal(int signo) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStoppedBySignal()");
  SetStopped();
  m_stop_info.reason = lldb::eStopReasonSignal;
  m_stop_info.signo = signo;
}

void ThreadNVIDIAGPU::SetStoppedByDynamicLoader() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStoppedByException()");
  SetStopped();

  m_stop_info.reason = lldb::eStopReasonTrace;
  m_stop_description = "NVIDIA GPU Thread Stopped by Dynamic Loader";
}

void ThreadNVIDIAGPU::SetStoppedByException() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStoppedByException()");
  SetStopped();

  m_stop_info.reason = lldb::eStopReasonException;
  m_stop_description = "NVIDIA GPU Thread Stopped by Exception";
}

void ThreadNVIDIAGPU::SetStopped() {
  if (m_state == lldb::eStateStopped)
    return;

  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStopped()");

  // On every stop, clear any cached information.
  GetRegisterContext().InvalidateAllRegisters();

  m_state = lldb::eStateStopped;
  m_stop_description.clear();
  m_stop_info.signo = 0;
}

void ThreadNVIDIAGPU::SetRunning() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::Resume()");
  m_state = lldb::eStateRunning;
}