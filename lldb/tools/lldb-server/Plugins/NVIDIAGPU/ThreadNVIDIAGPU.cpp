//===----------------------------------------------------------------------===//
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
                                 const ThreadState *thread_state)
    : NativeThreadProtocol(gpu, tid), m_state(lldb::eStateInvalid),
      m_stop_info(), m_reg_context(*this), m_thread_state(thread_state) {}

std::string ThreadNVIDIAGPU::GetName() {
  if (!m_thread_state)
    return "Invalid thread";

  const CuDim3 &thread_idx = m_thread_state->GetThreadIdx();
  return llvm::formatv("threadIdx(x={} y={} z={})", thread_idx.x, thread_idx.y,
                       thread_idx.z);
}

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
  SetStopped(lldb::eStopReasonSignal, /*description=*/std::nullopt, signo);
}

void ThreadNVIDIAGPU::SetStoppedByDynamicLoader() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStoppedByDynamicLoader()");
  SetStopped(lldb::eStopReasonDynamicLoader, "NVIDIA GPU Thread Stopped by Dynamic Loader");
}

void ThreadNVIDIAGPU::SetStoppedByException(
    const ExceptionInfo &exception_info) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStoppedByException()");
  SetStopped(lldb::eStopReasonException,
             llvm::formatv("CUDA Exception({}): {}", exception_info.exception,
                           exception_info.ToString())
                 .str());
}

void ThreadNVIDIAGPU::SetStoppedByInitialization() {
  LLDB_LOG(GetLog(GDBRLog::Plugin),
           "ThreadNVIDIAGPU::SetStoppedByInitialization()");
  SetStopped(lldb::eStopReasonDynamicLoader, "NVIDIA GPU is initializing");
}

void ThreadNVIDIAGPU::SetStopped(lldb::StopReason reason,
                                 std::optional<llvm::StringRef> description,
                                 uint32_t signo) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::SetStopped()");

  m_state = lldb::eStateStopped;
  if (description)
    m_stop_description = *description;
  else
    m_stop_description.clear();

  m_stop_info.reason = reason;
  m_stop_info.signo = signo;
}

void ThreadNVIDIAGPU::SetRunning() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVIDIAGPU::Resume()");
  m_state = lldb::eStateRunning;
}
