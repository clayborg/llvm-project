//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadNVGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessNVGPU.h"

using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_server;

/// Global thread ID counter. This is used to assign a unique thread ID to each
/// thread created by the GPU. This is not related to logical or hardware
/// coordinates.
static lldb::tid_t g_thread_id = 1;

ThreadNVGPU::ThreadNVGPU(ProcessNVGPU &gpu, const ThreadState *thread_state)
    : ThreadNVGPU(gpu, thread_state, g_thread_id++) {}

ThreadNVGPU::ThreadNVGPU(ProcessNVGPU &gpu, const ThreadState *thread_state,
                         lldb::tid_t tid)
    : NativeThreadProtocol(gpu, tid), m_state(lldb::eStateInvalid),
      m_stop_info(), m_reg_context(*this), m_thread_state(thread_state) {}

std::string ThreadNVGPU::GetName() {
  if (!m_thread_state)
    return "Invalid thread";

  const CuDim3 &thread_idx = m_thread_state->GetThreadIdx();
  const CuDim3 &block_idx = m_thread_state->GetWarpState().GetBlockIdx();
  return llvm::formatv("blockIdx(x={} y={} z={}) threadIdx(x={} y={} z={})",
                       block_idx.x, block_idx.y, block_idx.z,
                       thread_idx.x, thread_idx.y, thread_idx.z);
}

lldb::StateType ThreadNVGPU::GetState() { return m_state; }

bool ThreadNVGPU::GetStopReason(ThreadStopInfo &stop_info,
                                std::string &description) {
  stop_info = m_stop_info;
  description = m_stop_description;
  return true;
}

lldb::StopReason ThreadNVGPU::GetStopReason() const {
  return m_stop_info.reason;
}

ProcessNVGPU &ThreadNVGPU::GetGPU() { return static_cast<ProcessNVGPU &>(m_process); }

const ProcessNVGPU &ThreadNVGPU::GetGPU() const {
  return static_cast<const ProcessNVGPU &>(m_process);
}

void ThreadNVGPU::SetStoppedBySignal(int signo) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVGPU::SetStoppedBySignal()");
  SetStopped(lldb::eStopReasonSignal, /*description=*/std::nullopt, signo);
}

void ThreadNVGPU::SetStoppedByDynamicLoader() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVGPU::SetStoppedByDynamicLoader()");
  SetStopped(lldb::eStopReasonDynamicLoader, "NVIDIA GPU Thread Stopped by Dynamic Loader");
}

void ThreadNVGPU::SetStoppedByException(const ExceptionInfo &exception_info) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVGPU::SetStoppedByException()");
  SetStopped(lldb::eStopReasonException,
             llvm::formatv("CUDA Exception({}): {}", exception_info.exception,
                           exception_info.ToString())
                 .str());
}

void ThreadNVGPU::SetStoppedByInitialization() {
  LLDB_LOG(GetLog(GDBRLog::Plugin),
           "ThreadNVGPU::SetStoppedByInitialization()");
  SetStopped(lldb::eStopReasonDynamicLoader, "NVIDIA GPU is initializing");
}

void ThreadNVGPU::SetStoppedByBreakpoint() {
  SetStopped(lldb::eStopReasonBreakpoint);
}

void ThreadNVGPU::SetStopped(lldb::StopReason reason,
                             std::optional<llvm::StringRef> description,
                             uint32_t signo) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVGPU::SetStopped()");

  m_state = lldb::eStateStopped;
  if (description)
    m_stop_description = *description;
  else
    m_stop_description.clear();

  m_stop_info.reason = reason;
  m_stop_info.signo = signo;
}

void ThreadNVGPU::SetRunning() {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "ThreadNVGPU::Resume()");
  m_state = lldb::eStateRunning;
}
