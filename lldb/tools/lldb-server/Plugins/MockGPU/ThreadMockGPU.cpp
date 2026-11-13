//===-- ThreadMockGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadMockGPU.h"
#include "ProcessMockGPU.h"

using namespace lldb_private;
using namespace lldb_server;

ThreadMockGPU::ThreadMockGPU(ProcessMockGPU &process, lldb::tid_t tid,
                             lldb::tid_t lane_id,
                             std::optional<lldb::tid_t> simd_id,
                             std::optional<lldb::tid_t> simt_id)
    : NativeThreadProtocol(process, tid, lane_id, simd_id, simt_id),
      m_reg_context(*this) {
  m_stop_info.reason = lldb::eStopReasonTrace;
}

// NativeThreadProtocol Interface
std::string ThreadMockGPU::GetName() { return "Mock GPU Thread Name"; }

lldb::StateType ThreadMockGPU::GetState() { return lldb::eStateStopped; }

bool ThreadMockGPU::GetStopReason(ThreadStopInfo &stop_info,
                                  std::string &description) {
  stop_info = m_stop_info;
  description = "Mock GPU Thread Stop Reason";
  return true;
}

std::optional<llvm::json::Value>
ThreadMockGPU::GetExtendedInfo(const llvm::json::Value &args) {
  llvm::json::Object info{
      {"tid", static_cast<int64_t>(GetID())},
      {"name", GetName()},
      {"is_active", GetIsActive()},
      {"dispatch",
       llvm::json::Object{
           {"kernel_name", "mock_gpu_kernel"},
           {"grid_size", llvm::json::Array{16, 8, 1}},
           {"work_group_size", llvm::json::Array{4, 4, 1}},
       }},
      // Echo the request back so clients can verify the arguments made it to
      // the thread intact.
      {"request", args},
  };
  if (m_lane_id)
    info["lane_id"] = static_cast<int64_t>(*m_lane_id);
  if (m_simd_id)
    info["simd_id"] = static_cast<int64_t>(*m_simd_id);
  if (m_simt_id)
    info["simt_id"] = static_cast<int64_t>(*m_simt_id);
  return llvm::json::Value(std::move(info));
}

Status ThreadMockGPU::SetWatchpoint(lldb::addr_t addr, size_t size,
                                    uint32_t watch_flags, bool hardware) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::RemoveWatchpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

ProcessMockGPU &ThreadMockGPU::GetProcess() {
  return static_cast<ProcessMockGPU &>(m_process);
}

const ProcessMockGPU &ThreadMockGPU::GetProcess() const {
  return static_cast<const ProcessMockGPU &>(m_process);
}
