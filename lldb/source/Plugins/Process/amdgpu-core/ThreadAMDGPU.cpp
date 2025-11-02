//===-- ThreadAMDGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadAMDGPU.h"
#include "lldb/Target/StopInfo.h"

using namespace lldb_private;

void ThreadAMDGPU::RefreshStateAfterStop() {
  GetRegisterContext()->InvalidateIfNeeded(false);
}

lldb::RegisterContextSP ThreadAMDGPU::GetRegisterContext() {
  if (!m_reg_context_sp) {
    m_reg_context_sp = std::make_shared<RegisterContextAmdGpu>(*this);
  }
  return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadAMDGPU::CreateRegisterContextForFrame(StackFrame *frame) {
  // TODO: Implement this
  return GetRegisterContext();
}

// NativeThreadProtocol Interface
const char *ThreadAMDGPU::GetName() {
  if (!m_thread_name.empty())
    return m_thread_name.c_str();

  if (!m_wave_id)
    return "AMD Native Shadow Thread";
  
  return "AMD GPU Thread";
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
ThreadAMDGPU::GetSiginfo(size_t max_size) const {
  // TODO: how to implement this?
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no siginfo note");
}

bool ThreadAMDGPU::CalculateStopInfo() {
  lldb::ProcessSP process_sp(GetProcess());
  if (!process_sp)
    return false;

  // TODO: how to implement this?
  SetStopInfo(StopInfo::CreateStopReasonWithSignal(*this, 5));
  return true;
}
