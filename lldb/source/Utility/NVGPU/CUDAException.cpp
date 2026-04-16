//===-- CUDAException.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/CUDAException.h"
#include "llvm/Support/FormatVariadic.h"

#include <cassert>

using namespace lldb_private;

llvm::StringRef lldb_private::CUDAExceptionToString(CUDBGException_t exception) {
  switch (exception) {
  case CUDBG_EXCEPTION_NONE:
    return "No exception";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
    return "Warp Illegal Instruction";
  case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
    return "Warp Out-of-range Address";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
    return "Warp Misaligned Address";
  case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
    return "Warp Invalid Address Space";
  case CUDBG_EXCEPTION_WARP_INVALID_PC:
    return "Warp Invalid PC";
  case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
    return "Warp Hardware Stack Overflow";
  case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
    return "Device Illegal Address";
  case CUDBG_EXCEPTION_WARP_ASSERT:
    return "Warp Assert";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS:
    return "Warp Illegal Address";
  case CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT:
    return "Cluster Block Not Present";
  case CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS:
    return "Cluster Out-of-range Address";
  case CUDBG_EXCEPTION_WARP_STACK_CANARY:
    return "Warp Stack Canary";
  case CUDBG_EXCEPTION_WARP_TMEM_ACCESS_CHECK:
    return "Warp Tensor Memory Access Check";
  case CUDBG_EXCEPTION_WARP_TMEM_LEAK:
    return "Warp Tensor Memory Leak";
  case CUDBG_EXCEPTION_WARP_CALL_REQUIRES_NEWER_DRIVER:
    return "Warp Call Requires Newer Driver";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_PC:
    return "Warp Misaligned PC";
  case CUDBG_EXCEPTION_WARP_PC_OVERFLOW:
    return "Warp PC Overflow";
  case CUDBG_EXCEPTION_WARP_MISALIGNED_REG:
    return "Warp Misaligned Register";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_ENCODING:
    return "Warp Illegal Instruction Encoding";
  case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_PARAM:
    return "Warp Illegal Instruction Parameter";
  case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_REGISTER:
    return "Warp Out-of-range Register";
  case CUDBG_EXCEPTION_WARP_INVALID_CONST_ADDR_LDC:
    return "Warp Invalid Constant Address";
  case CUDBG_EXCEPTION_WARP_MMU_FAULT:
    return "Warp MMU Fault";
  case CUDBG_EXCEPTION_WARP_ARRIVE:
    return "Warp Barrier Arrival Mismatch";
  case CUDBG_EXCEPTION_CLUSTER_POISON:
    return "Cluster Shared Memory Issue";
  case CUDBG_EXCEPTION_WARP_API_STACK_ERROR:
    return "Warp API Stack Error";
  case CUDBG_EXCEPTION_WARP_BLOCK_NOT_PRESENT:
    return "Warp Block Not Present";
  case CUDBG_EXCEPTION_WARP_USER_STACK_OVERFLOW:
    return "Warp User Stack Overflow";
  default:
    return "Device Unknown Exception";
  }
}

CUDAExceptionInfo::CUDAExceptionInfo(CUDBGException_t exception,
                                     std::optional<uint64_t> errorPC)
    : exception(exception), errorPC(errorPC) {
  assert(exception != CUDBG_EXCEPTION_NONE &&
         "CUDAExceptionInfo: exception should not be CUDBG_EXCEPTION_NONE");
}

std::string CUDAExceptionInfo::ToString() const {
  std::string result = CUDAExceptionToString(exception).str();
  if (errorPC)
    result += llvm::formatv(" at {0:x}", *errorPC);
  return result;
}
