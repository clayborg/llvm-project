//===-- AmdGpuStopReason.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps AMD debug API wave stop reasons to LLDB stop reasons and descriptions.
/// Used by both the live GPU debugging (lldb-server AMDGPU plugin) and GPU
/// core file debugging (ProcessAmdGpuCore).
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_AMDGPUSTOPREASON_H
#define LLDB_UTILITY_AMDGPUSTOPREASON_H

#include "lldb/lldb-enumerations.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <string>

namespace lldb_private {

/// Map an AMD debugger API wave stop reason to an LLDB stop reason and
/// optional description string.
///
/// \param[in] reason
///     The AMD debug API wave stop reason bitmask.
///
/// \param[out] description
///     If non-null and the stop reason has a human-readable description,
///     the string is set accordingly (e.g. "Memory access violation").
///
/// \return
///     The corresponding LLDB stop reason enum value.
inline lldb::StopReason
GetLldbStopReasonForDbgApiStopReason(amd_dbgapi_wave_stop_reasons_t reason,
                                     std::string *description) {
  if (description)
    description->clear();

  auto make_exception = [&](const char *stop_description) {
    if (description)
      *description = stop_description;
    return lldb::StopReason::eStopReasonException;
  };

  // If none of the bits are set, then we explicitly stopped the wave with
  // a call to `amd_dbgapi_wave_stop`.
  if (reason == AMD_DBGAPI_WAVE_STOP_REASON_NONE)
    return lldb::StopReason::eStopReasonInterrupt;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL)
    return make_exception("Floating-point input denormal");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0)
    return make_exception("Floating-point divide by zero");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW)
    return make_exception("Floating-point overflow");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW)
    return make_exception("Floating-point underflow");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT)
    return make_exception("Floating-point inexact result");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION)
    return make_exception("Floating-point invalid operation");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0)
    return make_exception("Integer divide by zero");

  // The AMD debug API can report multiple stop bits at once. Prefer
  // exception-class stops over generic breakpoint/stepping reasons so LLDB
  // surfaces the actionable failure when both kinds of bits are present.
  //
  // Check specific error reasons before generic TRAP/ASSERT_TRAP because
  // memory faults and other errors can also set the TRAP bit. If TRAP is
  // checked first, the specific description is lost.
  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION)
    return make_exception("Memory access violation");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR)
    return make_exception("Address error (address out of range)");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION)
    return make_exception("Illegal instruction");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR)
    return make_exception("Unrecoverable ECC error");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT)
    return make_exception("Hardware fatal halt");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP)
    return make_exception("Assert trap");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_TRAP)
    return make_exception("Trap instruction");

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT)
    return lldb::StopReason::eStopReasonBreakpoint;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT)
    return lldb::StopReason::eStopReasonWatchpoint;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP)
    return lldb::StopReason::eStopReasonTrace;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP)
    return lldb::StopReason::eStopReasonBreakpoint;

  return lldb::StopReason::eStopReasonInvalid;
}

} // namespace lldb_private

#endif // LLDB_UTILITY_AMDGPUSTOPREASON_H
