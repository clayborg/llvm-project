//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the WaveAMDGPU class, which is used
/// to represent a wave on the GPU.
///
//===----------------------------------------------------------------------===//

#include "WaveAMDGPU.h"
#include "AmdDbgApiHelpers.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ThreadAMDGPU.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/MathExtras.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <cstddef>
#include <cstdint>
#include <memory>

using namespace lldb_private;
using namespace lldb_server;
using namespace lldb_private::process_gdb_remote;

template <typename T>
static llvm::Error QueryDispatchInfo(amd_dbgapi_dispatch_id_t dispatch_id,
                                     amd_dbgapi_dispatch_info_t info_type,
                                     T *dest) {
  amd_dbgapi_status_t status =
      amd_dbgapi_dispatch_get_info(dispatch_id, info_type, sizeof(*dest), dest);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to get %s for dispatch %" PRIu64 ": status=%s",
        AmdDbgApiDispatchInfoKindToString(info_type), dispatch_id.handle,
        AmdDbgApiStatusToString(status));
  }
  return llvm::Error::success();
}

static llvm::Expected<size_t>
ComputeNumLanesInWave(const DbgApiWaveInfo &wave_info) {
  uint16_t workgroup_sizes[3];
  if (llvm::Error err = QueryDispatchInfo(
          wave_info.dispatch_id, AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
          &workgroup_sizes))
    return err;

  // Compute the total number of threads and waves.
  const size_t wave_size = wave_info.num_lanes_supported;
  const size_t total_num_lanes =
      workgroup_sizes[0] * workgroup_sizes[1] * workgroup_sizes[2];
  const size_t num_waves =
      std::max(llvm::divideCeil(total_num_lanes, wave_size), size_t(1));

  // Most waves have a full wave_size worth of lanes.
  size_t num_lanes = wave_size;

  // If this is the last wave, it may not have a full wave_size worth of lanes.
  const bool is_last_wave = wave_info.index_in_workgroup == num_waves - 1;
  if (is_last_wave) {
    const size_t num_lanes_with_full_waves = (num_waves - 1) * wave_size;
    num_lanes = total_num_lanes - num_lanes_with_full_waves;
  }

  return num_lanes;
}

// This is used to have a unique thread ID for each thread.
// Returns the base thread ID for the wave.
static lldb::tid_t ReserveTidsForWave(size_t num_lanes) {
  // Static counter for thread ids.
  // Skip over the id that is reserved for the shadow thread.
  // This is not a thread-safe variable, but that should be ok since we
  // only have a lldb-server single thread in the main loop.
  static lldb::tid_t s_tid = ThreadAMDGPU::AMDGPU_SHADOW_THREAD_ID + 1;

  lldb::tid_t tid_base = s_tid;
  s_tid += num_lanes;
  return tid_base;
}

void WaveAMDGPU::AddThreadsToList(
    ProcessAMDGPU &process,
    std::vector<std::unique_ptr<NativeThreadProtocol>> &threads) {
  Log *log = GetLog(GDBRLog::Plugin);

  // Reserve a contiguous range of thread IDs for this wave.
  llvm::Expected<size_t> num_lanes = ComputeNumLanesInWave(m_wave_info);
  if (!num_lanes) {
    LLDB_LOG_ERROR(log, num_lanes.takeError(),
                   "Failed to compute number of lanes for wave {1}. {0}.",
                   m_wave_id.handle);
    return;
  }
  lldb::tid_t tid_base = ReserveTidsForWave(*num_lanes);

  // Create a thread for each lane in the wave.
  LLDB_LOG(log, "Creating {} threads for wave {}", *num_lanes,
           m_wave_id.handle);
  for (size_t i = 0; i < *num_lanes; ++i) {
    lldb::tid_t tid = tid_base + i;
    threads.push_back(
        std::make_unique<ThreadAMDGPU>(process, tid, shared_from_this(), i));
  }
}

void WaveAMDGPU::SetDbgApiInfo(const DbgApiWaveInfo &wave_info) {
  m_wave_info = wave_info;
  UpdateStopReasonFromWaveInfo();
}

static lldb::StopReason
GetLldbStopReasonForDbgApiStopReason(amd_dbgapi_wave_stop_reasons_t reason) {
  // If none of the bits are set, then we explicitly stopped the wave with
  // a call to `amd_dbgapi_wave_stop`.
  if (reason == AMD_DBGAPI_WAVE_STOP_REASON_NONE)
    return lldb::StopReason::eStopReasonInterrupt;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT)
    return lldb::StopReason::eStopReasonBreakpoint;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT)
    return lldb::StopReason::eStopReasonWatchpoint;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP)
    return lldb::StopReason::eStopReasonTrace;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP)
    return lldb::StopReason::eStopReasonBreakpoint;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_TRAP)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR)
    return lldb::StopReason::eStopReasonException;

  if (reason & AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT)
    return lldb::StopReason::eStopReasonException;

  return lldb::StopReason::eStopReasonInvalid;
}

void WaveAMDGPU::UpdateStopReasonFromWaveInfo() {
  lldb::StopReason reason = lldb::StopReason::eStopReasonInvalid;
  switch (m_wave_info.state) {
  case AMD_DBGAPI_WAVE_STATE_RUN:
  case AMD_DBGAPI_WAVE_STATE_SINGLE_STEP:
    reason = lldb::StopReason::eStopReasonNone;
    break;
  case AMD_DBGAPI_WAVE_STATE_STOP:
    reason = GetLldbStopReasonForDbgApiStopReason(m_wave_info.stop_reason);
    break;
  }

  assert(reason != lldb::StopReason::eStopReasonInvalid);
  SetStopReason(reason);
}
