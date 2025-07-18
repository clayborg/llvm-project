//===-- Exceptions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Exceptions.h"
#include "Coords.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "Utils.h"

using namespace lldb_private::process_gdb_remote;

namespace lldb_private::lldb_server {
std::optional<ExceptionInfo> FindExceptionInfo(const CUDBGAPI_st &api) {
  // Find the thread that caused the exception.
  Log *log = GetLog(GDBRLog::Plugin);
  PhysicalCoords physical_coords;
  const uint32_t dev_id = 0;
  uint32_t num_sms;
  CUDBGResult res = api.getNumSMs(dev_id, &num_sms);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::FindExceptionInfo(). Failed to get number of SMs: {0}",
        cudbgGetErrorString(res));
  }

  std::vector<uint64_t> sm_exceptions(num_sms / 64 + 1, 0);
  res = api.readDeviceExceptionState(dev_id, sm_exceptions.data(),
                                     sm_exceptions.size());
  if (res != CUDBG_SUCCESS) {
    LLDB_LOG(log, "NVIDIAGPU::FindExceptionInfo(). No exception found.");
    return std::nullopt;
  }

  // Find the first SM with an exception
  uint32_t sm_id = UINT32_MAX;
  for (uint32_t i = 0; i < num_sms; ++i) {
    if (sm_exceptions[i / 64] & (1ULL << (i % 64))) {
      sm_id = i;
      break;
    }
  }
  if (sm_id == UINT32_MAX) {
    logAndReportFatalError(
        "NVIDIAGPU::FindExceptionInfo(). No SMs with exceptions found");
  }

  // Find the first warp with an exception
  uint32_t num_warps;
  res = api.getNumWarps(dev_id, &num_warps);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("NVIDIAGPU::FindExceptionInfo(). Failed to get "
                           "number of warps: {0}",
                           cudbgGetErrorString(res));
  }

  uint64_t valid_warps_mask;
  res = api.readValidWarps(dev_id, sm_id, &valid_warps_mask);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::FindExceptionInfo(). Failed to read valid warps: {0}",
        cudbgGetErrorString(res));
  }

  for (uint32_t wp = 0; wp < num_warps; ++wp) {
    if (!(valid_warps_mask & (1ULL << wp)))
      continue;

    CUDBGWarpState warp;
    res = api.readWarpState(dev_id, sm_id, wp, &warp);
    if (res != CUDBG_SUCCESS) {
      logAndReportFatalError(
          "NVIDIAGPU::FindExceptionInfo(). Failed to read warp state: {0}",
          cudbgGetErrorString(res));
    }

    if (!warp.validLanes)
      continue;

    for (uint32_t ln = 0; ln < 32; ++ln) {
      if (warp.validLanes & (1 << ln)) {
        CUDBGException_t exception = CUDBGException_t::CUDBG_EXCEPTION_NONE;
        res = api.readLaneException(dev_id, sm_id, wp, ln, &exception);
        if (res != CUDBG_SUCCESS)
          logAndReportFatalError(
              "NVIDIAGPU::FindExceptionInfo(). Failed to read lane "
              "exception: {0}",
              cudbgGetErrorString(res));

        if (exception != CUDBGException_t::CUDBG_EXCEPTION_NONE) {
          physical_coords = PhysicalCoords(dev_id, sm_id, wp, ln);
          LLDB_LOG(log,
                   "Exception {0} found at dev_id: {1}, sm_id: {2}, wp: "
                   "{3}, ln: {4}",
                   exception, physical_coords.dev_id, physical_coords.sm_id,
                   physical_coords.warp_id, physical_coords.lane_id);
          return ExceptionInfo{physical_coords, exception};
        }
      }
    }
  }
  logAndReportFatalError("NVIDIAGPU::FindExceptionInfo(). Couldn't find "
                         "concrete exception info.");
}

} // namespace lldb_private::lldb_server
