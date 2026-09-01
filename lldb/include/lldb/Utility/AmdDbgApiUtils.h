//===-- AmdDbgApiUtils.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_AMDDBGAPIUTILS_H
#define LLDB_UTILITY_AMDDBGAPIUTILS_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <cstdlib>
#include <functional>
#include <optional>

namespace lldb_private {

#define AMD_DBGAPI_ENUM_TO_CSTR(e)                                             \
  case e:                                                                      \
    return #e

/// Convert an AMD debug API status code to a human-readable string.
inline const char *AmdDbgApiStatusToString(amd_dbgapi_status_t status) {
  switch (status) {
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_SUCCESS);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_FATAL);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    AMD_DBGAPI_ENUM_TO_CSTR(
        AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ALREADY_INITIALIZED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_RESTRICTION);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ALREADY_ATTACHED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_OUTSTANDING_STOP);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_RESUMABLE);
    AMD_DBGAPI_ENUM_TO_CSTR(
        AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(
        AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_RESUME_DISPLACED_STEPPING);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_LANE_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_CLASS_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS);
    AMD_DBGAPI_ENUM_TO_CSTR(
        AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_CLIENT_PROCESS_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_SYMBOL_NOT_FOUND);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WORKGROUP_ID);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INCOMPATIBLE_PROCESS_STATE);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_ALREADY_FROZEN);
    AMD_DBGAPI_ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_NOT_FROZEN);
  }
  llvm_unreachable("unhandled amd_dbgapi_status_t value");
}

#undef AMD_DBGAPI_ENUM_TO_CSTR

/// Run a command from the amd-dbgapi library and return an llvm::Error if not
/// successful.
inline llvm::Error
RunAmdDbgApiCommand(std::function<amd_dbgapi_status_t()> func) {
  amd_dbgapi_status_t status = func();
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AMD_DBGAPI_STATUS_ERROR: %s",
                                   AmdDbgApiStatusToString(status));
  return llvm::Error::success();
}

/// Query the architecture of the first GPU agent attached to a process.
inline llvm::Expected<amd_dbgapi_architecture_id_t>
QueryAmdGpuArchitectureFromFirstAgent(amd_dbgapi_process_id_t gpu_pid,
                                      void (*deallocate_memory)(void *)) {
  size_t agent_count = 0;
  amd_dbgapi_agent_id_t *agents = nullptr;
  amd_dbgapi_status_t status =
      amd_dbgapi_process_agent_list(gpu_pid, &agent_count, &agents, nullptr);
  auto agents_cleanup = llvm::make_scope_exit([agents, deallocate_memory]() {
    if (agents)
      deallocate_memory(agents);
  });

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError("amd_dbgapi_process_agent_list failed: %s",
                                   AmdDbgApiStatusToString(status));

  if (agent_count == 0)
    return llvm::createStringError("AMD debug API reported no GPU agents");

  amd_dbgapi_architecture_id_t architecture_id;
  status =
      amd_dbgapi_agent_get_info(agents[0], AMD_DBGAPI_AGENT_INFO_ARCHITECTURE,
                                sizeof(architecture_id), &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(
        "amd_dbgapi_agent_get_info(ARCHITECTURE) failed: %s",
        AmdDbgApiStatusToString(status));

  return architecture_id;
}

/// Get the AMD debug API log level from the AMD_DBGAPI_LOG_LEVEL environment
/// variable. Returns the log level if set, or std::nullopt if not set.
/// Valid values: "none", "fatal", "warning", "info", "verbose", "trace".
inline std::optional<amd_dbgapi_log_level_t> GetAmdDbgApiLogLevelFromEnv() {
  const char *env = std::getenv("AMD_DBGAPI_LOG_LEVEL");
  if (!env)
    return std::nullopt;
  llvm::StringRef val(env);
  return llvm::StringSwitch<std::optional<amd_dbgapi_log_level_t>>(val.lower())
      .Case("none", AMD_DBGAPI_LOG_LEVEL_NONE)
      .Case("fatal", AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR)
      .Case("warning", AMD_DBGAPI_LOG_LEVEL_WARNING)
      .Case("info", AMD_DBGAPI_LOG_LEVEL_INFO)
      .Case("verbose", AMD_DBGAPI_LOG_LEVEL_VERBOSE)
      .Case("trace", AMD_DBGAPI_LOG_LEVEL_TRACE)
      .Default(std::nullopt);
}

/// Convert amd_dbgapi_lane_id_t to lldb::tid_t
inline std::optional<lldb::tid_t> AmdDbgApiLaneIdToTid(amd_dbgapi_lane_id_t lane_id) {
  if (lane_id == AMD_DBGAPI_LANE_NONE)
    return std::nullopt;
  return static_cast<lldb::tid_t>(lane_id);
}

/// Convert amd_dbgapi_wave_id_t to lldb::tid_t
inline std::optional<lldb::tid_t> AmdDbgApiWaveIdToTid(amd_dbgapi_wave_id_t wave_id) {
  if (wave_id.handle == AMD_DBGAPI_WAVE_NONE.handle)
    return std::nullopt;
  return static_cast<lldb::tid_t>(wave_id.handle);
}

} // namespace lldb_private

#endif // LLDB_UTILITY_AMDDBGAPIUTILS_H
