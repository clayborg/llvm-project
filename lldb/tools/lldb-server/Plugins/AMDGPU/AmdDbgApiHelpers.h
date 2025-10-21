//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers for working with the amd-dbgapi.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_AMDDBGAPIHELPERS_H
#define LLDB_TOOLS_LLDB_SERVER_AMDDBGAPIHELPERS_H
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <amd-dbgapi/amd-dbgapi.h>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lldb_private {
namespace lldb_server {

#define ENUM_TO_CSTR(e)                                                        \
  case e:                                                                      \
    return #e

static const char *AmdDbgApiEventKindToString(amd_dbgapi_event_kind_t kind) {
  switch (kind) {
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_NONE);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_RUNTIME);
    ENUM_TO_CSTR(AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR);
  }
  assert(false && "unhandled amd_dbgapi_event_kind_t value");
}

inline const char *
AmdDbgApiWaveInfoKindToString(amd_dbgapi_wave_info_t info_kind) {
  switch (info_kind) {
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_STATE);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_STOP_REASON);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_WATCHPOINTS);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_WORKGROUP);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_DISPATCH);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_QUEUE);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_AGENT);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_PROCESS);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_ARCHITECTURE);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_PC);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_EXEC_MASK);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP);
    ENUM_TO_CSTR(AMD_DBGAPI_WAVE_INFO_LANE_COUNT);
  }
  assert(false && "unhandled amd_dbgapi_wave_info_t value");
}

inline const char *
AmdDbgApiDispatchInfoKindToString(amd_dbgapi_dispatch_info_t info_kind) {
  switch (info_kind) {
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_QUEUE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_AGENT);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_PROCESS);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_ARCHITECTURE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_BARRIER);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_ACQUIRE_FENCE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_RELEASE_FENCE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS);
    ENUM_TO_CSTR(AMD_DBGAPI_DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS);
  }
  assert(false && "unhandled amd_dbgapi_dispatch_info_t value");
}

inline const char *AmdDbgApiStatusToString(amd_dbgapi_status_t status) {
  switch (status) {
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_SUCCESS);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_FATAL);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ALREADY_INITIALIZED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_RESTRICTION);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ALREADY_ATTACHED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_OUTSTANDING_STOP);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_RESUMABLE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
    ENUM_TO_CSTR(
        AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_RESUME_DISPLACED_STEPPING);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_LANE_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_CLASS_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_CLIENT_PROCESS_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_SYMBOL_NOT_FOUND);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INVALID_WORKGROUP_ID);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_INCOMPATIBLE_PROCESS_STATE);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_ALREADY_FROZEN);
    ENUM_TO_CSTR(AMD_DBGAPI_STATUS_ERROR_PROCESS_NOT_FROZEN);
  }
  assert(false && "unhandled amd_dbgapi_status_t value");
}

struct AmdDbgApiEventSet {
  AmdDbgApiEventSet() = default;

  void SetLastEvent(amd_dbgapi_event_id_t event_id,
                    amd_dbgapi_event_kind_t event_kind) {
    m_last_event_id = event_id;
    m_events.push_back(event_kind);
  }

  amd_dbgapi_event_id_t GetLastEventID() const { return m_last_event_id; }

  bool HasEvent(amd_dbgapi_event_kind_t event_kind) const {
    return std::find(m_events.begin(), m_events.end(), event_kind) !=
           m_events.end();
  }

  bool HasWaveStopEvent() const {
    return HasEvent(AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
  }

  bool HasBreakpointResumeEvent() const {
    return HasEvent(AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);
  }

  std::string ToString() const {
    std::string s("[");
    for (size_t i = 0; i < m_events.size(); ++i) {
      if (i != 0)
        s += ", ";
      s += AmdDbgApiEventKindToString(m_events[i]);
    }
    return s + "]";
  }

private:
  // Keep track of events we have seen.
  // We use a vector to aid in debugging so that we can track both the
  // order of events and allow duplicates.
  std::vector<amd_dbgapi_event_kind_t> m_events;
  amd_dbgapi_event_id_t m_last_event_id = AMD_DBGAPI_EVENT_NONE;
};

// Custom deleter for std::unique_ptr that uses FreeDbgApiClientMemory
struct DbgApiClientMemoryDeleter {
  void operator()(void *ptr) const;
};

// Type alias for std::unique_ptr with the custom deleter
//
// Example usage:
// auto ptr =
// DbgApiClientMemoryPtr<SomeType>(static_cast<SomeType*>(raw_ptr_from_dbgapi));
// The memory will be automatically freed when ptr goes out of scope
template <typename T>
using DbgApiClientMemoryPtr = std::unique_ptr<T, DbgApiClientMemoryDeleter>;

// Custom hash function for amd_dbgapi_wave_id_t
struct WaveIdHash {
  std::size_t operator()(const amd_dbgapi_wave_id_t &wave_id) const noexcept {
    return std::hash<uint64_t>{}(wave_id.handle);
  }
};

// Run a command from the amd-dbgapi library and return an llvm::Error if not
// successful.
inline llvm::Error
RunAmdDbgApiCommand(std::function<amd_dbgapi_status_t()> func) {
  amd_dbgapi_status_t status = func();
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AMD_DBGAPI_STATUS_ERROR: %s",
                                   AmdDbgApiStatusToString(status));
  return llvm::Error::success();
}

// Convenient typedef for unordered_set of wave IDs with custom hasher.
using WaveIdSet = std::unordered_set<amd_dbgapi_wave_id_t, WaveIdHash>;

// Convenient typedef for unordered_map of wave IDs with custom hasher.
template <typename T>
using WaveIdMap = std::unordered_map<amd_dbgapi_wave_id_t, T, WaveIdHash>;

// Convenient typedef for vector of wave IDs.
using WaveIdList = std::vector<amd_dbgapi_wave_id_t>;

} // namespace lldb_server
} // namespace lldb_private

// Comparison operator for amd_dbgapi_wave_id_t to enable sorting
inline bool operator<(const amd_dbgapi_wave_id_t &lhs,
                      const amd_dbgapi_wave_id_t &rhs) {
  return lhs.handle < rhs.handle;
}
inline bool operator==(const amd_dbgapi_wave_id_t &lhs,
                       const amd_dbgapi_wave_id_t &rhs) {
  return lhs.handle == rhs.handle;
}
#endif
