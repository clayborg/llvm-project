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
#include <algorithm>
#include <amd-dbgapi/amd-dbgapi.h>
#include <memory>
#include <string>
#include <vector>

namespace lldb_private {
namespace lldb_server {
static const char *AmdDbgApiEventKindToString(amd_dbgapi_event_kind_t kind) {
  switch (kind) {
  case AMD_DBGAPI_EVENT_KIND_NONE:
    return "NONE";

  case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
    return "WAVE_STOP";

  case AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED:
    return "WAVE_COMMAND_TERMINATED";

  case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
    return "CODE_OBJECT_LIST_UPDATED";

  case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
    return "BREAKPOINT_RESUME";

  case AMD_DBGAPI_EVENT_KIND_RUNTIME:
    return "RUNTIME";

  case AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR:
    return "QUEUE_ERROR";
  }
  assert(false && "unhandled amd_dbgapi_event_kind_t value");
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

} // namespace lldb_server
} // namespace lldb_private
#endif
