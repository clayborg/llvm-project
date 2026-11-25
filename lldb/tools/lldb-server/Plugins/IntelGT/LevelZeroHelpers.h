//===-- LevelZeroHelpers.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LEVELZEROHELPERS_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LEVELZEROHELPERS_H

#include "lldb/Utility/Log.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace lldb_private {
namespace lldb_server {

// ---------------------------------------------------------------------------
// Hash / map keyed on ze_device_thread_t
// ---------------------------------------------------------------------------

struct ZeThreadHash {
  std::size_t operator()(ze_device_thread_t t) const noexcept {
    // Pack the four 32-bit fields into two 64-bit words and mix.
    uint64_t lo = ((uint64_t)t.slice << 32) | t.subslice;
    uint64_t hi = ((uint64_t)t.eu << 32) | t.thread;
    return std::hash<uint64_t>{}(lo) ^ (std::hash<uint64_t>{}(hi) << 1);
  }
};

struct ZeThreadEqual {
  bool operator()(ze_device_thread_t a, ze_device_thread_t b) const noexcept {
    return a.slice == b.slice && a.subslice == b.subslice && a.eu == b.eu &&
           a.thread == b.thread;
  }
};

template <typename T>
using ZeThreadMap =
    std::unordered_map<ze_device_thread_t, T, ZeThreadHash, ZeThreadEqual>;

// ---------------------------------------------------------------------------
// Equality for ze_device_thread_t (needed by unordered_map)
// ---------------------------------------------------------------------------

inline bool operator==(ze_device_thread_t a, ze_device_thread_t b) {
  return a.slice == b.slice && a.subslice == b.subslice && a.eu == b.eu &&
         a.thread == b.thread;
}

// ---------------------------------------------------------------------------
// Wildcard thread constant
// ---------------------------------------------------------------------------

inline ze_device_thread_t ZeWildcardThread() {
  return {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
}

inline bool ZeThreadIsWildcard(ze_device_thread_t t) {
  return t.slice == UINT32_MAX && t.subslice == UINT32_MAX &&
         t.eu == UINT32_MAX && t.thread == UINT32_MAX;
}

// ---------------------------------------------------------------------------
// Wrap ze_result_t into llvm::Error
// ---------------------------------------------------------------------------

inline llvm::Error RunZeCommand(ze_result_t result, llvm::StringRef msg) {
  if (result == ZE_RESULT_SUCCESS)
    return llvm::Error::success();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "%s: ze_result_t = 0x%x", msg.data(),
                                 static_cast<unsigned>(result));
}

// ---------------------------------------------------------------------------
// Enum-to-string helpers
// ---------------------------------------------------------------------------

#define ZE_ENUM_TO_CSTR(e)                                                     \
  case e:                                                                      \
    return #e

inline llvm::StringRef ZeResultToString(ze_result_t result) {
  switch (result) {
    ZE_ENUM_TO_CSTR(ZE_RESULT_SUCCESS);
    ZE_ENUM_TO_CSTR(ZE_RESULT_NOT_READY);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_DEVICE_LOST);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_NOT_AVAILABLE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_ARGUMENT);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_SIZE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_ENUMERATION);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
    ZE_ENUM_TO_CSTR(ZE_RESULT_ERROR_UNKNOWN);
  default:
    return "ZE_RESULT_UNKNOWN";
  }
}

inline llvm::StringRef ZeEventTypeToString(zet_debug_event_type_t type) {
  switch (type) {
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_INVALID);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_DETACHED);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_MODULE_LOAD);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE);
    ZE_ENUM_TO_CSTR(ZET_DEBUG_EVENT_TYPE_PAGE_FAULT);
  default:
    return "ZET_DEBUG_EVENT_TYPE_UNKNOWN";
  }
}

#undef ZE_ENUM_TO_CSTR

// ---------------------------------------------------------------------------
// ACK helper
// ---------------------------------------------------------------------------
// Every ZE event with ZET_DEBUG_EVENT_FLAG_NEED_ACK set MUST be acknowledged.
// Missing an ACK silently blocks all subsequent events on that session.

inline void ZeAckEvent(zet_debug_session_handle_t session,
                       const zet_debug_event_t &event) {
  if ((event.flags & ZET_DEBUG_EVENT_FLAG_NEED_ACK) == 0)
    return;
  ze_result_t result = zetDebugAcknowledgeEvent(session, &event);
  (void)result;
}

// ---------------------------------------------------------------------------
// Lane TID encoding / decoding
// ---------------------------------------------------------------------------
// Threads reported to LLDB are per-lane. The 64-bit lldb::tid_t is packed:
//
//     bit 63 .. 48   47 .. 32   31 .. 16   15 .. 8   7 .. 0
//        slice       subslice      eu       thr+1     lane
//        16 bits     16 bits    16 bits    8 bits    8 bits
//
// The `+1` on the hardware thread index keeps every EU-lane TID >= 0x100,
// leaving 0 (LLDB_INVALID_THREAD_ID) and 1 (INTELGT_SHADOW_THREAD_ID)
// available. The extra byte also prevents low-bit aliasing between two
// EU threads that only differ in `thread`.

struct LaneTIDFields {
  uint32_t slice;
  uint32_t subslice;
  uint32_t eu;
  uint32_t thread; // hardware thread index (already decoded, no +1 offset)
  uint32_t lane;
};

/// Encode a per-lane TID for a given EU thread and SIMD lane.
inline lldb::tid_t EncodeLaneTID(ze_device_thread_t ze_thread, uint32_t lane) {
  return (static_cast<uint64_t>(ze_thread.slice) << 48) |
         (static_cast<uint64_t>(ze_thread.subslice) << 32) |
         (static_cast<uint64_t>(ze_thread.eu) << 16) |
         (static_cast<uint64_t>(ze_thread.thread + 1) << 8) |
         static_cast<uint64_t>(lane & 0xFF);
}

/// Reverse of EncodeLaneTID.
inline LaneTIDFields DecodeLaneTID(lldb::tid_t tid) {
  LaneTIDFields f;
  f.slice = static_cast<uint32_t>((tid >> 48) & 0xFFFF);
  f.subslice = static_cast<uint32_t>((tid >> 32) & 0xFFFF);
  f.eu = static_cast<uint32_t>((tid >> 16) & 0xFFFF);
  f.thread = static_cast<uint32_t>(((tid >> 8) & 0xFF) - 1);
  f.lane = static_cast<uint32_t>(tid & 0xFF);
  return f;
}

/// Extract just the ze_device_thread_t portion of a lane TID.
inline ze_device_thread_t LaneTIDToZeThread(lldb::tid_t tid) {
  LaneTIDFields f = DecodeLaneTID(tid);
  ze_device_thread_t t{};
  t.slice = f.slice;
  t.subslice = f.subslice;
  t.eu = f.eu;
  t.thread = f.thread;
  return t;
}

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_LEVELZEROHELPERS_H
