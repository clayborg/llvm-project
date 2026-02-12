//===-- AmdGpuAddressSpaces.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_AMDGPUADDRESSSPACES_H
#define LLDB_UTILITY_AMDGPUADDRESSSPACES_H

#include "llvm/Support/Error.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <cassert>
#include <cstdint>
#include <functional>

namespace lldb_private {

// The address spaces defined here are based on the AMDGPU guide:
// https://llvm.org/docs/AMDGPUUsage.html#address-space-identifier
//
// These are the "dwarf" address spaces, which are used in DWARF operations as
// opposed to the "llvm" address spaces, which are used in LLVM IR. The amddbg
// api has its own internal representation of address spaces. We can translate
// between the two using the following amddbg api call:
// `amd_dbgapi_dwarf_address_space_to_address_space`.
//
enum class DW_ASPACE_AMDGPU : uint64_t {
  /// Address space that allows location expressions to specify the flat address
  /// space.
  generic = 1,

  /// Address space for GDS memory that is only present on older AMD GPUs.
  region = 2,

  /// Address space allows location expressions to specify the local address
  /// space corresponding to the wavefront that is executing the focused thread
  /// of execution.
  local = 3,

  /// Reserved for future use.
  RESERVED_4 = 4,

  /// Address space that allows location expressions to specify the private
  /// address space corresponding to the lane that is executing the focused
  /// thread of execution for languages that are implemented using a SIMD or
  /// SIMT execution model.
  private_lane = 5,

  /// Address space that allows location expressions to specify the unswizzled
  /// private address space corresponding to the wavefront that is executing the
  /// focused thread of execution
  private_wave = 6,
};

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
  assert(false && "unhandled amd_dbgapi_status_t value");
}

#undef AMD_DBGAPI_ENUM_TO_CSTR

/// Run a command from the amd-dbgapi library and return an llvm::Error if not
/// successful.
///
/// \param func A callable that returns an amd_dbgapi_status_t.
/// \return An llvm::Error indicating success or failure with a descriptive
///         message.
inline llvm::Error
RunAmdDbgApiCommand(std::function<amd_dbgapi_status_t()> func) {
  amd_dbgapi_status_t status = func();
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AMD_DBGAPI_STATUS_ERROR: %s",
                                   AmdDbgApiStatusToString(status));
  return llvm::Error::success();
}

} // namespace lldb_private

#endif // LLDB_UTILITY_AMDGPUADDRESSSPACES_H
