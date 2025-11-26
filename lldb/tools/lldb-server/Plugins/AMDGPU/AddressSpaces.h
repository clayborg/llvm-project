//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_AMDGPU_ADDRESSSPACES_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_AMDGPU_ADDRESSSPACES_H

#include <cstdint>

namespace lldb_private::lldb_server {

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

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_AMDGPU_ADDRESSSPACES_H
