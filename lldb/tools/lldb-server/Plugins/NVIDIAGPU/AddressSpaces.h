//===-- AddressSpaces.h ---------------------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_ADDRESSSPACES_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_ADDRESSSPACES_H

#include <cstdint>

namespace lldb_private::lldb_server {

// The numeric values of the address spaces are the same as the ones defined in
// https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf-definitions
//
// Interesting information on generic address spaces can be found in
// https://docs.nvidia.com/cuda/parallel-thread-execution/#generic-addressing
//
// There are some additional address spaces that seem not to be used anymore by
// debuggers but are present in cudadebugger.h. We mention them here as an
// explicit acknowledgement that we decided not to support them in LLDB.
//
//   ptxCodeStorage
//   ptxRegStorage
//   ptxSregStorage
//   ptxSurfStorage
//   ptxTexStorage
//   ptxTexSamplerStorage
//   ptxIParamStorage
//   ptxOParamStorage
//   ptxFrameStorage
//   ptxURegStorage
enum AddressSpace : uint64_t {
  /// Memory used to store constant data visible to all threads.
  ConstStorage = 4,
  /// Main DRAM memory for the device accessible to all threads.
  GlobalStorage = 5,
  /// Memory that is local to each thread.
  LocalStorage = 6,
  /// Memory that is used to pass parameters to a kernel.
  ParamStorage = 7,
  /// Memory that is shared between threads in the same block.
  SharedStorage = 8,
  /// Memory that unifies global, shared and local address for a given thread.
  GenericStorage = 12,
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_ADDRESSSPACES_H
