//===-- AddressSpaces.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_ADDRESSSPACES_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_ADDRESSSPACES_H

#include "lldb/Utility/NVGPU/CUDAAddressSpaces.h"

namespace lldb_private::lldb_server {

using AddressSpace = lldb_private::nvgpu::AddressSpace;

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_ADDRESSSPACES_H
