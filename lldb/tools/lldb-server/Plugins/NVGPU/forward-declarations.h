//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_NVGPU_FORWARD_DECLARATIONS_H
#define LLDB_TOOLS_LLDB_SERVER_NVGPU_FORWARD_DECLARATIONS_H

#include "lldb/Utility/NVGPU/CUDAException.h"

namespace lldb_private {
class TCPSocket;

namespace lldb_server {
class DeviceState;
class MainLoopEventNotifier;
class ProcessNVGPU;
class SMState;
class ThreadNVGPU;
class ThreadState;
class WarpState;
using ExceptionInfo = lldb_private::CUDAExceptionInfo;
} // namespace lldb_server

} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_NVGPU_FORWARD_DECLARATIONS_H
