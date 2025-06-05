//===-- Utils.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

using namespace lldb;

namespace lldb_private::lldb_server {

llvm::StringRef StateToString(lldb::StateType state) {
  switch (state) {
  case eStateStopped:
    return "stopped";
  case eStateRunning:
    return "running";
  case eStateInvalid:
    return "invalid";
  case eStateUnloaded:
    return "unloaded";
  case eStateConnected:
    return "connected";
  case eStateAttaching:
    return "attaching";
  case eStateLaunching:
    return "launching";
  case eStateStepping:
    return "stepping";
  case eStateCrashed:
    return "crashed";
  case eStateDetached:
    return "detached";
  case eStateExited:
    return "exited";
  case eStateSuspended:
    return "suspended";
  default:
    return "unknown";
  }
}

} // namespace lldb_private::lldb_server