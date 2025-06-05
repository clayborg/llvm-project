//===-- Utils.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_UTILS_H
#define LLDB_TOOLS_LLDB_SERVER_UTILS_H

#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "llvm/Support/Error.h"

namespace lldb_private::lldb_server {

/// Variant of `createStringError` that uses `formatv` to format the error
/// message. This allows complex string interpolations.
template <typename... Args>
llvm::Error createStringErrorFmt(const char *format, Args &&...args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 llvm::formatv(format, args...).str());
}

/// This is the preferred way to abort the lldb-server due to a programmer
/// error. It logs the error message and then calls `llvm::report_fatal_error`
/// which will cause the lldb-server to crash and print a backtrace. It's worth
/// mentioning that the backtrace is only printed if lldb-server is started
/// manually on a terminal.
template <typename... Args>
void logAndReportFatalError(const char *format, Args &&...args) {
  Log *log = GetLog(process_gdb_remote::GDBRLog::Plugin);
  std::string err_msg = llvm::formatv(format, args...).str();
  LLDB_LOG(log, "{0}", err_msg);
  llvm::report_fatal_error(llvm::createStringError(err_msg));
}

/// Get a user-friendly string representation of a state.
llvm::StringRef StateToString(lldb::StateType state);

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_UTILS_H