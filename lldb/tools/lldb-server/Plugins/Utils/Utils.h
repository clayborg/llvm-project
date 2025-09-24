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
[[noreturn]] void logAndReportFatalError(llvm::StringRef err_msg);

template <typename... Args>
[[noreturn]] void logAndReportFatalError(const char *format, Args &&...args) {
  logAndReportFatalError(llvm::formatv(format, args...).str());
}

/// Variant of `logAndReportFatalError` that allows passing a callback that
/// gets invoked with the error message.
template <typename... Args>
[[noreturn]] void logAndReportFatalError(
    std::function<void(llvm::StringRef)> additional_log_callback,
    const char *format, Args &&...args) {
  std::string err_msg = llvm::formatv(format, args...).str();
  additional_log_callback(err_msg);
  logAndReportFatalError(err_msg);
}

/// Get a user-friendly string representation of a state.
llvm::StringRef StateToString(lldb::StateType state);

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_UTILS_H
