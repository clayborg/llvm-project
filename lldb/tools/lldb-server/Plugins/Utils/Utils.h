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
#include "lldb/Host/common/NativeThreadProtocol.h"
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

/// Helper class to provide range-based iteration over
/// std::vector<std::unique_ptr<NativeThreadProtocol>> with automatic casting to
/// the underlying thread `T&` type.
template <typename T> class GPUThreadRange {
public:
  /// Iterator class that automatically casts to the underlying thread `T&`
  /// type.
  class iterator {
  public:
    /// Constructor for the iterator.
    ///
    /// \param[in] it
    ///     Iterator to the underlying thread container.
    iterator(std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator it)
        : m_it(it) {}

    /// Dereference operator with automatic casting to underlying thread `T&`
    /// type.
    ///
    /// \return
    ///     Reference to Thread object.
    T &operator*() const { return static_cast<T &>(**m_it); }

    /// Pre-increment operator.
    ///
    /// \return
    ///     Reference to incremented iterator.
    iterator &operator++() {
      ++m_it;
      return *this;
    }

    /// InEquality comparison operator.
    ///
    /// \param[in] other
    ///     Iterator to compare against.
    ///
    /// \return
    ///     True if iterators are not equal, false otherwise.
    bool operator!=(const iterator &other) const { return m_it != other.m_it; }

    /// Equality comparison operator.
    ///
    /// \param[in] other
    ///     Iterator to compare against.
    ///
    /// \return
    ///     True if iterators are equal, false otherwise.
    bool operator==(const iterator &other) const { return m_it == other.m_it; }

  private:
    std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator m_it;
  };

  /// Constructor for the range object.
  ///
  /// \param[in] threads
  ///     Reference to the thread container.
  GPUThreadRange(std::vector<std::unique_ptr<NativeThreadProtocol>> &threads)
      : m_threads(threads) {}

  /// Get iterator to the beginning of the range.
  ///
  /// \return
  ///     Iterator pointing to the first element.
  iterator begin() { return iterator(m_threads.begin()); }

  /// Get iterator to the end of the range.
  ///
  /// \return
  ///     Iterator pointing past the last element.
  iterator end() { return iterator(m_threads.end()); }

private:
  std::vector<std::unique_ptr<NativeThreadProtocol>> &m_threads;
};
} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_UTILS_H
