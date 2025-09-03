//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_NVIDIAGPU_MAINLOOPEVENTNOTIFIER_H
#define LLDB_PLUGINS_NVIDIAGPU_MAINLOOPEVENTNOTIFIER_H

#include <sys/socket.h>

#include "lldb/Host/MainLoop.h"

namespace lldb_private::lldb_server {

class MainLoopEventNotifier;
using MainLoopEventNotifierUP = std::unique_ptr<MainLoopEventNotifier>;

/// This class is used to notify a main loop that an event has occurred.
/// It uses a file descriptor pair to communicate with the main loop, which is
/// in a different thread. The file descriptor pair is probably the fastest way
/// to achieve this inter-thread communication.
class MainLoopEventNotifier {
public:
  ~MainLoopEventNotifier();

  /// Create a notifier for an event callback registering it to the given main
  /// loop.
  ///
  /// \param name The name of the notifier for error reporting purposes.
  /// \param main_loop The main loop to register the notifier to.
  /// \param callback The callback to invoke when the event is fired.
  ///
  /// \returns A unique pointer to the notifier or an error.
  static llvm::Expected<MainLoopEventNotifierUP>
  CreateForEventCallback(llvm::StringRef name, MainLoop &main_loop,
                         std::function<void()> callback);

  /// Fire an event. This will invoke the callback.
  void FireEvent();

private:
  MainLoopEventNotifier(llvm::StringRef name, std::function<void()> callback);

  void InvokeCallback();

  int GetReadFD() const { return m_fds[0]; }
  int GetWriteFD() const { return m_fds[1]; }

  lldb::FileSP GetReadFile() const { return m_files[0]; }
  lldb::FileSP GetWriteFile() const { return m_files[1]; }

  /// The name of the notifier.
  std::string m_name;
  /// Raw fds for the notifier.
  int m_fds[2] = {-1, -1};
  /// Corresponding FileSPs for the fds. The MainLoop class wants these objects.
  lldb::FileSP m_files[2] = {nullptr, nullptr};
  /// The handle gotten after registering an event handler in the MainLoop
  /// class.
  MainLoop::ReadHandleUP m_debugger_api_main_loop_handle;
  /// The callback to invoke when the event is fired.
  std::function<void()> m_callback;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_PLUGINS_NVIDIAGPU_MAINLOOPEVENTNOTIFIER_H
