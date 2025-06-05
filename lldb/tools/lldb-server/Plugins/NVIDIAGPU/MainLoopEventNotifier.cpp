//===-- MainLoopFDNotifier.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MainLoopEventNotifier.h"
#include "Utils.h"
#include "lldb/Host/File.h"

using namespace lldb_private;
using namespace lldb_private::lldb_server;

MainLoopEventNotifier::MainLoopEventNotifier(llvm::StringRef name,
                                             std::function<void()> callback)
    : m_name(name), m_callback(callback) {
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, m_fds) == -1) {
    m_fds[0] = -1;
    m_fds[1] = -1;
  }
  m_files[0] = std::make_shared<NativeFile>(m_fds[0], File::eOpenOptionReadOnly,
                                            /*take_ownership=*/false);
  m_files[1] =
      std::make_shared<NativeFile>(m_fds[1], File::eOpenOptionWriteOnly,
                                   /*take_ownership=*/false);
}

void MainLoopEventNotifier::InvokeCallback() {
  char buf[1];
  // Read 1 bytes from the fd
  if (recv(GetReadFD(), buf, sizeof(buf), 0) == 1)
    return m_callback();

  // We crash if we can't read the file descriptor, as this would be a
  // terrible bug in the plugin implementation.
  logAndReportFatalError("Couldn't read the {0} file descriptor", m_name);
}

llvm::Expected<MainLoopEventNotifierUP>
MainLoopEventNotifier::CreateForEventCallback(llvm::StringRef name,
                                              MainLoop &main_loop,
                                              std::function<void()> callback) {
  MainLoopEventNotifierUP notifier(new MainLoopEventNotifier(name, callback));
  Status error;
  notifier->m_debugger_api_main_loop_handle = main_loop.RegisterReadObject(
      notifier->GetReadFile(),
      [notifier_ptr = notifier.get()](MainLoopBase &loop) {
        notifier_ptr->InvokeCallback();
      },
      error);

  if (error.Fail())
    return createStringErrorFmt(
        "Failed to register the {0} file descriptor. {1}", name, error);
  return notifier;
}

MainLoopEventNotifier::~MainLoopEventNotifier() {
  if (m_fds[0] != -1) {
    close(m_fds[0]);
    m_fds[0] = -1;
  }
  if (m_fds[1] != -1) {
    close(m_fds[1]);
    m_fds[1] = -1;
  }
}

void MainLoopEventNotifier::FireEvent() {
  if (m_fds[1] == -1)
    return;
  if (write(m_fds[1], "1", 1) != 1)
    logAndReportFatalError("Failed to send data to the {0} file descriptor",
                           m_name);
}