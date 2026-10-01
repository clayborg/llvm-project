//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_NVGPU_ATTACHHANDSHAKE_H
#define LLDB_UTILITY_NVGPU_ATTACHHANDSHAKE_H

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

namespace lldb_private::nvgpu {

/// The values lldb writes into the application's libcuda before it asks the
/// driver to attach to a running process. Only lldb-server knows them, so the
/// NVGPU plug-in sends them as the "platform_data" of its
/// "jGPUPluginInitialize" reply.
struct AttachHandshake {
  /// The pid of the process that runs the debugger API, which is lldb-server.
  uint32_t api_client_pid = 0;
  /// The API revision to announce. The negotiated one replaces it once the
  /// debug engine is up.
  uint32_t api_client_revision = 0;
  uint32_t session_id = 0;
  /// The CUDBG_DEBUGGER_CAPABILITY_* flags to request.
  uint32_t capabilities = 0;
  /// The debug engine to inject instead of the driver's own, if any.
  std::optional<std::string> injection_path;
};

bool fromJSON(const llvm::json::Value &value, AttachHandshake &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AttachHandshake &data);

} // namespace lldb_private::nvgpu

#endif // LLDB_UTILITY_NVGPU_ATTACHHANDSHAKE_H
