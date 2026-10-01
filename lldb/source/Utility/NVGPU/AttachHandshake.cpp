//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/AttachHandshake.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private::nvgpu {

bool fromJSON(const Value &value, AttachHandshake &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("api_client_pid", data.api_client_pid) &&
         o.map("api_client_revision", data.api_client_revision) &&
         o.map("session_id", data.session_id) &&
         o.map("capabilities", data.capabilities) &&
         o.mapOptional("injection_path", data.injection_path);
}

Value toJSON(const AttachHandshake &data) {
  Object object{{"api_client_pid", data.api_client_pid},
                {"api_client_revision", data.api_client_revision},
                {"session_id", data.session_id},
                {"capabilities", data.capabilities}};
  if (data.injection_path)
    object["injection_path"] = *data.injection_path;
  return Value(std::move(object));
}

} // namespace lldb_private::nvgpu
