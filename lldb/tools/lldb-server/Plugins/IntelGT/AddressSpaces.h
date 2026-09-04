//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_ADDRESSSPACES_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_ADDRESSSPACES_H

#include <cstdint>

namespace lldb_private::lldb_server {

// DWARF address spaces for IntelGT (distinct from LLVM IR address spaces).
enum class DW_ASPACE_INTELGT : uint64_t {
  global = 0, ///< Default.
  slm = 1,    ///< Shared Local Memory (GPU).
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_ADDRESSSPACES_H
