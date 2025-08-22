//===-- AddressSpace.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ADDRESSSPACE_H
#define LLDB_UTILITY_ADDRESSSPACE_H

#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

/// See docs/lldb-gdb-remote.txt for more information.
namespace lldb_private {

struct AddressSpaceInfo {
  std::string name; ///< The name of the address space.
  uint64_t value; ///< The integer identifier of the address space.
  bool is_thread_specific; ///< True if the address space is thread specific.
};

bool fromJSON(const llvm::json::Value &value, AddressSpaceInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AddressSpaceInfo &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ADDRESSSPACE_H
