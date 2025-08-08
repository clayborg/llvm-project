//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Bitset.h"
#include <llvm/ADT/StringExtras.h>

using namespace lldb_private::lldb_server;

std::string DynamicBitset::AsHex() const {
  std::string hex_mask = llvm::toHex(m_bytes);
  std::reverse(hex_mask.begin(), hex_mask.end());
  return hex_mask;
}
