//===-- AmdGpuSymbolLoader.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_AMDGPUSYMBOLLOADER_H
#define LLDB_TARGET_AMDGPUSYMBOLLOADER_H

#include "lldb/lldb-forward.h"

namespace lldb_private {

class Target;

/// Attempt to attach separate debug info to an AMDGPU code-object module.
///
/// AMDGPU core files can create modules for code objects before their separate
/// debug info has been attached. This helper resolves the module's UUID/build
/// ID through LLDB's external symbol lookup path, attaches the resolved symbol
/// file, creates its SymbolFile, and notifies the target.
///
/// Lookup follows LLDB's normal external-symbol order: the target platform's
/// locate-module callback, registered symbol-locator plugins and debug-file
/// search paths, and explicit symbol-locator download callbacks.
///
/// Returns true if a symbol file was found and loaded.
bool LoadAmdGpuCodeObjectSymbols(Target &target,
                                 const lldb::ModuleSP &module_sp);

} // namespace lldb_private

#endif // LLDB_TARGET_AMDGPUSYMBOLLOADER_H
