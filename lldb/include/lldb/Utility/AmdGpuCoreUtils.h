//===-- AmdGpuCoreUtils.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_AMDGPUCOREUTILS_H
#define LLDB_UTILITY_AMDGPUCOREUTILS_H

#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace lldb_private {

/// Structure representing a code object loaded by the AMD GPU driver.
/// This is used as input to ParseLibraryInfo.
struct AmdGpuCodeObject {
  /// The URI of the code object as reported by the AMD debug API.
  /// Format can be either:
  ///   file://<path>#offset=<file-offset>&size=<file-size>
  ///   memory://<name>#offset=<image-addr>&size=<image-size>
  std::string uri;

  /// The load address of the code object in GPU memory.
  uint64_t load_address;

  /// Whether this code object is being loaded (true) or unloaded (false).
  bool is_loaded;

  AmdGpuCodeObject(llvm::StringRef uri_str, uint64_t addr, bool loaded)
      : uri(uri_str.str()), load_address(addr), is_loaded(loaded) {}
};

/// Parse AMD GPU code object URI and convert it to GPUDynamicLoaderLibraryInfo.
///
/// This function parses the URI format used by AMD's GPU driver to describe
/// loaded code objects. The URI can be in one of two formats:
///   - file://<path>#offset=<file-offset>&size=<file-size>
///   - memory://<name>#offset=<image-addr>&size=<image-size>
///
/// \param[in] code_object
///     The AMD GPU code object structure containing URI and load information.
///
/// \return
///     A GPUDynamicLoaderLibraryInfo structure if the URI was successfully
///     parsed, or std::nullopt if parsing failed.
std::optional<GPUDynamicLoaderLibraryInfo>
ParseLibraryInfo(const AmdGpuCodeObject &code_object);

} // namespace lldb_private

#endif // LLDB_UTILITY_AMDGPUCOREUTILS_H
