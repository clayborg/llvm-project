//===-- AmdGpuCoreUtils.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AmdGpuCoreUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb_private;

std::optional<GPUDynamicLoaderLibraryInfo>
lldb_private::ParseLibraryInfo(const AmdGpuCodeObject &code_object) {
  // This function will parse the shared library string that AMD's GPU driver
  // sends to the debugger. The format is one of:
  //  file://<path>#offset=<file-offset>&size=<file-size>
  //  memory://<name>#offset=<image-addr>&size=<image-size>
  //
  // For file:// URIs, pathname is set to the real file path so LLDB can locate
  // the file on disk and resolve dwp/dwo debug info.
  // For memory:// URIs, pathname is set to "amd_memory_kernel[start, end)" for
  // uniqueness since memory modules are not used for file probing.
  GPUDynamicLoaderLibraryInfo lib_info;
  lib_info.load = code_object.is_loaded;
  lib_info.load_address = code_object.load_address;

  auto get_offset_and_size = [](llvm::StringRef &values,
                                std::optional<uint64_t> &offset,
                                std::optional<uint64_t> &size) {
    offset = std::nullopt;
    size = std::nullopt;
    llvm::StringRef value;
    uint64_t uint_value;
    std::tie(value, values) = values.split('&');
    while (!value.empty()) {
      if (value.consume_front("offset=")) {
        if (!value.getAsInteger(0, uint_value))
          offset = uint_value;
      } else if (value.consume_front("size=")) {
        if (!value.getAsInteger(0, uint_value))
          size = uint_value;
      }
      std::tie(value, values) = values.split('&');
    }
  };

  llvm::StringRef lib_spec = code_object.uri;
  if (lib_spec.consume_front("file://")) {
    llvm::StringRef path, values;
    std::tie(path, values) = lib_spec.split('#');
    if (path.empty())
      return std::nullopt;
    lib_info.pathname = path.str();
    get_offset_and_size(values, lib_info.file_offset, lib_info.file_size);
  } else if (lib_spec.consume_front("memory://")) {
    llvm::StringRef name, values;
    std::tie(name, values) = lib_spec.split('#');
    if (name.empty())
      return std::nullopt;
    get_offset_and_size(values, lib_info.native_memory_address,
                        lib_info.native_memory_size);
    // We must have a valid address and size for memory objects.
    if (!(lib_info.native_memory_address.has_value() &&
          lib_info.native_memory_size.has_value()))
      return std::nullopt;
    uint64_t start = *lib_info.native_memory_address;
    uint64_t end = start + *lib_info.native_memory_size;
    lib_info.pathname =
        llvm::formatv("amd_memory_kernel[{0}, {1})", llvm::format_hex(start, 1),
                      llvm::format_hex(end, 1));
  } else {
    return std::nullopt;
  }

  return lib_info;
}
