//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_EXCEPTIONS_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_EXCEPTIONS_H

#include "Coords.h"
#include "cudadebugger.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace lldb_private::lldb_server {

/// This holds the information about an exception and the thread that caused it.
struct ExceptionInfo {
  PhysicalCoords physical_coords;
  CUDBGException_t exception;
};

/// Find the exception info for the first exception on the first SM.
///
/// \param[in] api
//    The CUDA debugger API.
///
/// \return
//    The exception info if found, otherwise \b std::nullopt.
std::optional<ExceptionInfo> FindExceptionInfo(const CUDBGAPI_st &api);

/// Convert a CUDA exception to a string.
///
/// \param[in] exception
//    The CUDA exception.
///
/// \return The string representation of the exception.
llvm::StringRef CudaExceptionToString(CUDBGException_t exception);

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_EXCEPTIONS_H
