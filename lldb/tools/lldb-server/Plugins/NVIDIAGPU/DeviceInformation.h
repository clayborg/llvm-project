//===-- DeviceInformation.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICE_INFORMATION_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICE_INFORMATION_H

#include "cudadebugger.h"
#include <cstddef>
#include <optional>

namespace lldb_private::lldb_server {

/// Class that can be used to query and cache the information about a device.
class DeviceInformation {
public:
  DeviceInformation(const CUDBGAPI_st &api, int device_id)
      : m_api(api), m_device_id(device_id) {}

  /// \return the number of R registers for the device. The result is cached.
  size_t GetNumRRegisters();

private:
  CUDBGAPI_st m_api;
  int m_device_id;
  std::optional<size_t> m_num_r_registers;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICE_INFORMATION_H
