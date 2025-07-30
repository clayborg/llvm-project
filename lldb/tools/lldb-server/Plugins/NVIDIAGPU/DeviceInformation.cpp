//===-- DeviceInformation.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceInformation.h"
#include "Utils.h"

using namespace lldb_private::lldb_server;

size_t DeviceInformation::GetNumRRegisters() {
  if (m_num_r_registers)
    return *m_num_r_registers;

  uint32_t num_r_registers = 0;
  CUDBGResult res = m_api.getNumRegisters(m_device_id, &num_r_registers);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("DeviceInformation::GetNumRRegisters(). "
                           "getNumRegisters failed: {0}",
                           cudbgGetErrorString(res));
  }
  m_num_r_registers = static_cast<size_t>(num_r_registers);
  return *m_num_r_registers;
}
