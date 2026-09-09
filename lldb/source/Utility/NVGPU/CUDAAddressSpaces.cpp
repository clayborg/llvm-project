//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/CUDAAddressSpaces.h"

#include <array>

using namespace lldb_private;

// is_thread_specific should be true for all address spaces that may return
// a different value for different threads.
static const std::array<AddressSpaceInfo, 6> g_address_space_infos = {{
    {"const", nvgpu::ConstStorage, /*is_thread_specific=*/false},
    {"global", nvgpu::GlobalStorage, /*is_thread_specific=*/false},
    {"local", nvgpu::LocalStorage, /*is_thread_specific=*/true},
    {"param", nvgpu::ParamStorage, /*is_thread_specific=*/true},
    {"shared", nvgpu::SharedStorage, /*is_thread_specific=*/true},
    {"generic", nvgpu::GenericStorage, /*is_thread_specific=*/true},
}};

llvm::ArrayRef<AddressSpaceInfo> nvgpu::GetAddressSpaceInfos() {
  return g_address_space_infos;
}

const AddressSpaceInfo *
nvgpu::FindAddressSpaceInfo(lldb::addr_space_t space_id) {
  for (const AddressSpaceInfo &info : g_address_space_infos) {
    if (info.space_id == space_id)
      return &info;
  }
  return nullptr;
}
