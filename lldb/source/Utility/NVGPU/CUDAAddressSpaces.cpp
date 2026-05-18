//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/NVGPU/CUDAAddressSpaces.h"

using namespace lldb_private;

// is_thread_specific should be true for all address spaces that may return
// a different value for different threads.
std::vector<AddressSpaceInfo> nvgpu::GetAddressSpaceInfos() {
  return {
      {"const", ConstStorage, /*is_thread_specific=*/false},
      {"global", GlobalStorage, /*is_thread_specific=*/false},
      {"local", LocalStorage, /*is_thread_specific=*/true},
      {"param", ParamStorage, /*is_thread_specific=*/true},
      {"shared", SharedStorage, /*is_thread_specific=*/true},
      {"generic", GenericStorage, /*is_thread_specific=*/true},
  };
}
