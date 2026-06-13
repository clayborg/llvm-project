// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <hip/hip_runtime.h>

// Own translation unit -> its own GPU code object in a.out.
__global__ void kernel_alpha(int *o) {
  int i = threadIdx.x;
  o[i] = i * 2;
}

void launch_alpha(int *d) { kernel_alpha<<<1, 32>>>(d); }
