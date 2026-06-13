// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <hip/hip_runtime.h>

// Second translation unit -> a second code object at a different offset in the
// same a.out (same FileSpec, different slice).
__global__ void kernel_beta(int *o, int n) {
  int i = threadIdx.x;
  if (i < n) {
    o[i] = i + 1000; // BETA BREAKPOINT
  }
}

void launch_beta(int *d, int n) { kernel_beta<<<1, 32>>>(d, n); }
