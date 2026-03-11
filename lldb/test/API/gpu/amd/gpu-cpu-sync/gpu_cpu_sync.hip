// Test program for GPU-CPU synchronization.
// This program launches a GPU kernel without any CPU-side breakpoints.
// The debugger should be able to stop the CPU process when the GPU target
// is created, set a GPU breakpoint, and hit it.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                         \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error: \"%s\" at %s:%d\n",                          \
              hipGetErrorString(error), __FILE__, __LINE__);                    \
      exit(-1);                                                                \
    }                                                                          \
  }

__global__ void simple_kernel(int *out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    out[idx] = idx * idx; // GPU BREAKPOINT
  }
}

int main() {
  const int n = 64;
  int *d_out;
  HIP_CHECK(hipMalloc(&d_out, n * sizeof(int)));

  // Launch the kernel — no CPU breakpoints here.
  simple_kernel<<<1, n>>>(d_out, n);

  int h_out[64];
  HIP_CHECK(hipMemcpy(h_out, d_out, n * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));

  // Verify a few values.
  printf("out[0]=%d out[1]=%d out[7]=%d\n", h_out[0], h_out[1], h_out[7]);
  return 0;
}
