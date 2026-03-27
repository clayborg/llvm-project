#include <hip/hip_runtime.h>

#include <cstdio>
#include <iostream>

constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                              \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      std::exit(error_exit_code);                                         \
    }                                                                     \
  }

// Kernel that performs an out-of-bounds write to trigger a memory violation.
__global__ void memory_violation_kernel(int *buf, size_t size) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    buf[size * 1024 * 1024] = 42; // GPU EXCEPTION
  }
  if (tid < size) {
    buf[tid] = tid; // GPU BREAKPOINT
  }
}

int main() {
  const size_t num_threads = 64;
  const size_t buffer_size = num_threads * sizeof(int);

  int *d_buf;
  HIP_CHECK(hipMalloc(&d_buf, buffer_size));

  printf("Launching kernel...\n"); // CPU BREAKPOINT - BEFORE LAUNCH

  memory_violation_kernel<<<dim3(1), dim3(num_threads), 0,
                            hipStreamDefault>>>(d_buf, num_threads);

  HIP_CHECK(hipDeviceSynchronize());

  printf("Kernel finished.\n");

  HIP_CHECK(hipFree(d_buf));

  return 0;
}
