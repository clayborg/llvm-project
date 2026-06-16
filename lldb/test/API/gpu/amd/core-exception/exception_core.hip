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

// LLDB lists the later wave first for this core. Keep that later wave stopped
// normally while an earlier wave triggers the memory violation, so default
// first-thread selection is not enough to pass the selected-thread test.
__global__ void memory_violation_kernel(int *buf, unsigned int *ready,
                                        unsigned int *keep_alive, size_t size) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  bool faulting_block = blockIdx.x == 0;
  if (!faulting_block && threadIdx.x == 0) {
    atomicAdd(ready, 1u);
  }

  if (faulting_block && threadIdx.x == 0) {
    while (atomicAdd(ready, 0u) < gridDim.x - 1) {
    }
    buf[size * 1024 * 1024] = 42; // GPU EXCEPTION
  }

  if (!faulting_block) {
    while (atomicAdd(keep_alive, 0u) == 0) {
    }
  }

  if (tid < size) {
    buf[tid] = tid; // GPU BREAKPOINT
  }
}

int main() {
  const size_t num_threads = 64;
  const size_t buffer_size = num_threads * sizeof(int);

  int *d_buf;
  unsigned int *d_ready;
  unsigned int *d_keep_alive;
  HIP_CHECK(hipMalloc(&d_buf, buffer_size));
  HIP_CHECK(hipMalloc(&d_ready, sizeof(*d_ready)));
  HIP_CHECK(hipMalloc(&d_keep_alive, sizeof(*d_keep_alive)));
  HIP_CHECK(hipMemset(d_ready, 0, sizeof(*d_ready)));
  HIP_CHECK(hipMemset(d_keep_alive, 0, sizeof(*d_keep_alive)));

  printf("Launching kernel...\n"); // CPU BREAKPOINT - BEFORE LAUNCH

  memory_violation_kernel<<<dim3(2), dim3(num_threads), 0, hipStreamDefault>>>(
      d_buf, d_ready, d_keep_alive, num_threads);

  HIP_CHECK(hipDeviceSynchronize());

  printf("Kernel finished.\n");

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipFree(d_ready));
  HIP_CHECK(hipFree(d_keep_alive));

  return 0;
}
