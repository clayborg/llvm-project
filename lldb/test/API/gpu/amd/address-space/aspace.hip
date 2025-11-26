#include <cstdio>
#include <hip/hip_runtime.h>

constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                        \
    if (error != hipSuccess) {                                                 \
      printf("An error encountered: \"%s\""                                    \
             " at %s:%d",                                                      \
             hipGetErrorString(error), __FILE__, __LINE__);                    \
      std::exit(error_exit_code);                                              \
    }                                                                          \
  }

constexpr int NUM_THREADS = 64;
constexpr int SHARED_MEM_SIZE = NUM_THREADS;
__global__ void address_space_demo_kernel(int *output, int size) {
  // Private variable (per-thread)
  int idx = threadIdx.x;

  // Shared memory (local to thread block)
  // Initialize shared memory by first thread in block
  __shared__ int shared_mem[SHARED_MEM_SIZE];
  if (threadIdx.x == 0) {
    for (int i = 0; i < SHARED_MEM_SIZE; i++) {
      shared_mem[i] = i * 2;
    }
  }

  __syncthreads();

  if (idx < size) {
    output[idx] = shared_mem[idx];
  } // GPU BREAKPOINT
}
  
int *host_output = nullptr;
int *device_output = nullptr;

int main() {
  // Allocate and initialize host buffer.
  size_t output_size_in_bytes = NUM_THREADS * sizeof(int);
  host_output = (int*)malloc(output_size_in_bytes);
  memset(host_output, 0, output_size_in_bytes);

  HIP_CHECK(
      hipMalloc(&device_output, NUM_THREADS * sizeof(int))); // CPU BREAKPOINT - BEFORE LAUNCH
  // Launch kernel with 1 block of 256 threads
  hipLaunchKernelGGL(address_space_demo_kernel, dim3(1), dim3(NUM_THREADS), 0,
                     0, device_output, NUM_THREADS);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_output, device_output, NUM_THREADS * sizeof(int),
                      hipMemcpyDeviceToHost));
  // Print some results
  for (int i = 0; i < 10; i++) {
    printf("output[%d] = %d\n", i, host_output[i]);
  }
  HIP_CHECK(hipFree(device_output));
  free(host_output);
  return 0;
}
