#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                              \
  {                                                                       \
    const hipError_t error = (condition);                                 \
    if (error != hipSuccess) {                                            \
      fprintf(stderr, "HIP error: \"%s\" at %s:%d\n",                    \
              hipGetErrorString(error), __FILE__, __LINE__);              \
      exit(error_exit_code);                                              \
    }                                                                     \
  }

__global__ void add_one(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx + 1; // GPU BREAKPOINT
}

int main() {
    const int n = 4;
    int host_data[n] = {0, 0, 0, 0};
    int *device_data;
    printf("Starting GPU test...\n"); // CPU BREAKPOINT - BEFORE LAUNCH
    // Allocate device memory
    HIP_CHECK(hipMalloc(&device_data, n * sizeof(int)));
    // Copy data to device
    HIP_CHECK(hipMemcpy(device_data, host_data, n * sizeof(int), hipMemcpyHostToDevice));
    // Launch kernel (single block, 4 threads)
    add_one<<<1, n>>>(device_data);
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    // Wait for GPU to finish
    HIP_CHECK(hipDeviceSynchronize());
    // Copy results back to host
    HIP_CHECK(hipMemcpy(host_data, device_data, n * sizeof(int), hipMemcpyDeviceToHost));
    printf("Results: %d %d %d %d\n", host_data[0], host_data[1], host_data[2], host_data[3]); // CPU BREAKPOINT - AFTER LAUNCH
    // Cleanup
    HIP_CHECK(hipFree(device_data));
    return 0;
}
