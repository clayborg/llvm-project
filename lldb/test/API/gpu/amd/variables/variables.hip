#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the
/// program with an error code.
constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                        \
    if (error != hipSuccess) {                                                 \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error)      \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;       \
      std::exit(error_exit_code);                                              \
    }                                                                          \
  }

// User-defined struct type
struct Point {
  int x;
  int y;
};

// Global device variables (accessible by all threads across all blocks)
__device__ int g_scalar = 100;
__device__ int g_array[4] = {10, 20, 30, 40};
__device__ Point g_struct = {50, 60};

// Kernel to demonstrate different variable types and scopes
__global__ void test_variables_kernel(int *output, int num_elements) {
  // Get thread and block indices
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  if (gid >= num_elements)
    return;

  // ====== THREAD LOCAL VARIABLES ======
  // These are private to each thread, stored in registers or local memory

  // Thread local scalar
  int thread_scalar = tid * 2;

  // Thread local array
  int thread_array[4];
  for (int i = 0; i < 4; i++) {
    thread_array[i] = tid + i;
  }

  // Thread local struct
  Point thread_struct;
  thread_struct.x = tid * 10;
  thread_struct.y = tid * 20;

  // ====== SHARED VARIABLES ======
  // These are shared among all threads in the same block

  // Shared scalar
  __shared__ int shared_scalar;

  // Shared array
  __shared__ int shared_array[32];

  // Shared struct
  __shared__ Point shared_struct;

  // Initialize shared variables (let thread 0 do this)
  if (tid == 0) {
    shared_scalar = bid + 1000;
    shared_struct.x = bid + 100;
    shared_struct.y = bid + 200;
  }

  // Each thread initializes its own element in shared array
  if (tid < 32) {
    shared_array[tid] = tid;
  }

  // Synchronize to ensure all shared memory is initialized
  __syncthreads();

  // ====== ACCESSING GLOBAL VARIABLES ======
  // Global variables are accessible by all threads

  int global_scalar_value = g_scalar;
  int global_array_value = g_array[tid % 4];
  Point global_struct_value = g_struct;

  // ====== COMPUTE OUTPUT ======
  // Combine all variable types for demonstration
  int result = 0;

  // Add thread local contributions
  result += thread_scalar;
  result += thread_array[tid % 4];
  result += thread_struct.x + thread_struct.y;

  // Add shared variable contributions
  result += shared_scalar;
  result += shared_array[tid % 32];
  result += shared_struct.x + shared_struct.y;

  // Add global variable contributions
  result += global_scalar_value;
  result += global_array_value;
  result += global_struct_value.x + global_struct_value.y;

  // Synchronize before writing output
  __syncthreads();

  // Write result to global memory
  output[gid] = result; // GPU BREAKPOINT
}

int main() {
  std::cout << "=== HIP Variables Test ===" << std::endl;
  std::cout << "Testing thread local, shared, and global variables"
            << std::endl;
  std::cout << "with scalars, arrays, and structs" << std::endl << std::endl;

  // Set up execution configuration
  const int num_blocks = 1;
  const int threads_per_block = 32;
  const int num_elements = num_blocks * threads_per_block;

  std::cout << "Launch configuration: " << num_blocks << " blocks, "
            << threads_per_block << " threads per block" << std::endl;
  std::cout << "Total threads: " << num_elements << std::endl << std::endl;

  // Allocate host memory
  int *h_output = new int[num_elements]; // CPU BREAKPOINT - BEFORE LAUNCH

  // Allocate device memory
  int *d_output;
  HIP_CHECK(hipMalloc(&d_output, num_elements * sizeof(int)));

  // Launch kernel
  hipLaunchKernelGGL(test_variables_kernel, dim3(num_blocks),
                     dim3(threads_per_block), 0, 0, d_output, num_elements);

  // Check for kernel launch errors
  HIP_CHECK(hipGetLastError());

  // Wait for kernel to complete
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back to host
  HIP_CHECK(hipMemcpy(h_output, d_output, num_elements * sizeof(int),
                      hipMemcpyDeviceToHost));

  // Display results
  std::cout << "Output from kernel (first 10 elements):" << std::endl;
  for (int i = 0; i < 10 && i < num_elements; i++) {
    std::cout << "  Thread " << i << ": " << h_output[i] << std::endl;
  }

  std::cout << std::endl << "Verification:" << std::endl;
  std::cout << "Thread 0 from block 0: " << h_output[0] << std::endl;
  std::cout << "Thread 1 from block 0: " << h_output[1] << std::endl;
  std::cout << "Thread 0 from block 1: " << h_output[32] << " (if available)"
            << std::endl;

  // Clean up
  HIP_CHECK(hipFree(d_output));
  delete[] h_output;

  std::cout << std::endl << "Test completed successfully!" << std::endl;

  return 0;
}
