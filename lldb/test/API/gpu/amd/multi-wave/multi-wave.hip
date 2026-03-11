#include <cstddef>
#include <hip/hip_runtime.h>

#include <iostream>
#include <limits>

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the
/// program with an error code.
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

static constexpr unsigned divideCeil(unsigned Numerator, unsigned Denominator) {
  assert(Denominator && "Division by zero");
  uint64_t Bias = (Numerator != 0);
  return (Numerator - Bias) / Denominator + Bias;
}

static std::string get_num_threads_in_wave(unsigned num_threads_in_block, unsigned wave_size ) {
  std::vector<unsigned> num_threads_in_wave(divideCeil(num_threads_in_block, wave_size), wave_size);
  if (num_threads_in_block % wave_size != 0) {
    num_threads_in_wave.back() = num_threads_in_block % wave_size;
  }
  
  std::string result = "[";
  for (size_t i = 0; i < num_threads_in_wave.size(); ++i) {
    result += std::to_string(num_threads_in_wave[i]);
    if (i != num_threads_in_wave.size() - 1) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

__device__ unsigned block_counter = 0;
__device__ void wait_for_all_threads() {
  // Wait until all threads in the block have reached this point.
  __syncthreads();

  // Wait until all blocks have reached this point.
  const unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
  const bool is_first_thread_in_block = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (is_first_thread_in_block) {
    atomicAdd(&block_counter, 1);
    while (atomicAdd(&block_counter, 0) < num_blocks) {
    }
  }

  // Wait until all threads in the block have reached this point.
  // This is to ensure that all blocks have incremented the block counter
  // before continuing.
  __syncthreads();
}

__device__ unsigned int get_global_idx() {
  // Calculate block ID in the grid
  unsigned block_id =
      blockIdx.x + blockIdx.y * gridDim.y + blockIdx.z * gridDim.x * gridDim.y;

  // Calculate thread ID within the block
  unsigned thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x +
                                threadIdx.z * blockDim.x * blockDim.y;

  // Calculate global thread ID
  unsigned threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  unsigned global_thread_id = block_id * threads_per_block + thread_id_in_block;

  return global_thread_id;
}

// Simple kernel that assigns each thread a unique ID
// and stores it in the output buffer.
__global__ void multi_wave_kernel(unsigned *buf, bool sync_threads) {
  if (sync_threads) {
    wait_for_all_threads();
  }
  unsigned int thread_idx = get_global_idx();
  buf[thread_idx] = thread_idx; // GPU BREAKPOINT
}

int main() {
  const dim3 blocks(2, 2, 2);  // 3D grid specifying number of blocks to launch
  const dim3 threads(5, 4, 6); // 3D grid specifying number of threads to launch
  const size_t num_blocks = blocks.x * blocks.y * blocks.z;
  const size_t threads_per_block = threads.x * threads.y * threads.z;
  const size_t num_threads = num_blocks * threads_per_block;
  const size_t buffer_size = num_threads * sizeof(unsigned);

  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));
  unsigned wave_size = deviceProp.warpSize;
  printf("===================TEST CONFIGURATION===================\n");
  printf("Running on device:       %s\n", deviceProp.name);
  printf("Wave size:               %d\n", wave_size);
  printf("Total number of threads: %zu\n", num_threads);
  printf("Total number of blocks:  %zu\n", num_blocks);
  printf("Total number of waves:   %d\n", divideCeil(num_threads, wave_size));
  printf("Threads per block:       %zu\n", threads_per_block);
  printf("Waves per block:         %d\n",
         divideCeil(threads_per_block, wave_size));
  printf("Threads per wave:        %s\n",
         get_num_threads_in_wave(threads_per_block, wave_size).c_str());
  printf("===================TEST CONFIGURATION===================\n");
  printf("\n\n");

  // Allocate host vectors
  std::vector<unsigned> h_buf(num_threads,
                              std::numeric_limits<unsigned>::max());

  // Allocate device memory for the output data
  unsigned *d_buf;
  HIP_CHECK(hipMalloc(&d_buf, buffer_size));

  // Copy data from host to device
  printf("Copying data to device...\n");
  HIP_CHECK(hipMemcpy(d_buf, h_buf.data(), buffer_size, hipMemcpyHostToDevice));

  // Launch the kernel.
  printf("Launching multi-wave kernel with %zu threads...\n",
         num_threads);
  multi_wave_kernel<<<blocks, threads, 0, hipStreamDefault>>>(d_buf, true);

  // Copy data from device to host
  printf("Copying data to host...\n");   // CPU BREAKPOINT - AFTER LAUNCH
  HIP_CHECK(hipMemcpy(h_buf.data(), d_buf, buffer_size, hipMemcpyDeviceToHost));

  // Free device memory
  HIP_CHECK(hipFree(d_buf)); // CPU BREAKPOINT - AFTER FINISH

  // Print the output
  printf("Validating output...\n");
  for (size_t i = 0; i < num_threads; ++i) { 
    if (h_buf[i] != i) {
      std::cerr << "Error: Expected " << i << " but got " << h_buf[i] << std::endl;
      return error_exit_code;
    }
  }
  printf("Output matches expected values!\n");
  return 0;
}
