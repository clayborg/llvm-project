#include <hip/hip_runtime.h>
#include <iostream>

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

struct Point {
  int x;
  int y;
};

// Leaf function: breakpoint set here so the full call chain is on the stack.
// __noinline__ prevents the compiler from inlining the call chain.
__device__ __noinline__ void leaf_function(int tid, int *out,
                                           int scalar_arg,
                                           int *array_arg,
                                           Point *struct_arg) {
  // Compute values using the arguments so the compiler keeps them live.
  int leaf_local = tid * 100;
  out[tid] = leaf_local + scalar_arg + array_arg[0] + struct_arg->x; // GPU BREAKPOINT
}

// Middle function with its own local variables that should be visible
// when unwinding from the leaf frame.
__device__ __noinline__ void middle_function(int tid, int *out) {
  int middle_scalar = tid * 10 + 5;

  int middle_array[4];
  for (int i = 0; i < 4; i++) {
    middle_array[i] = tid + i + 100;
  }

  Point middle_struct;
  middle_struct.x = tid * 3;
  middle_struct.y = tid * 7;

  leaf_function(tid, out, middle_scalar, middle_array, &middle_struct);
}

// Kernel entry — the outermost frame in the GPU backtrace.
__global__ void stack_unwind_kernel(int *output, int num_elements) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  if (gid >= num_elements)
    return;

  middle_function(tid, output);
}

int main() {
  const int num_blocks = 1;
  const int threads_per_block = 32;
  const int num_elements = num_blocks * threads_per_block;

  int *h_output = new int[num_elements]; // CPU BREAKPOINT - BEFORE LAUNCH

  int *d_output;
  HIP_CHECK(hipMalloc(&d_output, num_elements * sizeof(int)));

  hipLaunchKernelGGL(stack_unwind_kernel, dim3(num_blocks),
                     dim3(threads_per_block), 0, 0, d_output, num_elements);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(
      hipMemcpy(h_output, d_output, num_elements * sizeof(int),
                hipMemcpyDeviceToHost));

  for (int i = 0; i < 4; i++) {
    std::cout << "output[" << i << "] = " << h_output[i] << std::endl;
  }

  HIP_CHECK(hipFree(d_output));
  delete[] h_output;

  return 0;
}
