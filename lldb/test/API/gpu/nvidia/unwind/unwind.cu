#include <assert.h>
#include <cstdio>

// Breakpoint function for tests to break in
__device__ void breakpoint() {
  // Use a volatile variable to ensure the function isn't optimized away
  volatile int stop_here = 1;
  (void)stop_here; // gpu breakpoint, frame_breakpoint
}

// Unwind without arguments
__device__ int level3() {
  breakpoint(); // frame_level3
  return 3;
}

__device__ int level2() { return level3(); } // frame_level2

__device__ int level1() { return level2(); } // frame_level1

__device__ int level0() { return level1(); } // frame_level0

// Unwind with arguments
__device__ int level3_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  breakpoint(); // frame_level3_with_arguments
  return 3;
}

__device__ int level2_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  // Add local variables to change frame size
  int sum = a + b + c + d + (int)e + (int)f + (int)g + (int)h;
  // clang-format off
  int result = level3_with_arguments(a, b, c, d, e, f, g, h); // frame_level2_with_arguments
  // clang-format on
  return sum + result;
}

__device__ int level1_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  // Add local variables to change frame size
  int sum1 = a;
  int sum2 = b + c;
  int sum3 = d + (int)e + (int)f + (int)g + (int)h;
  int sum4 = sum1 + sum2 + sum3;
  // clang-format off
  int result = level2_with_arguments(a, b, c, d, e, f, g, h); // frame_level1_with_arguments
  // clang-format on
  return sum4 + result;
}

__device__ int level0_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  // clang-format off
  return level1_with_arguments(a, b, c, d, e, f, g, h); // frame_level0_with_arguments
  // clang-format on
}

// Unwind with divergent control flow
__device__ int level3_with_divergent_control_flow() {
  breakpoint(); // frame_level3_with_divergent_control_flow
  return 3;
}

__device__ int level2_with_divergent_control_flow() {
  return level3_with_divergent_control_flow(); // frame_level2_with_divergent_control_flow
}

__device__ int level1_with_divergent_control_flow() {
  return level2_with_divergent_control_flow(); // frame_level1_with_divergent_control_flow
}

__device__ int level0_with_divergent_control_flow() {
  return level1_with_divergent_control_flow(); // frame_level0_with_divergent_control_flow
}

// Null function pointer test
typedef void (*TestFunc)();

__device__ void call_function(const TestFunc &func) {
  func(); // frame_call_function
}

__device__ int level3_null_call() {
  if (threadIdx.x == 0) {
    TestFunc null_func = nullptr;
    call_function(null_func); // frame_level3_null_call
  }
  return 3;
}

__device__ int level2_null_call() {
  return level3_null_call(); // frame_level2_null_call
}

__device__ int level1_null_call() {
  return level2_null_call(); // frame_level1_null_call
}

__device__ int level0_null_call() {
  return level1_null_call(); // frame_level0_null_call
}

// Aggregator kernel that runs all test cases
__global__ void unwind_test_kernel() {
  // Test 1: No arguments
  level0(); // frame_kernel_test1

  // Test 2: With arguments
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  float e = 5.0f;
  float f = 6.0f;
  double g = 7.0;
  double h = 8.0;
  level0_with_arguments(a, b, c, d, e, f, g, h); // frame_kernel_test2

  // Test 3: Divergent control flow
  if (threadIdx.x == 0)
    level0_with_divergent_control_flow(); // frame_kernel_test3

  // Test 4: Null function pointer (causes fault, must be last)
  level0_null_call(); // frame_kernel_test4
}

int main(void) {
  // Initialize CUDA device to ensure GPU target is created for debugging
  cudaDeviceReset();

  const int N = 1024;
  int threadsPerBlock = 256;
  int blocksPerGrid =
      (N + threadsPerBlock - 1) / threadsPerBlock; // cpu breakpoint

  // Run single aggregator kernel that executes all test pathways
  unwind_test_kernel<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();

  return 0; // breakpoint before exit
}
