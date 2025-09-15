#include <assert.h>
#include <cstdio>

// Unwind without arguments
__device__ int level3() {
  if (threadIdx.x == 0) {
    int fault_here = *(volatile int *)0x03;

    if (fault_here)
      printf("Fault here\n");
  }
  return 3;
}

__device__ int level2() { return level3(); }

__device__ int level1() { return level2(); }

__device__ int level0() { return level1(); }

__global__ void unwind_no_arguments() { level0(); }

// Unwind with arguments
__device__ int level3_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  if (threadIdx.x == 0) {
    int fault_here = *(volatile int *)0x03;
  }
  return 3;
}

__device__ int level2_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  return level3_with_arguments(a, b, c, d, e, f, g, h);
}

__device__ int level1_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  return level2_with_arguments(a, b, c, d, e, f, g, h);
}

__device__ int level0_with_arguments(int a, int b, int c, int d, float e,
                                     float f, double g, double h) {
  return level1_with_arguments(a, b, c, d, e, f, g, h);
}

__global__ void unwind_with_arguments(int a, int b, int c, int d, float e,
                                      float f, double g, double h) {
  level0_with_arguments(a, b, c, d, e, f, g, h);
}

// Unwind with divergent control flow
__device__ int level3_with_divergent_control_flow() {
  if (threadIdx.x == 0) {
    int fault_here = *(volatile int *)0x03;
  }
  return 3;
}

__device__ int level2_with_divergent_control_flow() {
  return level3_with_divergent_control_flow();
}

__device__ int level1_with_divergent_control_flow() {
  return level2_with_divergent_control_flow();
}

__device__ int level0_with_divergent_control_flow() {
  return level1_with_divergent_control_flow();
}

__global__ void unwind_with_divergent_control_flow() {
  if (threadIdx.x == 0)
    level0_with_divergent_control_flow();
  else
    printf("no fault\n");
}

int main(void) {
  const int N = 1024;
  // Launch kernel with 1024 threads (e.g., 128 blocks of 8 threads, or 32
  // blocks of 32 threads)
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // unwind_no_arguments<<<blocksPerGrid, threadsPerBlock>>>();
  // unwind_with_arguments<<<blocksPerGrid, threadsPerBlock>>>(1, 2, 3, 4, 5.0f,
  //                                                          6.0f, 7.0, 8.0);
  unwind_with_divergent_control_flow<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
  printf("done\n");
  return 0;
}
