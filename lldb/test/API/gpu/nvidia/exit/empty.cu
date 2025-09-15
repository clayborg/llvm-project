#include <assert.h>
#include <cstdio>

__global__ void empty_kernel() {}

int main(int argc, char **argv) {
  cudaDeviceReset();
  const int N = 1024;
  // Launch kernel with 1024 threads (e.g., 128 blocks of 8 threads, or 32
  // blocks of 32 threads)
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // breakpoint1

  empty_kernel<<<blocksPerGrid, threadsPerBlock>>>();

  // We don't synchronize here to be able to test the case in which the GPU server is
  // doing some work and the CPU reports it has exited.

  if (argv[1][0] == '1') {
    printf("will exit with 1\n");
    return 1;
  }

  printf("will exit with 0\n");
  return 0;
}
