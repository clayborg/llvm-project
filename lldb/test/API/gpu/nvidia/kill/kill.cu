// A program for the kill tests: a kernel of one thread with a loop to stop
// in. The tests end the program at that stop, so it checks nothing itself.
#include <cuda_runtime.h>

__global__ void count(int *result, int iterations) {
  int sum = 0;
  for (int i = 0; i < iterations; ++i)
    sum += i; // breakpoint in kernel
  *result = sum;
}

int main() {
  int *result = nullptr;
  cudaMalloc(&result, sizeof(int));
  count<<<1, 1>>>(result, 4);
  cudaDeviceSynchronize();
  cudaFree(result);
  return 0;
}
