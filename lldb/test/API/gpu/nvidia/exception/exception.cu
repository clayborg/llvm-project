// A program for the exception tests: a kernel that faults, and a host that
// reports whether it was told so.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fault() {
  int *target = nullptr;
  target[threadIdx.x] = 1; // faulting store
}

int main() {
  fault<<<1, 32>>>();
  cudaError_t error = cudaDeviceSynchronize();
  printf("cudaDeviceSynchronize returned %s\n", cudaGetErrorName(error));
  // The kernel failed, so success here is the wrong answer.
  return error == cudaSuccess ? 1 : 0;
}
