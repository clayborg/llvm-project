// A program for the exception tests. With no argument, a kernel faults and
// the host reports whether it was told so. With the argument "trap", one
// thread of a kernel runs __trap().
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void fault() {
  int *target = nullptr;
  target[threadIdx.x] = 1; // faulting store
}

__global__ void trap_in_one_thread() {
  if (threadIdx.x == 5)
    __trap(); // trap
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "trap") == 0) {
    trap_in_one_thread<<<1, 32>>>();
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 2;
  }

  fault<<<1, 32>>>();
  cudaError_t error = cudaDeviceSynchronize();
  printf("cudaDeviceSynchronize returned %s\n", cudaGetErrorName(error));
  // The kernel failed, so success here is the wrong answer.
  return error == cudaSuccess ? 1 : 0;
}
