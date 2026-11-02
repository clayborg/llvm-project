#include <unistd.h>

#include <cuda_runtime.h>

__global__ void store(int *out) { *out = 42; }

// Initialize CUDA and run a kernel, then keep running until killed.
int main() {
  int *out = nullptr;
  if (cudaMalloc(&out, sizeof(int)) != cudaSuccess)
    return 1;
  store<<<1, 1>>>(out);
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 1;
  while (true)
    sleep(1);
}
