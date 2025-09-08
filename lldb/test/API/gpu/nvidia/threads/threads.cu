#include <stdio.h>

__global__ void setAndCopyKernel(int *arr1, int *arr2) {
  int tid = threadIdx.x;

  arr1[tid] = tid;

  // Synchronize all 512 threads.
  __syncthreads();

  if (tid == 5) {
    int fault_here = *(volatile int *)0x03;
    if (fault_here)
      printf("Fault here\n");
  }
  __syncthreads();

  arr2[tid] = arr1[tid];
}

int main() {
  const int N = 512;
  size_t size = N * sizeof(int);

  int *d_arr1, *d_arr2;
  int h_arr2[N];

  // Allocate device memory.
  cudaMalloc((void **)&d_arr1, size);
  cudaMalloc((void **)&d_arr2, size);

  // Launch kernel with 512 threads in 1 block.
  setAndCopyKernel<<<1, N>>>(d_arr1, d_arr2); // before kernel launch
  cudaDeviceSynchronize();

  cudaMemcpy(h_arr2, d_arr2, size, cudaMemcpyDeviceToHost);

  cudaFree(d_arr1);
  cudaFree(d_arr2);

  return 0;
}
