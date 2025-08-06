#include <cassert>
#include <cstdint>
#include <iostream>

__global__ void fill_with_indices(int32_t *array) {
  array[threadIdx.x] = threadIdx.x;
  if (threadIdx.x == 5) {
    int fault_here = *(volatile int *)0x03;

    if (fault_here)
      printf("Fault here\n");
  }
}

int main(void) {
  const int N = 1024;
  int32_t *d_arr;
  int32_t *h_arr = new int32_t[N];

  cudaMalloc(&d_arr, N * sizeof(int32_t));
  std::cout << "d_arr: " << d_arr << std::endl;

  int threadsPerBlock = 256;
  int blocksPerGrid =
      (N + threadsPerBlock - 1) / threadsPerBlock; // before kernel launch
  fill_with_indices<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

  cudaMemcpy(h_arr, d_arr, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i)
    std::cout << "h_arr[" << i << "] = " << h_arr[i] << std::endl;

  cudaFree(d_arr);
  delete[] h_arr;

  return 0;
}
