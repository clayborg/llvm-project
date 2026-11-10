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

  __syncthreads();
}

int main(void) {
  const int N = 1024;
  int32_t *d_arr;
  int32_t *h_arr = new int32_t[N];

  cudaMalloc(&d_arr, N * sizeof(int32_t));
  std::cout << "d_arr: " << d_arr << std::endl;

  // A buffer larger than one memory-read packet. Its pattern has a period of
  // 251 bytes, which no packet size is a multiple of, so a read that drops or
  // repeats a packet's worth of bytes does not match it.
  const size_t big_size = 1 << 20;
  uint8_t *d_big;
  uint8_t *h_big = new uint8_t[big_size];
  for (size_t i = 0; i < big_size; ++i)
    h_big[i] = i % 251;
  cudaMalloc(&d_big, big_size);
  cudaMemcpy(d_big, h_big, big_size, cudaMemcpyHostToDevice);
  delete[] h_big; // big buffer filled

  int threadsPerBlock = 256;
  int blocksPerGrid =
      (N + threadsPerBlock - 1) / threadsPerBlock; // before kernel launch
  fill_with_indices<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

  cudaMemcpy(h_arr, d_arr, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i)
    std::cout << "h_arr[" << i << "] = " << h_arr[i] << std::endl;

  cudaFree(d_arr);
  cudaFree(d_big);
  delete[] h_arr;

  return 0; // breakpoint before exit
}
