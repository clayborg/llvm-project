#include <cstdio>

// Defined in other.cu.
__device__ int twice(int value);
__global__ void add_one(int *data);

__global__ void parent(int *data) {
  data[threadIdx.x] = twice(threadIdx.x); // call to twice
  __syncthreads();
  // Launching a kernel from the device needs the device runtime library.
  if (threadIdx.x == 0)
    add_one<<<1, blockDim.x>>>(data);
}

int main() {
  constexpr int thread_count = 4;
  int *data;
  cudaMalloc(&data, thread_count * sizeof(int));
  cudaMemset(data, 0, thread_count * sizeof(int)); // breakpoint before launch
  parent<<<1, thread_count>>>(data);
  cudaDeviceSynchronize();
  int result[thread_count];
  cudaMemcpy(result, data, sizeof(result), cudaMemcpyDeviceToHost);
  cudaFree(data);
  for (int i = 0; i < thread_count; ++i) {
    if (result[i] != 2 * i + 1) {
      printf("result[%d] is %d, expected %d\n", i, result[i], 2 * i + 1);
      return 1;
    }
  }
  return 0;
}
