#include <cstdio>

// One warp, so that each breakpoint is one stop.
constexpr int N = 32;

__global__ void store_indices(int *data) {
  data[threadIdx.x] = threadIdx.x; // gpu breakpoint first
  data[threadIdx.x] += 1;          // gpu breakpoint second
}

int main() {
  int *data;
  cudaMalloc(&data, N * sizeof(int));
  store_indices<<<1, N>>>(data); // cpu breakpoint before launch
  int host[N];
  cudaMemcpy(host, data, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(data);

  for (int i = 0; i < N; ++i)
    if (host[i] != i + 1) {
      printf("data[%d] is %d\n", i, host[i]);
      return 1;
    }
  return 0;
}
