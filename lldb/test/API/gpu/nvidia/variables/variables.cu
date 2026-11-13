__device__ __noinline__ int triple(int value) {
  int tripled = value * 3;
  return tripled; // breakpoint in triple
}

__global__ void accumulate(int *totals) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = triple(index);
  // An array is in local memory, not in registers.
  int steps[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i)
    total += steps[i];
  totals[index] = total; // breakpoint in accumulate
}

int main() {
  const int blocks = 2, threads = 32, n = blocks * threads;
  int *totals;
  cudaMalloc(&totals, n * sizeof(int));
  accumulate<<<blocks, threads>>>(totals);
  cudaDeviceSynchronize();

  int host[n];
  cudaMemcpy(host, totals, sizeof(host), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i)
    if (host[i] != 3 * i + 10)
      return 1;
  return 0;
}
