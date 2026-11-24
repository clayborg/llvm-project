#include <cstdio>

__device__ __noinline__ int triple(int value) {
  int tripled = value * 3;
  return tripled; // breakpoint in triple
}

struct Scaler {
  int factor;
  __device__ __noinline__ int scale(int value) const {
    int scaled = value * factor;
    return scaled; // breakpoint in scale
  }
};

__global__ void accumulate(int *totals) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = triple(index);
  // An array is in local memory, not in registers.
  int steps[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i)
    total += steps[i];
  totals[index] = total; // breakpoint in accumulate
}

__global__ void scale_all(int *scaled) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // The object is in the thread's local memory, and scale reads it through
  // `this`.
  Scaler scaler = {2};
  scaled[index] = scaler.scale(index);
}

__device__ __forceinline__ void keep_alive(int v) {
  asm volatile("" : : "r"(v) : "memory");
}

__device__ __forceinline__ void keep_alive(double v) {
  asm volatile("" : : "d"(v) : "memory");
}

__global__ void variables_kernel(int *results) {
  int reg_scalar = threadIdx.x + 0x1000;
  // A 64-bit variable is in two registers.
  double reg_double = threadIdx.x + 0.5;
  keep_alive(reg_scalar); // gpu breakpoint
  // Used after the breakpoint, so that they are readable there. The store
  // shows whether the kernel sees a value written at the breakpoint.
  results[threadIdx.x] = reg_scalar;
  keep_alive(reg_double);
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

  scale_all<<<blocks, threads>>>(totals);
  cudaDeviceSynchronize();
  cudaMemcpy(host, totals, sizeof(host), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i)
    if (host[i] != 2 * i)
      return 2;
  cudaFree(totals);

  int *results;
  cudaMalloc(&results, threads * sizeof(int));
  variables_kernel<<<1, threads>>>(results); // before kernel launch
  cudaDeviceSynchronize();

  // Lane 0, which the test writes, and lane 1, which it leaves alone.
  cudaMemcpy(host, results, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  printf("reg_scalar: %#x %#x\n", host[0], host[1]);
  cudaFree(results);
  return 0;
}
