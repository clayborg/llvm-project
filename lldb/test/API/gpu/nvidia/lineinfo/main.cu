__global__ void square(int *data);

int main() {
  constexpr int thread_count = 4;
  int *data;
  cudaMalloc(&data, thread_count * sizeof(int));
  square<<<1, thread_count>>>(data); // breakpoint before launch
  cudaDeviceSynchronize();
  cudaFree(data);
  return 0;
}

// Defined after main: where a test program defines its kernels does not matter
// to the breakpoint run_to_cpu_line sets above.
__global__ void square(int *data) {
  data[threadIdx.x] = threadIdx.x * threadIdx.x;
}
