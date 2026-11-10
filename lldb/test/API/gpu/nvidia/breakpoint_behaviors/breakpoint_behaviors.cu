// A program for the breakpoint behavior tests. Each kernel keeps what the
// tests count deterministic.
//
// loop_kernel runs one thread, so a breakpoint in
// its loop is hit by that thread once per iteration; reduce runs one warp, so a
// breakpoint in its loop is hit by 32 threads at once, once per iteration.
#include <cstdio>

constexpr int kIterations = 8;
constexpr int kWarpSize = 32;

__device__ int square(int value) {
  return value * value; // breakpoint in square
}

// Inlined into both kernels, so a breakpoint on it has a location in each.
// Both kernels work on ints so that this one function, not an overload or a
// template, is the one inlined into each.
__device__ __forceinline__ int accumulate(int total, int value) {
  return total + value; // breakpoint in accumulate
}

__global__ void loop_kernel(int *result, int iterations) {
  int sum = 0;                           // first statement
  for (int i = 0; i < iterations; ++i) { // loop header
    sum += square(i);                    // breakpoint in loop
  }
  *result = accumulate(*result, sum); // breakpoint after loop
}

// Sums values[0..31] into every element with a butterfly: in each iteration a
// thread adds the value of its partner `width / 2` lanes away. Every thread
// runs every line, so a breakpoint in the loop stops the whole warp at once,
// once per iteration.
__global__ void reduce(int *values) {
  int x = threadIdx.x;
  for (int width = 2; width <= kWarpSize; width *= 2) { // reduce loop
    int partner = values[x ^ (width / 2)];              // partner read
    __syncthreads();
    values[x] = accumulate(values[x], partner); // reduce add
    __syncthreads();
  }
}

int main() {
  int *result = nullptr;
  int *values = nullptr;
  int initial[kWarpSize];
  for (int i = 0; i < kWarpSize; ++i)
    initial[i] = i;
  cudaMalloc(&result, sizeof(int));
  cudaMemset(result, 0, sizeof(int));
  cudaMalloc(&values, sizeof(initial));
  cudaMemcpy(values, initial, sizeof(initial), cudaMemcpyHostToDevice);

  int launches = 0;                           // breakpoint before launch
  loop_kernel<<<1, 1>>>(result, kIterations); // first launch
  ++launches;
  cudaDeviceSynchronize();
  int partial = 0;
  cudaMemcpy(&partial, result, sizeof(int), cudaMemcpyDeviceToHost);
  printf("sum after launch %d: %d\n", launches, partial); // between launches
  loop_kernel<<<1, 1>>>(result, kIterations);             // second launch
  ++launches;
  reduce<<<1, kWarpSize>>>(values); // reduce launch
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    printf("kernel failed: %s\n", cudaGetErrorString(error));
    return 2;
  }

  int sum = 0;
  int total = 0;
  cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&total, values, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(result);
  cudaFree(values);
  printf("sum after launch %d: %d, total %d\n", launches, sum, total);
  // Each launch adds 0*0 + 1*1 + ... + 7*7 = 140; the reduction sums 0..31
  // into every element.
  if (sum != 2 * 140)
    return 3;
  if (total != 496)
    return 4;
  return 0; // breakpoint before exit
}
