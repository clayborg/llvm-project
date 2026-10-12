#include <cstdio>

void launchIntegrate(int *data);

// Defined before main so that the CPU breakpoint below, which LLDB copies to
// the GPU target, has no device code after it to move to.
__global__ void singleKernel(int *data) { data[2] = data[0] + data[1]; }

int main() {
  int *data;
  cudaMalloc(&data, 3 * sizeof(int));
  launchIntegrate(data);
  singleKernel<<<1, 1>>>(data);
  cudaDeviceSynchronize();
  cudaFree(data); // breakpoint after launches
  printf("done\n");
  return 0;
}
