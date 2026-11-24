#include <stdio.h>

__global__ void register_write_stop() {
  // Keep R20/R21 allocated for the composite RA write test.
  printf("%u\n", threadIdx.x); // register write breakpoint
}

int main() {
  cudaDeviceReset();
  register_write_stop<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
