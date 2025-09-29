#include <assert.h>
#include <iostream>

__global__ void breakpoints(int value) {
  printf("%d\n", value);
  if (threadIdx.x == 0) {
    printf("will break %d\n", value); // gpu breakpoint
    int fault_here = *(volatile int *)0x03;

    if (fault_here)
      printf("Fault here\n");
  }
  __syncthreads();
}

int main(void) {
  const int N = 16;

  cudaDeviceReset();

  int threadsPerBlock = 4;
  int blocksPerGrid =
      (N + threadsPerBlock - 1) / threadsPerBlock; // cpu breakpoint
  breakpoints<<<blocksPerGrid, threadsPerBlock>>>(22);

  cudaDeviceSynchronize();

  return 0; // breakpoint before exit
}
