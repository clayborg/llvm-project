#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256

__global__ void fault() {
  int fault_here = *(volatile int *)0x03;

  if (fault_here)
    printf("Fault here\n");
}

int main(void) {
  cudaDeviceReset();
  fault<<<20, N>>>(); // breakpoint1
  cudaDeviceSynchronize();

  return 0;
}
