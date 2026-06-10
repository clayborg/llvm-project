#include <stdio.h>

__global__ void my_kernel(int value) {
    printf("GPU: %d\n", value); // gpu breakpoint
}

__global__ void my_other_kernel(int value) {
    printf("GPU other: %d\n", value); // second gpu breakpoint
}

int main(void) {
  cudaDeviceReset();

  my_kernel<<<1, 1>>>(42); // cpu breakpoint
  my_other_kernel<<<1, 1>>>(99);
  cudaDeviceSynchronize();
  return 0;
}
