#include <stdio.h>

__global__ void my_kernel(int value) {
    printf("GPU: %d\n", value); // gpu breakpoint
}

int main(void) {
    // cpu breakpoint
    my_kernel<<<1, 1>>>(42);
    cudaDeviceSynchronize();
    return 0;
}
