#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void lifecycle_kernel(int *out) {
  int lane = threadIdx.x;
  int value = lane;
  if (lane % 2 == 0)
    value += 100; // GPU BREAKPOINT
  out[blockIdx.x * blockDim.x + lane] = value;
}

int main() {
  int *out;
  if (hipMalloc(&out, 128 * sizeof(int)) != hipSuccess)
    return 1;
  lifecycle_kernel<<<2, 64>>>(out);
  if (hipDeviceSynchronize() != hipSuccess)
    return 2;
  int host[2] = {0, 0};
  if (hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost) != hipSuccess)
    return 3;
  printf("out[0]=%d out[1]=%d\n", host[0], host[1]);
  return host[0] == 100 && host[1] == 1 ? 0 : 4;
}
