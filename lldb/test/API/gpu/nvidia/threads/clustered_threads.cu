#include <cuda_runtime.h>

#ifndef CLUSTER_X
#define CLUSTER_X 2
#endif

__global__ void clusteredKernel() {
  int fault_here = *(volatile int *)0x03; // clustered kernel stop
  if (fault_here)
    asm volatile("");
}

int main() {
  cudaDeviceProp properties = {};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    return 1;

  volatile int device_major = properties.major;
  if (device_major < 9) // capability queried before cluster launch
    return 0;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(4, 1, 1);
  config.blockDim = dim3(1, 1, 1);

  cudaLaunchAttribute attribute = {};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = CLUSTER_X;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;

  if (cudaLaunchKernelEx(&config, clusteredKernel) != cudaSuccess)
    return 2;
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : 3;
}
