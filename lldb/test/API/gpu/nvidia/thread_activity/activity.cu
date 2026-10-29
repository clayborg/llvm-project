#include <cuda.h>
#include <cuda_runtime.h>

__global__ void activity(unsigned *output) {
  if ((threadIdx.x & 1) != 0)
    output[threadIdx.x] = threadIdx.x; // divergent store
  __syncthreads();
}

int main() {
  unsigned *output = nullptr;
  if (cudaMalloc(&output, 32 * sizeof(*output)) != cudaSuccess)
    return 1;
  activity<<<1, 32>>>(output); // before kernel launch
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 1;
  // The test stops the GPU at this module load.
  CUmodule module;
  if (cuModuleLoad(&module, "loaded.fatbin") != CUDA_SUCCESS)
    return 1;
  return cudaFree(output) == cudaSuccess ? 0 : 1;
}
