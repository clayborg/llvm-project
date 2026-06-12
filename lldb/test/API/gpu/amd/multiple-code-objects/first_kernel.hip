#include <hip/hip_runtime.h>

static __global__ void first_code_object_kernel(int *value) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *value = 1;
}

extern "C" void launch_first_code_object_kernel(int *value) {
  first_code_object_kernel<<<dim3(1), dim3(1)>>>(value);
}
