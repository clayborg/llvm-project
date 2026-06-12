#include <hip/hip_runtime.h>

static __global__ void second_code_object_kernel(int *value) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && *value == 1)
    __builtin_trap();
}

extern "C" void launch_second_code_object_kernel(int *value) {
  second_code_object_kernel<<<dim3(1), dim3(1)>>>(value);
}
