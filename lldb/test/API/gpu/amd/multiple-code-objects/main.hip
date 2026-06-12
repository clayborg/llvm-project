#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

#define HIP_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl;     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

extern "C" void launch_first_code_object_kernel(int *value);
extern "C" void launch_second_code_object_kernel(int *value);

int main() {
  int *d_value = nullptr;
  HIP_CHECK(hipMalloc(&d_value, sizeof(int)));
  HIP_CHECK(hipMemset(d_value, 0, sizeof(int)));

  launch_first_code_object_kernel(d_value);
  HIP_CHECK(hipDeviceSynchronize());

  launch_second_code_object_kernel(d_value);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_value));
  return 0;
}
