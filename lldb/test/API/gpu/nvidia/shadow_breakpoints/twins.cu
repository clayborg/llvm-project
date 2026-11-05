#include <cstdio>

// Host functions with the names and parameters of kernels in
// shadow_breakpoints.cu, which do not launch a kernel. The name of each pair
// says which of the two is static.
static void static_host_twin(int value) {
  printf("host static_host_twin %d\n", value); // static_host_twin body
}

void static_kernel_twin(int value) {
  printf("host static_kernel_twin %d\n", value); // static_kernel_twin body
}

void call_twins(int value) {
  static_host_twin(value);
  static_kernel_twin(value);
}
