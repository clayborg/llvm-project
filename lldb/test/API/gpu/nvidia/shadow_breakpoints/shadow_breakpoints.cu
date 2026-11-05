#include <cstdio>

__global__ void kernel(int value) {
  printf("kernel %d\n", value); // kernel body
}

extern "C" __global__ void c_kernel(int value) {
  printf("c_kernel %d\n", value); // c_kernel body
}

static __global__ void static_kernel(int value) {
  printf("static_kernel %d\n", value); // static_kernel body
}

namespace ns {
template <typename T> __global__ void template_kernel(T value) {
  printf("template_kernel %d\n", (int)value); // template_kernel body
}
} // namespace ns

// Kernels with the names and parameters of host functions in twins.cu. The
// name of each pair says which of the two is static.
__global__ void static_host_twin(int value) {
  printf("static_host_twin %d\n", value);
}

static __global__ void static_kernel_twin(int value) {
  printf("static_kernel_twin %d\n", value);
}

// Host code that a breakpoint on a kernel's name or on a line can resolve in,
// and that does not launch a kernel.
void kernel(float value) {
  printf("host kernel %f\n", value); // host kernel body
}

__host__ __device__ int both(int value) {
  return value + 1; // both body
}

void call_twins(int value);

int main() {
  int status = 0;
  kernel<<<1, 1>>>(1);
  cudaDeviceSynchronize(); // break after the first launch
  c_kernel<<<1, 1>>>(2);
  static_kernel<<<1, 1>>>(3);
  ns::template_kernel<<<1, 1>>>(4);
  static_host_twin<<<1, 1>>>(7);
  static_kernel_twin<<<1, 1>>>(8);
  cudaDeviceSynchronize();
  kernel(5.0f);
  status = both(6) == 7 ? 0 : 1;
  call_twins(9);
  return status; // break before exit
}
