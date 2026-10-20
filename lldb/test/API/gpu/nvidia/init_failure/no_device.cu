#include <cstdio>

// The test hides every CUDA device. The first CUDA call initializes the
// driver, which is when lldb-server initializes the CUDA debugger API.
int main() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  printf("cudaGetDeviceCount: %s\n", cudaGetErrorName(err));
  return err == cudaErrorNoDevice ? 0 : 1;
}
