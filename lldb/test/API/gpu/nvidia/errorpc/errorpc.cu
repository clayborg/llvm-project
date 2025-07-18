#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Super simple assert test. Hits an assertion on device side

static void VerifyCudaError(cudaError_t err, const char *file, int line,
                            cudaError_t expected_err) {
  if (err != expected_err) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_CUDA_ERROR(err)                                                 \
  (VerifyCudaError(err, __FILE__, __LINE__, cudaSuccess))

#define N 256

__global__ void fault() {
  int fault_here = *(volatile int *)0x03;

  if (fault_here)
    printf("Fault here\n");
}

int main(void) {
  cudaDeviceReset();
  fault<<<20, N>>>(); // breakpoint1
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaGetLastError());

  return 0;
}
