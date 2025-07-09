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

__global__ void assert_one(unsigned fail_block, unsigned fail_lane) {
  __attribute__((unused)) int val = 21;
  __attribute__((unused)) int array[N] = {
      9, 7, 5, 3, 1, 0, 2, 4, 9, 7, 5, 3, 1, 0, 2, 4, 9, 7, 5, 3, 1, 1234,
      2, 4, 9, 7, 5, 3, 1, 0, 2, 4, 9, 7, 5, 3, 1, 0, 2, 4, 9, 7, 5, 3,
      1, 0, 2, 4, 9, 7, 5, 3, 1, 0, 2, 4, 9, 7, 5, 3, 1, 0, 2, 4};

  if (threadIdx.x == fail_lane && blockIdx.x == fail_block)
    assert(fail_lane != threadIdx.x); // Assert here.
}

int main(void) {
  assert_one<<<20, N>>>(8, 21);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaGetLastError());

  return 0;
}