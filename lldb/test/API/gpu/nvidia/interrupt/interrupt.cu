#include <cstdio>

#include <cuda_runtime.h>

// Spins as long as *keep_running is set. Thread 0 counts the loop's
// iterations in *progress, so that the host can tell the kernel is running.
__global__ void spinKernel(volatile int *keep_running,
                           volatile unsigned long long *progress) {
  while (*keep_running) {
    if (threadIdx.x == 0) // breakpoint in the loop
      *progress = *progress + 1;
  }
}

int main() {
  // Mapped host memory, which the host reads while the kernel runs.
  volatile int *keep_running = nullptr;
  volatile unsigned long long *progress = nullptr;
  if (cudaHostAlloc((void **)&keep_running, sizeof(*keep_running),
                    cudaHostAllocMapped) != cudaSuccess ||
      cudaHostAlloc((void **)&progress, sizeof(*progress),
                    cudaHostAllocMapped) != cudaSuccess) {
    fprintf(stderr, "cudaHostAlloc failed\n");
    return 1;
  }
  *keep_running = 1;
  *progress = 0;

  int *device_keep_running = nullptr;
  unsigned long long *device_progress = nullptr;
  if (cudaHostGetDevicePointer((void **)&device_keep_running,
                               (void *)keep_running, 0) != cudaSuccess ||
      cudaHostGetDevicePointer((void **)&device_progress, (void *)progress,
                               0) != cudaSuccess) {
    fprintf(stderr, "cudaHostGetDevicePointer failed\n");
    return 1;
  }
  spinKernel<<<1, 32>>>(device_keep_running, device_progress);
  // Without the kernel, the wait below would never end.
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "spinKernel failed to launch\n");
    return 1;
  }

  while (*progress == 0) {
  }
  printf("kernel is running\n"); // kernel is running

  // Nothing clears *keep_running, so the host spins as long as the kernel
  // does: a test can resume the host and still find the kernel running, and
  // it ends the program itself.
  while (*keep_running) {
  }
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}
