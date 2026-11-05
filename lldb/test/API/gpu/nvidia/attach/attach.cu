#include "attach.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

#include <cuda_runtime.h>

// Spins until the program exits, since nothing clears *keep_running. Thread 0
// counts the iterations in *progress, which tells the host that the kernel is
// running.
__global__ void spinKernel(volatile int *keep_running,
                           volatile unsigned long long *progress) {
  while (*keep_running) {
    if (threadIdx.x == 0)
      *progress = *progress + 1; // breakpoint in the loop
  }
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err));      \
      return false;                                                            \
    }                                                                          \
  } while (0)

static bool CreateFile(const std::string &path) {
  FILE *file = fopen(path.c_str(), "w");
  if (!file) {
    fprintf(stderr, "cannot create '%s'\n", path.c_str());
    return false;
  }
  fclose(file);
  return true;
}

static bool WaitForFile(const std::string &path) {
  for (int i = 0; access(path.c_str(), F_OK) != 0; ++i) {
    if (i == 60000)
      return false;
    usleep(10000);
  }
  return true;
}

// Launch the kernel, and return once it is running.
static bool LaunchKernel() {
  int *keep_running = nullptr;
  CHECK_CUDA(cudaMalloc(&keep_running, sizeof(int)));
  const int one = 1;
  CHECK_CUDA(
      cudaMemcpy(keep_running, &one, sizeof(int), cudaMemcpyHostToDevice));

  // Mapped host memory, which the host can read while the kernel runs.
  unsigned long long *progress = nullptr;
  CHECK_CUDA(cudaHostAlloc(&progress, sizeof(*progress), cudaHostAllocMapped));
  *progress = 0;
  unsigned long long *device_progress = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(&device_progress, progress, 0));

  // Two warps.
  spinKernel<<<1, 64>>>(keep_running, device_progress);
  CHECK_CUDA(cudaGetLastError());
  while (*static_cast<volatile unsigned long long *>(progress) == 0)
    usleep(1000);
  return true;
}

// Call cuInit, as a library that makes sure that CUDA is initialized does.
static bool InitializeCUDAAgain() {
  void *libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  auto cu_init =
      libcuda ? reinterpret_cast<int (*)(unsigned)>(dlsym(libcuda, "cuInit"))
              : nullptr;
  if (!cu_init || cu_init(0) != 0) {
    fprintf(stderr, "cuInit failed\n");
    return false;
  }
  return true;
}

// Usage: a.out kernel|before-init|after-init|init-again|init-after-kernel DIR
//
// Create DIR/ready, so that a debugger can attach: in kernel, once the kernel
// runs; in before-init, before the program initializes CUDA; and in
// after-init, init-again and init-after-kernel, once it has initialized CUDA
// but has no CUDA context yet. In all but kernel, launch the kernel once
// DIR/go exists, after calling cuInit again in init-again. Create DIR/running
// once the kernel runs, then call cuInit again in init-after-kernel.
//
// The test kills the program. The time limits only stop a program that it
// leaves behind from running forever.
int main(int argc, char **argv) {
  lldb_enable_attach();
  if (argc != 3)
    return 1;
  const char *mode = argv[1];
  const std::string dir = argv[2];
  const bool waits = strcmp(mode, "kernel") != 0;
  const bool init_again = strcmp(mode, "init-again") == 0;
  const bool init_after_kernel = strcmp(mode, "init-after-kernel") == 0;

  if (waits) {
    // Querying the devices initializes CUDA without creating a context.
    int device_count = 0;
    if ((strcmp(mode, "after-init") == 0 || init_again || init_after_kernel) &&
        cudaGetDeviceCount(&device_count) != cudaSuccess)
      return 1;
    if (!CreateFile(dir + "/ready") || !WaitForFile(dir + "/go"))
      return 1;
    if (init_again && !InitializeCUDAAgain())
      return 1;
  }
  if (!LaunchKernel() || !CreateFile(dir + "/running"))
    return 1;
  if (init_after_kernel && !InitializeCUDAAgain())
    return 1;
  if (!waits && !CreateFile(dir + "/ready"))
    return 1;

  sleep(600);
  return 0;
}
