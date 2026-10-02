#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <cuda_runtime.h>

// A resident kernel that spins until the host clears keep_running (which it
// never does; the kernel ends when the process exits). It sets *started to 1
// once it is actually executing on an SM, so the host can confirm the kernel
// is resident (rather than merely launched) before signalling readiness. This
// keeps GPU work executing so a debugger can attach to an already-running CUDA
// application and enumerate the threads of the in-flight kernel.
__global__ void spinKernel(volatile int *keep_running, volatile int *started) {
  *started = 1;
  while (*keep_running) {
    // Busy-wait so the warps stay resident on the SM.
  }
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(_err));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool WriteMarker(const char *path) {
  FILE *marker = fopen(path, "w");
  if (!marker) {
    fprintf(stderr, "failed to open marker '%s'\n", path);
    return false;
  }
  fputs("ready\n", marker);
  fclose(marker);
  return true;
}

int main(int argc, char **argv) {
  // Optional path to a readiness marker file. The test passes this and waits
  // for the file to appear before attaching, so the attach happens only after
  // a kernel is confirmed resident (exercising true late attach rather than the
  // cuInit initialization path).
  const char *ready_marker_path = (argc > 1) ? argv[1] : nullptr;
  // With a second path, report ready before touching CUDA instead, and wait for
  // that file to appear before initializing it, so a debugger can attach while
  // libcuda is not loaded yet.
  const char *go_marker_path = (argc > 2) ? argv[2] : nullptr;
  if (go_marker_path) {
    if (!WriteMarker(ready_marker_path))
      return 1;
    // The test creates the go marker once it has attached. The time limit
    // (about 600 s, as below) only stops an orphan from waiting forever if
    // that never happens.
    for (int i = 0; access(go_marker_path, F_OK) != 0; ++i) {
      if (i == 60000) {
        fprintf(stderr, "go marker '%s' never appeared\n", go_marker_path);
        return 1;
      }
      usleep(10000);
    }
  }

  int *d_keep_running = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_keep_running, sizeof(int)));

  int one = 1;
  CHECK_CUDA(
      cudaMemcpy(d_keep_running, &one, sizeof(int), cudaMemcpyHostToDevice));

  // Pinned, host-mapped flag the kernel sets once it is resident. Zero-copy
  // mapped memory is safe to poll from the host while the kernel runs.
  int *h_started = nullptr;
  CHECK_CUDA(
      cudaHostAlloc((void **)&h_started, sizeof(int), cudaHostAllocMapped));
  *h_started = 0;
  int *d_started = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer((void **)&d_started, h_started, 0));

  // Launch two warps worth of threads so the thread list is non-empty when a
  // debugger attaches.
  spinKernel<<<1, 64>>>(d_keep_running, d_started);
  // A launch failure is reported asynchronously via the next runtime call, but
  // cudaGetLastError surfaces immediate launch-configuration errors.
  CHECK_CUDA(cudaGetLastError());

  // Wait until the kernel is actually executing on the device.
  while (*((volatile int *)h_started) == 0) {
    usleep(1000);
  }

  // The kernel is confirmed resident. Write the readiness marker (if requested)
  // and also print a marker for humans, flushing so it is observable promptly.
  if (ready_marker_path && !go_marker_path && !WriteMarker(ready_marker_path))
    return 1;
  printf("CUDA_KERNEL_RESIDENT\n");
  fflush(stdout);

  // Keep the host process alive (and the kernel resident) so a debugger can
  // attach. The test kills this process during teardown; the time limit only
  // stops an orphan from occupying the GPU indefinitely if that never happens.
  for (int i = 0; i < 600; ++i)
    sleep(1);

  return 0;
}
