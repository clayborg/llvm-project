#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include <cuda_runtime.h>

// A resident kernel that spins until the host clears keep_running (which it
// never does; the kernel ends when the process exits). Thread 0 counts the
// loop's iterations in *progress, which the host reads to confirm the kernel
// is resident (rather than merely launched) before signalling readiness, and
// reports so a test can tell that the kernel keeps running. This keeps GPU
// work executing so a debugger can attach to an already-running CUDA
// application and enumerate the threads of the in-flight kernel.
__global__ void spinKernel(volatile int *keep_running,
                           volatile unsigned long long *progress) {
  while (*keep_running) {
    if (threadIdx.x == 0)
      *progress = *progress + 1;
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

// Replace the report at path with how many times the host has woken up, the
// kernel's iteration count and that count's address. The report is written
// aside and renamed into place, so a reader never sees half of one.
static bool WriteProgress(const std::string &path, unsigned long host_ticks,
                          volatile unsigned long long *gpu_progress) {
  std::string tmp_path = path + ".tmp";
  FILE *report = fopen(tmp_path.c_str(), "w");
  if (!report) {
    fprintf(stderr, "failed to open progress report '%s'\n", tmp_path.c_str());
    return false;
  }
  fprintf(report, "%lu %llu %p\n", host_ticks, *gpu_progress,
          (void *)gpu_progress);
  fclose(report);
  return rename(tmp_path.c_str(), path.c_str()) == 0;
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

  // Pinned, host-mapped counter the kernel advances while it runs. Zero-copy
  // mapped memory is safe to poll from the host while the kernel runs.
  unsigned long long *h_progress = nullptr;
  CHECK_CUDA(cudaHostAlloc((void **)&h_progress, sizeof(*h_progress),
                           cudaHostAllocMapped));
  *h_progress = 0;
  unsigned long long *d_progress = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer((void **)&d_progress, h_progress, 0));
  volatile unsigned long long *gpu_progress = h_progress;

  // Launch two warps worth of threads so the thread list is non-empty when a
  // debugger attaches.
  spinKernel<<<1, 64>>>(d_keep_running, d_progress);
  // A launch failure is reported asynchronously via the next runtime call, but
  // cudaGetLastError surfaces immediate launch-configuration errors.
  CHECK_CUDA(cudaGetLastError());

  // Wait until the kernel is actually executing on the device.
  while (*gpu_progress == 0) {
    usleep(1000);
  }

  // The progress report sits next to the readiness marker. It is written once
  // before the marker below, so a test that waits for the marker never reads a
  // report left by an earlier run.
  std::string progress_path;
  if (ready_marker_path) {
    progress_path = std::string(ready_marker_path) + ".progress";
    if (!WriteProgress(progress_path, 0, gpu_progress))
      return 1;
  }

  // The kernel is confirmed resident. Write the readiness marker (if requested)
  // and also print a marker for humans, flushing so it is observable promptly.
  if (ready_marker_path && !go_marker_path && !WriteMarker(ready_marker_path))
    return 1;
  printf("CUDA_KERNEL_RESIDENT\n");
  fflush(stdout);

  // Keep the host process alive (and the kernel resident) so a debugger can
  // attach, and report progress every 100 ms. The test kills this process
  // during teardown; the time limit only stops an orphan from occupying the GPU
  // indefinitely if that never happens.
  for (unsigned long tick = 1; tick <= 6000; ++tick) {
    usleep(100000);
    if (!progress_path.empty())
      WriteProgress(progress_path, tick, gpu_progress);
  }

  return 0;
}
