#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

__global__ void lifecycle_kernel(int *out) {
  int lane = threadIdx.x;
  int value = lane;
  if (lane % 2 == 0)
    value += 100; // GPU BREAKPOINT
  out[blockIdx.x * blockDim.x + lane] = value;
}

// A test attaches to this kernel while it spins and releases it by setting
// g_release.
__device__ int g_release = 0;

__global__ void spin_kernel(int *started) {
  *started = 1;
  __threadfence_system();
  volatile int *release = &g_release;
  while (*release == 0)
    ;
}

// Write the file under a temporary name, then rename it, so that the file is
// complete once it exists.
static void write_file(const char *path, const char *text) {
  char tmp[4096];
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);
  FILE *file = fopen(tmp, "w");
  if (!file)
    return;
  fputs(text, file);
  fclose(file);
  rename(tmp, path);
}

// Write ready_path once the kernel spins, and result_path once it is released.
static int spin(const char *ready_path, const char *result_path) {
  int *started;
  if (hipHostMalloc(&started, sizeof(int), hipHostMallocCoherent) != hipSuccess)
    return 1;
  *started = 0;
  spin_kernel<<<1, 64>>>(started);
  while (!__atomic_load_n(started, __ATOMIC_ACQUIRE))
    usleep(1000);
  write_file(ready_path, "ready\n");
  if (hipDeviceSynchronize() != hipSuccess)
    return 2;
  write_file(result_path, "released\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 3 && strcmp(argv[1], "spin") == 0)
    return spin(argv[2], argv[3]);
  int *out;
  if (hipMalloc(&out, 128 * sizeof(int)) != hipSuccess)
    return 1;
  lifecycle_kernel<<<2, 64>>>(out);
  if (hipDeviceSynchronize() != hipSuccess)
    return 2;
  int host[2] = {0, 0};
  if (hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost) != hipSuccess)
    return 3;
  printf("out[0]=%d out[1]=%d\n", host[0], host[1]);
  return host[0] == 100 && host[1] == 1 ? 0 : 4;
}
