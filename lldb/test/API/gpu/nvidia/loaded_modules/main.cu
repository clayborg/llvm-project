#include <cstdio>
#include <cstdlib>
#include <cuda.h>

static void check(CUresult result, const char *call) {
  if (result != CUDA_SUCCESS) {
    printf("%s failed with %d\n", call, result);
    exit(1);
  }
}

#define CHECK(call) check(call, #call)

// Load a module file from the working directory, where the test puts it, and
// run its kernel.
static void run(const char *file, const char *kernel) {
  CUmodule module;
  CHECK(cuModuleLoad(&module, file));
  CUfunction function;
  CHECK(cuModuleGetFunction(&function, module, kernel));
  constexpr int thread_count = 4;
  CUdeviceptr data;
  CHECK(cuMemAlloc(&data, thread_count * sizeof(int)));
  void *args[] = {&data};
  CHECK(cuLaunchKernel(function, 1, 1, 1, thread_count, 1, 1, 0, nullptr, args,
                       nullptr));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemFree(data));
}

int main() {
  CHECK(cuInit(0));
  CUdevice device;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));
  run("fatbin_kernel.fatbin", "fatbin_kernel"); // breakpoint before loading
  run("ptx_kernel.ptx", "ptx_kernel");
  CHECK(cuDevicePrimaryCtxRelease(device));
  return 0;
}
