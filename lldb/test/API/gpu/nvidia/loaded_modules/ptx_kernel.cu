// Built into PTX, which the driver compiles when main.cu loads it.
// extern "C" keeps the name that cuModuleGetFunction looks up.
extern "C" __global__ void ptx_kernel(int *data) {
  data[threadIdx.x] = threadIdx.x; // breakpoint in ptx_kernel
}
