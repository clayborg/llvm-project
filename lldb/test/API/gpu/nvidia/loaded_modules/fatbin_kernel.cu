// Built into a fatbin, with a cubin for each GPU architecture, that main.cu
// loads.
extern "C" __global__ void fatbin_kernel(int *data) {
  data[threadIdx.x] = threadIdx.x; // breakpoint in fatbin_kernel
}
