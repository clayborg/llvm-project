// Device code that main.cu calls and launches. Without relocatable device
// code (-rdc=true), a source file's device code can only use its own.

__device__ int twice(int value) {
  return 2 * value; // breakpoint in twice
}

__global__ void add_one(int *data) { data[threadIdx.x] += 1; }
