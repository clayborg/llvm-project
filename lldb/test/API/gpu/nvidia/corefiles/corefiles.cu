#include <cassert>
#include <cstdint>
#include <cstdlib>

static const int B = 32;
static const int N = 4;

struct Point {
  int x;
  int y;
};

__device__ volatile int global_data[B][N];
__device__ __constant__ int constant_data[N] = {0x10000000, 0x10000001,
                                                0x10000002, 0x10000003};

// exercise frame/backtrace information
extern "C" __device__ __noinline__ void crash_kernel_2(int exception_type) {
  volatile int local_scratch = 0;
  int result;
  switch (exception_type) {
  case 1: // stop reason = trap
    /* *TRAP* */ asm volatile("trap;");
    break;
  case 2: // stop reason = CUDA Exception: Warp MMU Fault
    ((volatile int *)&global_data[0][0])[1 << 27] = 0;
    break;
  case 3: // stop reason = CUDA Exception: Warp Misaligned Address
    *(volatile int *)0x01 = 0;
    break;
  case 4: // stop reason = CUDA Exception: Warp Out-of-range Address
    asm volatile("st.shared.u32 [%0], %1;" : : "r"(1 << 20), "r"(0));
    break;
  case 5: // stop reason = CUDA Exception: Warp Misaligned PC
    ((void (*)())((uintptr_t)&crash_kernel_2 + 1))();
    break;
  case 6: // stop reason = CUDA Exception: Warp Invalid Address Space
    asm volatile("atom.global.add.u32 %0, [%1], 1;"
                 : "=r"(result)
                 : "l"(&local_scratch));
    break;
  }
}

extern "C" __device__ __noinline__ void crash_kernel_1(int exception_type) {
  volatile int middle_scalar = 0xcafebabe;
  volatile int middle_array[2] = { 0x10, 0x11 };
  volatile Point middle_point = { (int)blockIdx.x, (int)threadIdx.x };
  crash_kernel_2(exception_type);
}

extern "C" __device__ __noinline__ void crash_kernel_0(int exception_type) {
  // Avoid compiler bug: produces incorrect DWARF-CFI for CFA if this
  // frame is a thin forwarder. Add some bogus locals.
  // Details of bug: DTCLLDB-197
  volatile int make_this_frame_large[1] = {0};
  crash_kernel_1(exception_type);
}

extern "C" __global__ void crash_kernel(int exception_type) {
  // exercise global memory
  global_data[blockIdx.x][threadIdx.x] = blockIdx.x * N + threadIdx.x;

  // exercise shared memory
  volatile __shared__ int shared_data[N];
  shared_data[threadIdx.x] = 0xabcd0000 + threadIdx.x;

  __syncthreads();

  // exercise the exception-type-specific faulting CTA
  if (blockIdx.x == 13)
    crash_kernel_0(exception_type);

  while (clock64() != 0x7FFFFFFFFFFFFFFFLL)
    ;
}

int main(int argc, char **argv) {
  int exception_type = argc > 1 ? atoi(argv[1]) : 1;

  crash_kernel<<<B, N>>>(exception_type);
  cudaDeviceSynchronize();

  return 0;
}
