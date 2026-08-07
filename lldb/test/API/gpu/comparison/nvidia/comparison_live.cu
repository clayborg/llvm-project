// The kernel faults (trap) inside the nested chain
//   live_crash_0 -> live_crash_1 -> live_crash_2
// on a single CTA, so both debuggers default-select the same faulting thread.

extern "C" __device__ __noinline__ void live_crash_2() {
  /* *TRAP* */ asm volatile("trap;");
}

extern "C" __device__ __noinline__ void live_crash_1() {
  volatile int locals[2] = {0x10, 0x11};
  live_crash_2();
  (void)locals[0];
}

extern "C" __device__ __noinline__ void live_crash_0() {
  // Bogus local: a thin forwarder frame triggers a DWARF-CFI compiler bug
  // (DTCLLDB-197).
  volatile int pad[1] = {0};
  live_crash_1();
  (void)pad[0];
}

extern "C" __global__ void live_kernel() {
  if (blockIdx.x == 0)
    live_crash_0();
  while (clock64() != 0x7FFFFFFFFFFFFFFFLL)
    ;
}

int main() {
  live_kernel<<<4, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
