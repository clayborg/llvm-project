// Source for comparison_artificial.cubin, the relocated cubin fixture embedded in
// the artificial NVGPU core built by TestNVGPUCoreFileComparison. The fixture
// is checked in so the test needs no nvcc / GPU / CUDA driver at run time --
// only cuda-gdb. Nested device functions give a non-trivial backtrace for
// later scope; for the current thread-list scope the cubin only needs to be a
// loadable, symbolicatable relocated image.
//
// Regenerate (offline, requires nvcc):
//   python3 regenerate_comparison_cubin.py --nvcc /path/to/nvcc
//
// nvcc emits an unrelocated cubin whose .text.* sections all start at vaddr 0.
// A real coredump embeds a *relocated* image (functions at their load VAs), so
// the fixture is post-processed to assign each allocatable section a distinct
// page-aligned load address based at 0x00007fffcf200000 and to rewrite the
// matching symbol values. The regeneration script performs and validates that
// transformation. Afterward, compare_kernel resolves at
// 0x00007fffcf203000 (the lane PC used by the test).

extern "C" __device__ __noinline__ int leaf(int x) { return x * 2 + 1; }

extern "C" __device__ __noinline__ int middle(int x) { return leaf(x) + x; }

extern "C" __global__ void compare_kernel(int *out) {
  int v = middle(threadIdx.x + blockIdx.x);
  out[threadIdx.x] = v;
}
