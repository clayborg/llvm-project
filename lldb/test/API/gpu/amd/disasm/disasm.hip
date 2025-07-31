#include <hip/hip_runtime.h>
#include <iostream>

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the
/// program with an error code.
constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                              \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      std::exit(error_exit_code);                                         \
    }                                                                     \
  }

__global__ void asm_kernel() {
  // Create some assembly with a variety of instructions.
  // We test 32-bit and 64-bit encodings of alu instructions take from
  // https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
  //
  // The following instruction encodings all modify memory so we skip these to
  // avoid generating access violations.
  //  DS
  //  FLAT
  //  MTBUF
  //  MUBUF
  //  SMEM
  asm volatile(
  // SOP1: Scalar ALU, 1 operand
  "s_not_b32 s27, s28\n"

  // SOP2: Scalar ALU, 2 operands
  "s_add_u32 s23, s24, s25\n"
  "s_add_u32 s23, s24, 0x12345678\n"  // literal

  // SOPC: Scalar compare
  "s_cmp_eq_u32 s29, s30\n"

  // SOPK: Scalar ALU, immediate
  "s_movk_i32 s26, 0x1234\n"

  // SOPP: Scalar, immediate
  "s_nop 0x1\n"

  // VOP1: Vector ALU, 1 operands
  "v_accvgpr_mov_b32 a3, a5\n"

  // VOP2: Vector ALU, 2 operands
  "v_add_f32 v17, v18, v19\n"
  "v_add_f32 v17, v18, 0x3f800000\n"  // literal 1.0f

  // VOP3: Vector ALU, 3 operands
  "v_fma_f32 v20, v21, v22, v23\n"

  // VOP3P: Vector ALU, "packed"
  "v_accvgpr_read_b32 v17, a3\n"

  // VOPC: Vector compare
  "v_cmp_eq_f32 vcc, v24, v25\n"
  "v_cmp_eq_f32 vcc, v24, 0x3f800000\n"  // literal 1.0f
  :
  :
  : "a3", "a5",
    "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42",
    "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s34", "s35", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "vcc"
  );
  // GPU BREAKPOINT
}

int main() {
  // Launch the kernel.
  printf("Launching kernel...\n"); // CPU BREAKPOINT - BEFORE LAUNCH

  // Launch kernel.
  asm_kernel<<<
      dim3(1), // Number of blocks to launch
      dim3(1), // Number of threads to launch
      0, // number of bytes of additional shared memory to allocate
      hipStreamDefault // stream where the kernel should execute: default stream
      >>>();

  // Wait on all active streams on the current device.
  printf("Waiting for kernel to finish...\n");
  HIP_CHECK(hipDeviceSynchronize());

  printf("Done!\n");
}
