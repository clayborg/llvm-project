// MIT License
//
// Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <cstring>
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

// Functions marked with __device__ are executed on the device and called from
// the device only.
__device__ unsigned int get_thread_idx() {
  // Built-in threadIdx returns the 3D coordinate of the active work item in the
  // block of threads.
  return threadIdx.x;
}

// Functions marked with __global__ are executed on the device and called from
// the host only.
__global__ void helloworld_kernel(char *str, size_t size) {
  unsigned int thread_idx = get_thread_idx();
  unsigned int block_idx = blockIdx.x;
  unsigned int idx = thread_idx + block_idx * blockDim.x;

  char c = 'x';
  switch (idx) {
    case  0: c = 'H'; break;
    case  1: c = 'e'; break;
    case  2: c = 'l'; break;
    case  3: c = 'l'; break;
    case  4: c = 'o'; break;
    case  5: c = ','; break;
    case  6: c = ' '; break;
    case  7: c = 'w'; break; // DIVERGENT BREAKPOINT
    case  8: c = 'o'; break;
    case  9: c = 'r'; break;
    case 10: c = 'l'; break;
    case 11: c = 'd'; break;
    case 12: c = '!'; break;
  }

  if (idx < size) { // GPU BREAKPOINT
    str[idx] = c;
  }
}

int main() {
  // Allocate host vectors
  const size_t size = strlen("Hello, world!");
  std::string h_str(size, '?');
  printf("%s\n", h_str.c_str()); // CPU BREAKPOINT - BEFORE LAUNCH

  // Allocate device memory for the output data
  char *d_str;
  HIP_CHECK(hipMalloc(&d_str, size));

  // Copy data from host to device
  printf("Copying data to device...\n");
  HIP_CHECK(hipMemcpy(d_str, h_str.data(), size, hipMemcpyHostToDevice));

  // Launch the kernel.
  printf("Launching kernel...\n");
  helloworld_kernel<<<
      dim3(1), // 3D grid specifying number of blocks to launch: (1, 1, 1)
      dim3(h_str.size()), // 3D grid specifying number of threads to launch: (strlen, 1, 1)
      0, // number of bytes of additional shared memory to allocate
      hipStreamDefault // stream where the kernel should execute: default stream
      >>>(d_str, size);

  // Copy data from device to host
  printf("Copying data to host...\n");   // CPU BREAKPOINT - AFTER LAUNCH
  HIP_CHECK(hipMemcpy(h_str.data(), d_str, h_str.size(), hipMemcpyDeviceToHost));

  // Free device memory
  HIP_CHECK(hipFree(d_str)); // CPU BREAKPOINT - AFTER FINISH

  // Print the output
  printf("%s\n", h_str.c_str());
}
