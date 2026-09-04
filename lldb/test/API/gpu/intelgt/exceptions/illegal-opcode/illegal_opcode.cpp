#include <sycl/sycl.hpp>

// Simple kernel that will be stopped at a breakpoint.
// LLDB will then overwrite the instruction at PC with null bytes (illegal opcode).

int main() {
  sycl::queue q;
  int result = 0;
  sycl::buffer<int, 1> buf(&result, sycl::range<1>(1));

  q.submit([&](sycl::handler &h) {
    auto acc = buf.get_access<sycl::access::mode::write>(h);
    h.single_task([=]() {
      int x = 42;      // Breakpoint will be set here
      x = x + 1;       // After breakpoint, this will be overwritten
      acc[0] = x;
    });
  });

  q.wait();

  return 0;
}
