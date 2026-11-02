#include <cassert>
#include <sycl/sycl.hpp>

// Trigger software exception using assert(false).
// The SYCL runtime converts failed assertions to software exceptions
// that set CR0.1 bit 29 (software_exception_control).

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      assert(false && "software exception");
    });
  });

  q.wait();

  return 0;
}
