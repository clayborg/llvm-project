// SYCL kernel for testing GPU disassembly. Each work-item does a little
// arithmetic so there are several instructions to decode at the breakpoint.
// Built at -O0.

#include <sycl/sycl.hpp>

int main() {
    constexpr size_t N = 32;
    sycl::queue q;
    int results[N] = {0};
    sycl::buffer<int, 1> buf(results, sycl::range<1>(N));

    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            int gid = static_cast<int>(id[0]);
            int v = gid * 3 + 1;
            acc[id] = v;                 // inside-kernel
        });
    });

    q.wait();
    return 0;
}
