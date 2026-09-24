// 1-D SYCL parallel_for kernel: each work-item writes its global id.
//
// Launched over enough work-items that the hardware splits the range across
// more than one EU hardware thread, so a breakpoint inside the kernel is hit
// by multiple EU threads. Built at -O0 so each work-item's id lives in
// per-lane memory.

#include <sycl/sycl.hpp>

int main() {
    constexpr size_t N = 32;
    sycl::queue q;
    int results[N] = {0};
    sycl::buffer<int, 1> buf(results, sycl::range<1>(N));

    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            int gid = static_cast<int>(id[0]);  // first-line: global id
            int doubled = gid * 2;               // step-line-1
            int plus_one = doubled + 1;          // step-line-2
            acc[id] = plus_one;                  // inside-kernel (store)
        });
    });

    q.wait();
    return 0;
}
