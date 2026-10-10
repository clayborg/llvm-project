// Divergent 1-D SYCL parallel_for kernel: even and odd lanes take different
// branches of an if/else, so the EU thread's execution mask (CE) becomes
// half-masked inside each branch. Used to test SIMD lane divergence handling.
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
            int gid = static_cast<int>(id[0]);   // first-line
            int r;
            if (gid % 2 == 0)                      // branch-line
                r = gid + 100;                     // even-branch
            else
                r = gid + 200;                     // odd-branch
            acc[id] = r;                           // join-line
        });
    });

    q.wait();
    return 0;
}
