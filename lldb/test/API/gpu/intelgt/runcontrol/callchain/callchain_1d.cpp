// parallel_for with a noinline call chain: multiple EU threads AND real frames
// so finish/next scheduler-locking can be tested across EU threads.
#include <sycl/sycl.hpp>

[[intel::noinline]] int leaf(int x) {
    return x * 2;            // leaf-line
}

[[intel::noinline]] int mid(int x) {
    return leaf(x + 1);      // mid-line
}

int main() {
    constexpr size_t N = 32;
    sycl::queue q;
    int results[N] = {0};
    sycl::buffer<int, 1> buf(results, sycl::range<1>(N));
    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            int gid = static_cast<int>(id[0]);
            acc[id] = mid(gid);    // call-line
        });
    });
    q.wait();
    return 0;
}
