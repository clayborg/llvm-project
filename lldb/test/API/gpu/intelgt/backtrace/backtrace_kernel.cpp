// SYCL kernel with a genuine call chain for backtrace testing.
// inner() calls are not inlined at -O0, producing real stack frames.

#include <sycl/sycl.hpp>

[[intel::noinline]] int inner(int x) {
    return x * 2;  // backtrace-inner
}

[[intel::noinline]] int middle(int x) {
    return inner(x + 1);  // backtrace-middle
}

int main() {
    sycl::queue q;
    int result = 0;
    sycl::buffer<int, 1> buf(&result, sycl::range<1>(1));

    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.single_task([=]() {
            acc[0] = middle(3);  // backtrace-top
        });
    });

    q.wait();
    return 0;
}
