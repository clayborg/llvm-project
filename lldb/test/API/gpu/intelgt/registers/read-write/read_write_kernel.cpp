// SYCL kernel for testing register read/write operations via LLDB.

#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};

    std::cout << "SYCL: Using device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    sycl::buffer<int, 1> buf{sycl::range<1>{1}};

    q.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);

        h.single_task([=]() {
            int value = 42;  // read-write-breakpoint
            acc[0] = value;
        });
    });

    q.wait();
    return 0;
}
