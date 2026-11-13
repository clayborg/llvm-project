// Simple SYCL kernel for basic IntelGT debugging tests.
// Based on GDB's gdb.arch/sycl-simple.cpp

#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    int result = 0;

    std::cout << "SYCL: Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    {
        sycl::buffer<int> buf(&result, 1);
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(1, [=](sycl::id<1> id) {
                acc[0] = 42; // GPU BREAKPOINT
            });
        });
        int dummy = 0; // CPU BREAKPOINT - after submit, kernel may be running
    }

    std::cout << "Result: " << result << std::endl;

    return result == 42 ? 0 : 1;
}
