// SYCL kernel that triggers GPU page fault exception via invalid pointer write.
// Based on GDB's kernel-pagefault-write.cpp test.

#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    int data[1] {7};

    sycl::queue deviceQueue{sycl::gpu_selector_v};
    sycl::buffer<int, 1> buf{data, sycl::range<1>{1}};

    std::cout << "SYCL: Using device: " << deviceQueue.get_device().get_info<sycl::info::device::name>() << std::endl;

    deviceQueue.submit([&](sycl::handler& cgh) {
        auto numbers = buf.get_access<sycl::access::mode::read>(cgh);

        cgh.single_task<>([=]() {
            int *p = nullptr;
            int num = numbers[0];
            long long count = 100000000LL;
            *p = num;  // PAGEFAULT - write to nullptr
            while (count) count--;
        });
    });

    deviceQueue.wait();
    return 0;
}

