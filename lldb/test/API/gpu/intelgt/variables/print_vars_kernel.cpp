// SYCL kernel for testing per-lane variable printing in IntelGT.
//
// Built at -O0 so local variables are spilled to per-lane scratch memory
// rather than kept in GRF registers. Each active SIMD lane must observe a
// DIFFERENT value for the per-lane variables and the SAME value for a uniform
// control variable.
//
// Variable kinds exercised per lane (modelled on GDB's gdb.sycl/simd-locations):
//   - scalar locals          : gid, doubled, konst
//   - accessor read into local: from_acc (= in_acc[gid])
//   - struct local + members : p (p.a, p.b)
//   - reference to a local    : ref (int& bound to gid)
//
// Launched over a full SIMD width worth of work-items so that multiple
// lanes of a single EU thread are active simultaneously at the breakpoint.

#include <sycl/sycl.hpp>

struct pair_s {
    int a;
    int b;
};

int main() {
    constexpr size_t N = 32;  // one SIMD32 EU thread worth of lanes
    sycl::queue q;
    int results[N] = {0};
    int inputs[N] = {0};
    for (size_t i = 0; i < N; ++i)
        inputs[i] = static_cast<int>(i);  // in_acc[gid] == gid

    sycl::buffer<int, 1> buf(results, sycl::range<1>(N));
    sycl::buffer<int, 1> ibuf(inputs, sycl::range<1>(N));

    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        auto in_acc = ibuf.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            int gid = static_cast<int>(id[0]);  // per-lane: 0,1,2,...,N-1
            int doubled = gid * 2;               // per-lane derived: 2*gid
            int konst = 7;                        // uniform across all lanes
            int from_acc = in_acc[gid];           // per-lane accessor read == gid
            pair_s p;
            p.a = gid;                            // per-lane struct member
            p.b = gid + 100;                      // per-lane struct member
            int &ref = gid;                       // reference bound to gid
            acc[id] = doubled + konst + from_acc + p.a + p.b + ref; // inside-kernel
        });
    });

    q.wait();
    return 0;
}
