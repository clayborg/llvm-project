"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): stepping a
GPU thread over source lines over a barrier and around a loop, on LAUNCH_16.
simpleCooperativeGroups tests a step over plain lines. The expected behavior
is cuda-gdb's `next`. A step here can hang, and waits STEP_TIMEOUT seconds at
most (see matrixmul.py).
"""

from lldbsuite.test.decorators import expectedFailureAll

from matrixmul import THREAD, THREAD_CONDITION, MatrixMulTestCase, csub


class TestNVGPUSampleMatrixMulStepping(MatrixMulTestCase):
    @expectedFailureAll(
        "stepping over __syncthreads() runs the kernel on instead of stopping "
        "at the next line"
    )
    def test_step_over_barrier(self):
        """Stepping over the barrier after the loads, which every thread of
        the block has to reach, stops at the next line, the loop of
        multiply-adds."""
        _, thread = self.run_to_thread("// first barrier")
        self.step(thread, "// k loop")
        self.assertEqual(self.thread_coords(thread), THREAD)
        self.finish()

    @expectedFailureAll(
        "stepping over the loop header, which starts with a branch, needs a "
        "GPU instruction step, and the step never stops"
    )
    def test_step_through_multiply_add_loop(self):
        """Stepping over source lines moves a thread around the loop of
        multiply-adds: from the multiply-add to the loop's header and back,
        with k one more and Csub one product more each time around. nvcc does
        not unroll the loop in a debug build."""
        breakpoint, thread = self.run_to_thread(
            "// multiply-add", f"{THREAD_CONDITION} && k == 0"
        )
        breakpoint.SetEnabled(False)
        for k in range(1, 3):
            self.step(thread, "// k loop")
            self.step(thread, "// multiply-add")
            frame = thread.frame[0]
            self.assert_int_variable(frame, "k", k)
            self.assert_float_variable(frame, "Csub", csub(k))
        self.finish()
