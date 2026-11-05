"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): breakpoint
conditions that pick one of the kernel's threads, on its locals, on its
parameters and on the CUDA built-in variables.

lldb evaluates a condition in each thread that stops at the breakpoint, and
reports an error for each thread in which the condition fails. The tests
collect the errors, which would otherwise fill their output, and check that
there are none.
"""

from lldbsuite.test.decorators import expectedFailureAll

from matrixmul import (
    LAUNCH_16,
    LAUNCH_32,
    THREAD,
    THREAD_CONDITION,
    MatrixMulTestCase,
)


class TestNVGPUSampleMatrixMulConditions(MatrixMulTestCase):
    def test_conditional_breakpoint_on_locals(self):
        """A breakpoint with a condition on the kernel's locals that picks one
        thread of 4,096 stops only that thread: once in each iteration of the
        loop over tiles, and next in the kernel's next launch."""
        errors = self.collect_errors()
        breakpoint = self.run_to_gpu_marker(
            "// gpu breakpoint at tile load", THREAD_CONDITION
        )
        stopped = self.assert_gpu_stop_at_breakpoint(breakpoint)
        for stop in range(LAUNCH_32.tiles + 1):
            if stop:
                stopped = self.continue_gpu_until_breakpoint(breakpoint)
            self.assertEqual(
                errors(), [], f"errors evaluating the condition, stop {stop}"
            )
            self.assert_stopped_threads(stopped, THREAD, f"stop {stop}")
            # The last stop is in the first tile of the next launch.
            tile = stop % LAUNCH_32.tiles
            expected = LAUNCH_32.tile_locals(THREAD, tile)["a"]
            self.assert_int_variable(stopped[0].frame[0], "a", expected)
        self.assertEqual(breakpoint.GetHitCount(), LAUNCH_32.tiles + 1)
        self.finish()

    @expectedFailureAll(
        "a breakpoint condition cannot read a kernel parameter, which is in "
        "parameter memory"
    )
    def test_conditional_breakpoint_on_kernel_parameter(self):
        """A breakpoint condition can read the kernel's parameters: with a
        condition on wA, the width of A, which holds in every thread, and on
        the locals that pick one thread, only that thread stops."""
        self.run_to_thread(
            "// gpu breakpoint at tile load",
            f"wA == {LAUNCH_16.width} && {THREAD_CONDITION}",
        )
        self.finish()

    @expectedFailureAll(
        "breakpoint conditions do not know the CUDA built-in variables "
        "threadIdx and blockIdx"
    )
    def test_conditional_breakpoint_on_cuda_builtins(self):
        """A breakpoint condition can use the CUDA built-in variables
        blockIdx and threadIdx to pick one thread, which then is the only one
        to stop."""
        (bx, by, _), (tx, ty, _) = THREAD
        self.run_to_thread(
            "// gpu breakpoint at tile load",
            f"blockIdx.x == {bx} && blockIdx.y == {by} && "
            f"threadIdx.x == {tx} && threadIdx.y == {ty}",
        )
        self.finish()
