"""
Test a program built with relocatable device code (-rdc=true) from two source
files.
"""

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

# The threads of parent's one block, thread_count in main.cu.
THREADS = 4


class TestNVGPURDC(NVGPUTestCaseBase):
    def test_call_and_launch_across_source_files(self):
        """Breakpoints in both source files stop the kernel's threads in turn:
        at a line of the kernel in main.cu, in the device function it calls
        in other.cu, with the backtrace going from other.cu into main.cu,
        and in the kernel that parent launches from the device. Each counts
        a hit per thread, and the program checks the kernels' results."""
        self.build()
        self.run_to_cpu_line("main.cu", "// breakpoint before launch")
        gpu_target = self.gpu_target
        in_parent = gpu_target.BreakpointCreateByLocation(
            "main.cu", line_number("main.cu", "// call to twice")
        )
        in_twice = gpu_target.BreakpointCreateByName("twice")
        in_add_one = gpu_target.BreakpointCreateByName("add_one")

        self.continue_cpu_and_wait_for_gpu_to_stop()
        threads = self.assert_gpu_stop_at_breakpoint(in_parent)
        self.assertEqual(len(threads), THREADS, "threads at the call in main.cu")
        self.assert_frame_at(
            threads[0].frame[0], "parent(int*)", "main.cu", "// call to twice"
        )

        threads = self.continue_gpu_until_breakpoint(in_twice)
        self.assertEqual(len(threads), THREADS, "threads in twice")
        self.assert_frame_at(
            threads[0].frame[0], "twice(int)", "other.cu", "// breakpoint in twice"
        )
        self.assert_frame_at(
            threads[0].frame[1], "parent(int*)", "main.cu", "// call to twice"
        )

        threads = self.continue_gpu_until_breakpoint(in_add_one)
        self.assertEqual(len(threads), THREADS, "threads in add_one")
        self.assert_frame_at(
            threads[0].frame[0], "add_one(int*)", "other.cu", "// breakpoint in add_one"
        )

        self.assertEqual(
            [in_parent.GetHitCount(), in_twice.GetHitCount(), in_add_one.GetHitCount()],
            [THREADS] * 3,
            "hit counts",
        )
        # The program checks the kernels' results and exits with 1 if the
        # kernel that parent launched did not run. Both processes report the
        # program's exit status.
        self.continue_gpu_to_exit(keep_breakpoints=True)
