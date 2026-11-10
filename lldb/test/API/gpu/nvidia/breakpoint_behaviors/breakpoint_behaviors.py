"""What the breakpoint behavior tests share: what breakpoint_behaviors.cu
does, and the steps that depend on it.

The program runs loop_kernel twice with one thread, whose loop has ITERATIONS
iterations, then reduce with one warp, whose loop runs once per value in
WIDTHS. It checks its own results and exits 0.
"""

from typing import Callable, Tuple

import lldb
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "breakpoint_behaviors.cu"
EXE = "breakpoint_behaviors"
ITERATIONS = 8
LAUNCHES = 2
WARP = 32
# The values of `width` in reduce's loop, one iteration each.
WIDTHS = [2, 4, 8, 16, 32]
# loop_kernel adds the squares of its loop counter to its result in each
# launch; reduce sums the thread indices into every element.
LOOP_SUM = sum(i * i for i in range(ITERATIONS))
REDUCE_TOTAL = sum(range(WARP))
# What the program prints before it exits 0.
FINAL_OUTPUT = (
    f"sum after launch {LAUNCHES}: {LAUNCHES * LOOP_SUM}, total {REDUCE_TOTAL}"
)


class BreakpointBehaviorsTestCase(NVGPUTestCaseBase):
    """A test of breakpoint_behaviors.cu. The test files derive from this."""

    def start(self) -> None:
        """Run the program to the CPU line before the first launch, where the
        GPU target exists."""
        self.run_to_cpu_line(
            SOURCE, "// breakpoint before launch", exe_name=self.build_shared(EXE)
        )

    def run_to_loop(
        self, ignore_count: int = 0
    ) -> Tuple[lldb.SBBreakpoint, lldb.SBThread]:
        """Run to the first stop at the line marked "breakpoint in loop", with
        `ignore_count` on the breakpoint, and return the breakpoint and
        loop_kernel's one thread."""
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// breakpoint in loop")
        if ignore_count:
            breakpoint.SetIgnoreCount(ignore_count)
        self.continue_cpu_and_wait_for_gpu_to_stop()
        return breakpoint, self.one_thread_at(breakpoint)

    def assert_iteration(self, thread: lldb.SBThread, iteration: int) -> None:
        """`thread` must be at the loop's breakpoint line in `iteration`."""
        self.assert_frame_at(
            thread.frame[0], "loop_kernel(int*, int)", SOURCE, "// breakpoint in loop"
        )
        self.assert_int_variable(thread.frame[0], "i", iteration)

    def width_check(self, width: int) -> Callable[[lldb.SBThread], None]:
        """Return a check for for_each_thread_at_breakpoint: the thread is in
        reduce's iteration with `width`."""

        def check(thread: lldb.SBThread) -> None:
            self.assert_int_variable(thread.frame[0], "width", width)

        return check

    def finish(self) -> None:
        """Let the program run to its end with the breakpoints as they are:
        no stop is expected, both processes exit 0, and the program printed
        the results its checks passed."""
        self.continue_gpu_to_exit(keep_breakpoints=True)
        self.assert_stdout_contains(FINAL_OUTPUT)
