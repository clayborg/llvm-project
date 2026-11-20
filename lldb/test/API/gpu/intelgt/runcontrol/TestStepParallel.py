"""
Test that stepping is unaffected by the all-EU-thread gather on breakpoints.

The gather barrier only applies to breakpoint stops; single-step (trace) stops
must stay one-EU-thread-at-a-time. This test stops at the first kernel line,
steps a lane through the following source lines, and verifies the source line
advances as expected. Mirrors gdb.sycl/step-parallel-for.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtStepParallelTestCase(RunControlTestCaseBase):
    """Single-stepping a lane in a parallel_for kernel works after the fix."""

    def test_step_advances_source_line(self):
        """Stepping a stopped lane advances through the kernel body."""
        target, _ = self._create_target_with_breakpoint("first-line")
        self._launch(target)
        gpu_process = self._gpu_process()
        lanes = self._lane_threads(gpu_process)
        self.assertGreater(len(lanes), 0, "should have stopped lanes")

        # Pick the lane that reported the breakpoint as the one we step.
        thread = next((t for t in lanes
                       if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                      lanes[0])

        start_line = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertGreater(start_line, 0, "should have a valid start line")

        # Step over a couple of source lines; the line number must advance.
        thread.StepOver()
        self._wait_stopped(gpu_process)
        line_after_1 = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertGreater(
            line_after_1, start_line,
            f"source line should advance after step (was {start_line}, "
            f"now {line_after_1})")

        thread.StepOver()
        self._wait_stopped(gpu_process)
        line_after_2 = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertGreaterEqual(
            line_after_2, line_after_1,
            "source line should not go backwards on a second step")
