"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): the CPU
process's side of debugging the kernel. Breakpoints on the kernel set before
the run should stop the GPU and never the CPU, a synchronous launch should
show the GPU's stop, and the host reads the result the kernels computed.
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number

from matrixmul import (
    LAUNCH_16,
    LAUNCH_32,
    SOURCE,
    THREAD,
    THREAD_CONDITION,
    MatrixMulTestCase,
    csub,
)


class TestNVGPUSampleMatrixMulHost(MatrixMulTestCase):
    def test_conditional_kernel_line_breakpoint_does_not_stop_cpu(self):
        """A breakpoint on a kernel line with a condition on the kernel's
        locals, set before the program runs, stops the GPU thread the
        condition picks, and never the CPU."""
        target = self.make_target(LAUNCH_16)
        breakpoint = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// gpu breakpoint at tile load")
        )
        breakpoint.SetCondition(THREAD_CONDITION)
        self.run_to_launch(target)
        gpu_breakpoint = self.gpu_breakpoint_copy(breakpoint)
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_stopped_threads(
            self.assert_gpu_stop_at_breakpoint(gpu_breakpoint), THREAD, "the breakpoint"
        )

        self.gpu_target.BreakpointDelete(gpu_breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        # Apart from target-sync stops, nothing should stop the CPU before the
        # program exits.
        self.assertEqual(self.wait_for_exit(self.cpu_process), 0, "CPU exit")
        self.assertEqual(self.wait_for_exit(self.gpu_process), 0, "GPU exit")

    @expectedFailureAll(
        "a frame read from GPU thread 1 before the GPU process takes its "
        "first stop stays, with PC 0, after the stop"
    )
    def test_frame_read_before_gpu_stop_is_refreshed(self):
        """In synchronous mode a launch returns at the CPU's target-sync stop,
        before the GPU process has taken its stop, and the GPU process shows
        one thread with no PC yet. A frame read from that thread then, as the
        CLI does to show where it stopped, is gone once the GPU stop is
        applied: thread 1's frame is where its PC register is, in the
        kernel."""
        if not self.cpu_stops_with_gpu():
            self.skipTest(
                "the CPU does not stop when the GPU does, so a synchronous launch "
                "returns only when the program exits"
            )
        marker = "// gpu breakpoint at tile load"
        target = self.make_target()
        target.BreakpointCreateByLocation(SOURCE, line_number(SOURCE, marker))
        self.setAsync(False)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped, "CPU")
        self.gpu_process.GetThreadAtIndex(0).GetFrameAtIndex(0)

        self.wait_for_gpu_to_stop()
        frame = self.gpu_process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assertEqual(
            frame.GetPC(),
            frame.FindRegister("pc").GetValueAsUnsigned(),
            "the PC of thread 1's frame 0 and its PC register",
        )
        self.assert_frame_at(frame, LAUNCH_32.kernel, SOURCE, marker)

    def test_host_reads_result_after_kernels(self):
        """After the last launch, once the program has copied C back to the
        host, the CPU target reads it there: every element is the dot product
        of a row of A and a column of B."""
        _, _, thread, _ = self.run_to_cpu_line(
            SOURCE, "// cpu breakpoint after copy back", target=self.make_target()
        )
        frame = thread.frame[0]
        width = LAUNCH_32.width
        for index in (0, width * width - 1):
            self.assert_float_variable(frame, f"h_C[{index}]", csub(width))
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue())
        self.assert_exit_status(0)
        self.check_output()
