"""
Test that the CPU stops when the GPU stops, which is the default, and that the
test helpers resume it.
"""

import time

import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "cpu_stops_with_gpu.cu"


class TestNVGPUCPUStopsWithGPU(NVGPUTestCaseBase):
    def run_to_launch(self):
        """Run the program to its kernel launch and set the two GPU
        breakpoints; return them."""
        if not self.cpu_stops_with_gpu():
            self.skipTest("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1 keeps the CPU running")
        self.build()
        self.run_to_cpu_line(SOURCE, "// cpu breakpoint before launch")
        return [
            self.gpu_target.BreakpointCreateByLocation(
                SOURCE, line_number(SOURCE, marker)
            )
            for marker in ("// gpu breakpoint first", "// gpu breakpoint second")
        ]

    def assert_cpu_stopped_for_target_sync(self):
        """The CPU is stopped, and one of its threads reports a target-sync
        stop for the GPU target."""
        self.assertState(self.cpu_process.GetState(), lldb.eStateStopped)
        threads = [
            thread
            for thread in self.cpu_process.threads
            if thread.GetStopReason() == lldb.eStopReasonTargetSync
        ]
        self.assertTrue(threads, "no CPU thread stopped for target sync")
        index = self.dbg.GetIndexOfTarget(self.gpu_target)
        self.assertEqual(
            threads[0].GetStopDescription(256), f"target sync with target #{index}"
        )

    def finish(self, breakpoints):
        """Delete the breakpoints and let the program run to the end, where it
        checks the kernel's results."""
        for breakpoint in breakpoints:
            self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        self.assert_exit_status(0)

    def test_cpu_stops_for_target_sync(self):
        """At each GPU stop the CPU stops too, and continue_gpu_until resumes
        it along with the GPU."""
        first, second = breakpoints = self.run_to_launch()
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(first)

        self.continue_gpu_until_breakpoint(second)
        self.assert_cpu_stopped_for_target_sync()
        self.finish(breakpoints)

    def test_gpu_stop_queued_with_cpu_stop(self):
        """If both stop events are already queued when the test starts waiting
        for the GPU, the wait returns the GPU's stop and leaves the CPU
        stopped. Resuming the CPU there would run it past the GPU's stop."""
        first, _ = breakpoints = self.run_to_launch()
        cpu_stop_id = self.cpu_process.GetStopID()
        gpu_stop_id = self.gpu_process.GetStopID()
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue())

        # Wait for both stops without pulling any events: a process's stop ID
        # changes before lldb queues the stop event.
        deadline = time.monotonic() + 60
        while (
            self.cpu_process.GetStopID() == cpu_stop_id
            or self.gpu_process.GetStopID() == gpu_stop_id
        ):
            self.assertLess(time.monotonic(), deadline, "the processes did not stop")
            time.sleep(0.1)
        # Give lldb time to queue both events.
        time.sleep(2)

        self.wait_for_gpu_to_stop(keep_cpu_running=True)
        self.assert_gpu_stop_at_breakpoint(first)
        self.assert_cpu_stopped_for_target_sync()
        self.finish(breakpoints)
