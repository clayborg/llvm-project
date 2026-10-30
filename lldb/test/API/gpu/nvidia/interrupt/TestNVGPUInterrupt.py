import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_decorators import Driver, skipUnlessNVGPU
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "interrupt.cu"
# Interrupting the GPU needs a driver that suspends the devices in the
# background, which CUDA 13.2 added.
INTERRUPT_DRIVER = Driver(at_least=(13, 2))


class TestNVGPUInterrupt(NVGPUTestCaseBase):
    """Tests for interrupting the GPU while a kernel runs."""

    def run_to_spinning_kernel(self):
        """Run the CPU to the line where the kernel is running, and leave the
        GPU running it."""
        exe = self.build_shared("interrupt")
        self.run_to_cpu_line(SOURCE, "// kernel is running", exe_name=exe)
        self.setAsync(True)
        # Nothing pulled the GPU's events while the CPU ran to its breakpoint.
        self.assertEqual(
            self.pull_state(self.gpu_process),
            lldb.eStateRunning,
            "the GPU should be running the kernel",
        )

    def interrupt_gpu(self):
        """Interrupt the GPU, and check that it stopped with SIGSTOP and the
        kernel's threads."""
        self.assertSuccess(self.gpu_process.Stop(), "interrupt the GPU")
        self.wait_for_gpu_to_stop()

        self.assertEqual(
            sorted(self.thread_coords(thread) for thread in self.gpu_process.threads),
            [((0, 0, 0), (x, 0, 0)) for x in range(32)],
            self._describe_gpu_threads(),
        )
        thread = self.gpu_process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            self.gpu_process.GetUnixSignals().GetSignalNumberFromName("SIGSTOP"),
            thread.GetStopDescription(256),
        )

    def breakpoint_in_the_loop(self):
        breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in the loop")
        )
        self.assertGreater(breakpoint.GetNumLocations(), 0, "no breakpoint location")
        return breakpoint

    @skipUnlessNVGPU(INTERRUPT_DRIVER)
    def test_interrupt_a_running_kernel(self):
        """Interrupting the GPU while a kernel spins stops it with the kernel's
        threads, instead of timing out."""
        self.run_to_spinning_kernel()
        self.interrupt_gpu()

    @skipUnlessNVGPU(INTERRUPT_DRIVER)
    def test_continue_to_a_breakpoint_after_an_interrupt(self):
        """After an interrupt, the GPU runs on to a breakpoint in the kernel,
        which is reported as a breakpoint stop, with no trace of the
        interrupt."""
        self.run_to_spinning_kernel()
        self.interrupt_gpu()
        breakpoint = self.breakpoint_in_the_loop()

        self.continue_gpu_until_breakpoint(breakpoint)
        for thread in self.gpu_process.threads:
            self.assertNotEqual(
                thread.GetStopReason(),
                lldb.eStopReasonSignal,
                self._describe_gpu_threads(),
            )

    @skipUnlessNVGPU(INTERRUPT_DRIVER)
    def test_interrupt_leaves_the_cpu_running(self):
        """Interrupting the GPU does not stop the CPU, as a breakpoint or an
        exception would: lldb-server cannot tell the user's interrupt from one
        that lldb makes to change a breakpoint."""
        self.run_to_spinning_kernel()
        self.assertSuccess(self.cpu_process.Continue(), "continue the CPU")
        self.assertEqual(
            self._next_state(self.cpu_process, "CPU", self.dbg.GetListener(), 60),
            lldb.eStateRunning,
        )

        self.interrupt_gpu()
        # lldb-server would stop the CPU right after the GPU.
        self.assertFalse(
            self.dbg.GetListener().WaitForEventForBroadcasterWithType(
                2,
                self.cpu_process.GetBroadcaster(),
                lldb.SBProcess.eBroadcastBitStateChanged,
                lldb.SBEvent(),
            ),
            "the CPU process changed state after the GPU was interrupted",
        )

    @skipUnlessNVGPU(INTERRUPT_DRIVER)
    def test_set_a_breakpoint_while_a_kernel_runs(self):
        """Setting a breakpoint in a running kernel interrupts the GPU only to
        insert it: lldb resumes the GPU at once, with no stop the user sees,
        and the kernel then stops at the breakpoint."""
        self.run_to_spinning_kernel()
        breakpoint = self.breakpoint_in_the_loop()

        self.wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(breakpoint)
