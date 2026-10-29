"""
Test a program built with relocatable device code (-rdc=true) from two source
files.
"""

from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPURDC(NVGPUTestCaseBase):
    def test_call_and_launch_across_source_files(self):
        """Test a stop in a device function from the other source file, and
        that the kernel the program launches from the device runs."""
        self.build()
        self.run_to_cpu_line("main.cu", "// breakpoint before launch")

        breakpoint = self.gpu_target.BreakpointCreateByName("twice")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]

        # The backtrace goes from other.cu into the kernel in main.cu.
        self.assert_frame_at(
            thread.frame[0], "twice(int)", "other.cu", "// breakpoint in twice"
        )
        self.assert_frame_at(
            thread.frame[1], "parent(int*)", "main.cu", "// call to twice"
        )

        # The program checks the kernel's results and exits with 1 if the
        # kernel that parent launches did not run. Both processes report the
        # program's exit status.
        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        self.assertEqual(self.wait_for_exit(self.gpu_process), 0, "GPU exit status")
        self.assertEqual(self.wait_for_exit(self.cpu_process), 0, "CPU exit status")
