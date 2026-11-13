"""
Test a detach that keeps the process stopped, which lldb-server cannot do.
"""

import lldb
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "keep_stopped.cu"


class TestNVGPUDetachKeepStopped(NVGPUTestCaseBase):
    def run_to_the_first_gpu_stop(self):
        """Stop the GPU at its first breakpoint; return the second."""
        if not self.cpu_stops_with_gpu():
            self.skipTest("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1 keeps the CPU running")
        first = self.run_to_gpu_line(
            SOURCE,
            "// gpu breakpoint first",
            exe_name=self.build_shared("keep_stopped"),
        )
        self.assert_gpu_stop_at_breakpoint(first)
        return self.gpu_breakpoint(SOURCE, "// gpu breakpoint second")

    def assert_still_debugged(self, second):
        """lldb still debugs both processes: the GPU stops at its second
        breakpoint, and the program runs to its end."""
        self.assertState(self.cpu_process.GetState(), lldb.eStateStopped)
        self.assertState(self.gpu_process.GetState(), lldb.eStateStopped)
        self.continue_gpu_until_breakpoint(second)
        # The program checks the kernel's results and exits 0 if they are right.
        self.continue_gpu_to_exit()

    def test_detach_that_keeps_the_process_stopped(self):
        """The detach fails before it changes anything, so the GPU target
        stays attached too."""
        second = self.run_to_the_first_gpu_stop()
        error = self.cpu_process.Detach(True)
        self.assertTrue(error.Fail(), "the process was kept stopped")
        self.assertIn("Stays stopped not supported", error.GetCString())
        self.assert_still_debugged(second)

    def test_detach_from_the_gpu_that_keeps_it_stopped(self):
        """The same for a detach from the GPU alone, which disables the GPU's
        breakpoint sites before it fails, and must enable them again."""
        second = self.run_to_the_first_gpu_stop()
        error = self.gpu_process.Detach(True)
        self.assertTrue(error.Fail(), "the GPU process was kept stopped")
        self.assertIn("Stays stopped not supported", error.GetCString())
        self.assert_still_debugged(second)
