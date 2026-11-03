"""
Tests for how the GPU target follows the native process's lifecycle.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SOURCE = "lifecycle.hip"
GPU_BREAKPOINT = "// GPU BREAKPOINT"


class AmdGpuLifecycleTestCase(AmdGpuTestCaseBase):
    def stop_at_gpu_breakpoint(self) -> lldb.SBListener:
        """Run until the waves stop at the GPU breakpoint, then stop the CPU."""
        self.build()
        threads = self.run_to_gpu_breakpoint(SOURCE, GPU_BREAKPOINT)
        self.assertTrue(threads, "GPU should be stopped at the breakpoint")
        listener = self.dbg.GetListener()
        self.stop_cpu_if_running(listener)
        return listener

    def continue_cpu_to_exit(self, listener: lldb.SBListener):
        """Continue the CPU process and check that the kernel ran to its end:
        the program exits with 0 only if the kernel wrote the right values."""
        cpu_process = self.cpu_process
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, cpu_process, [lldb.eStateRunning, lldb.eStateExited]
        )
        self.assertEqual(cpu_process.GetExitStatus(), 0)

    def test_gpu_detach_resumes_waves(self):
        """Test that detaching from the GPU resumes the stopped waves and
        leaves the CPU process under the debugger."""
        listener = self.stop_at_gpu_breakpoint()

        gpu_process = self.gpu_process
        self.assertSuccess(gpu_process.Detach())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(self.cpu_process.GetState(), lldb.eStateStopped)

        self.continue_cpu_to_exit(listener)

    def test_gpu_kill_kills_program(self):
        """Test that killing the GPU process kills the program."""
        self.stop_at_gpu_breakpoint()

        gpu_process = self.gpu_process
        cpu_process = self.cpu_process
        self.assertSuccess(gpu_process.Kill())
        self.assertState(gpu_process.GetState(), lldb.eStateExited)
        self.assertState(cpu_process.GetState(), lldb.eStateExited)

    def test_gpu_target_delete_keeps_cpu_process(self):
        """Test that deleting the GPU target detaches from the GPU only."""
        listener = self.stop_at_gpu_breakpoint()

        self.assertTrue(self.dbg.DeleteTarget(self.gpu_target))
        self.assertEqual(self.dbg.GetNumTargets(), 1)

        self.continue_cpu_to_exit(listener)
