import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUExit(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_to_exit_code(self, exit_code: int):
        """Run to the exit code breakpoint."""
        self.killCPUOnTeardown()

        self.build()
        source = "empty.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")
        launch_info = lldb.SBLaunchInfo([str(exit_code)])
        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line, launch_info=launch_info)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.setAsync(True)
        listener = self.dbg.GetListener()
        self.cpu_process.Continue()
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateRunning, lldb.eStateExited])
        lldbutil.expect_state_changes(self, listener, self.cpu_process, [lldb.eStateRunning, lldb.eStateExited])

        # Process::SetExitStatus records the status before it raises the
        # eStateExited event, and the public state (which GetExitStatus checks)
        # is updated when the listener hands the event over, so the status is
        # readable as soon as expect_state_changes returns. No polling needed.
        gpu_status = self.gpu_process.GetExitStatus()
        cpu_status = self.cpu_process.GetExitStatus()
        self.assertNotEqual(gpu_status, -1, "GPU process reported no exit status")
        self.assertNotEqual(cpu_status, -1, "CPU process reported no exit status")

        self.assertEqual(cpu_status, exit_code)
        self.assertEqual(gpu_status, exit_code)

    def test_gpu_exit_0(self):
        """Test that both CPU and GPU exit with exit code 0."""
        self._run_to_exit_code(0)

    def test_gpu_exit_1(self):
        """Test that both CPU and GPU exit with exit code 1."""
        self._run_to_exit_code(1)
