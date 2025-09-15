import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvidiagpu_testcase import NvidiaGpuTestCaseBase


class TestNVIDIAGPUExit(NvidiaGpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_to_exit_code(self, exit_code: int):
        """Run to the exit code breakpoint."""
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

        self.assertEqual(self.cpu_process.GetExitStatus(), exit_code)
        self.assertEqual(self.gpu_process.GetExitStatus(), exit_code)

    def test_gpu_exit_0(self):
        """Test that both CPU and GPU exit with exit code 0."""
        self._run_to_exit_code(0)

    def test_gpu_exit_1(self):
        """Test that both CPU and GPU exit with exit code 1."""
        self._run_to_exit_code(1)
