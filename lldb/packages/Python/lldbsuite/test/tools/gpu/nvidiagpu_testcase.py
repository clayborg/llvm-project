from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
from lldbsuite.test import lldbutil

class NvidiaGpuTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python NVIDIA GPU tests.
    """
    NO_DEBUG_INFO_TESTCASE = True

    def continue_cpu_and_wait_for_gpu_to_stop(self):
        self.setAsync(True)
        listener = self.dbg.GetListener()
        self.cpu_process.Continue()
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
