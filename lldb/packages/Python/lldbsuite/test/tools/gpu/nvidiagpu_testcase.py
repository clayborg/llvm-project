from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
from lldbsuite.test import lldbutil
from typing import Callable

class NvidiaGpuTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python NVIDIA GPU tests.
    """
    NO_DEBUG_INFO_TESTCASE = True

    def continue_cpu_and_wait_for_gpu_to_stop(self):
        """ Resume the CPU process and wait for the GPU process to stop. The gpu_target must be already running."""
        self.setAsync(True)
        listener = self.dbg.GetListener()
        self.cpu_process.Continue()
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

    def find_some_thread(self, condition: Callable[[lldb.SBThread], bool]) -> lldb.SBThread:
        """ Find some thread that satisfies the given condition. It raises if no threads satisfy the condition."""
        return next(filter(condition, self.gpu_process.threads))

    def find_thread_by_name(self, name: str) -> lldb.SBThread:
        """ Find a thread by name. It raises if no threads have the given name."""
        return next(filter(lambda thread: thread.GetName() == name, self.gpu_process.threads))

    def find_thread_by_stop_reason(self, stop_reason: int) -> lldb.SBThread:
        """ Find a thread by stop reason. It raises if no threads have the given stop reason."""
        return next(filter(lambda thread: thread.GetStopReason() == stop_reason, self.gpu_process.threads))
