from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
from lldbsuite.test import lldbutil
from typing import Callable


class NVGPUTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python NVIDIA GPU tests.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def killCPUOnTeardown(self):
        # TestBase.tearDown deletes all targets (and kills their processes)
        # before running registered hooks, so by the time this fires
        # self.cpu_process may already be gone. Guard against that instead
        # of blowing up the test with an AttributeError.
        def kill_cpu():
            proc = self.cpu_process
            if proc:
                proc.Kill()
        self.addTearDownHook(kill_cpu)

    def continue_cpu_and_wait_for_gpu_to_stop(self):
        """Resume the CPU process and wait for the GPU process to stop. The gpu_target must be already running."""
        self.setAsync(True)
        listener = self.dbg.GetListener()
        self.cpu_process.Continue()
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

    def _describe_gpu_threads(self, limit: int = 8) -> str:
        """A short listing of GPU threads for assertion messages, one per line.

        Threads that stopped for a reason are listed first, since they are almost
        always the ones a failed lookup is about; a kernel can have thousands of
        idle lanes, so the listing is capped at `limit`.
        """
        stopped = []
        idle = []
        for thread in self.gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonNone:
                idle.append(f"  #{thread.idx} {thread.GetName()}")
            else:
                stopped.append(
                    f"  #{thread.idx} {thread.GetName()}  [{thread.GetStopDescription(256)}]"
                )
        lines = (stopped + idle)[:limit]
        total = len(stopped) + len(idle)
        if total > limit:
            lines.append(f"  ... ({total} total, {len(stopped)} with a stop reason)")
        return "\n".join(lines)

    def find_some_thread(
        self,
        condition: Callable[[lldb.SBThread], bool],
        description: str = "matching the condition",
    ) -> lldb.SBThread:
        """Return the first GPU thread satisfying `condition`.

        Fails the test with a message listing the threads when none matches, instead
        of raising StopIteration from inside the test body.
        """
        thread = next(filter(condition, self.gpu_process.threads), None)
        if thread is None:
            self.fail(
                f"no GPU thread {description}; GPU threads:\n{self._describe_gpu_threads()}"
            )
        return thread

    def find_thread_by_name(self, name: str) -> lldb.SBThread:
        """Return the first GPU thread whose name contains `name`; fail if none does."""
        return self.find_some_thread(
            lambda thread: name in thread.GetName(),
            description=f"with {name!r} in its name",
        )

    def find_thread_by_stop_reason(self, stop_reason: int) -> lldb.SBThread:
        """Return the first GPU thread with the given stop reason; fail if none has."""
        return self.find_some_thread(
            lambda thread: thread.GetStopReason() == stop_reason,
            description=f"with stop reason {lldbutil.stop_reason_to_str(stop_reason)}",
        )
