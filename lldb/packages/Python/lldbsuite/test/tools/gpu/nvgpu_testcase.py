import math
from typing import Any, Callable, Optional

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

# A stop matcher receives the stopped GPU process and returns what the caller
# wants from that stop (a thread, a list of threads, ...) or None when the stop
# is not the one expected. Return None, not an empty list, for "no match".
StopMatcher = Callable[[lldb.SBProcess], Optional[Any]]


class NVGPUTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python NVIDIA GPU tests.

    Which wait to use:

      continue_cpu_and_wait_for_gpu_to_stop
          The CPU is stopped and resuming it is what makes the GPU run and
          reach its next stop (the usual first GPU stop after launch).
      wait_for_gpu_to_stop
          The GPU is already running (attach, interrupt, step) and the test
          only needs to observe its next stop. Nothing is resumed.
      assert_gpu_stop(match)
          The GPU is stopped where the test expects; the current stop must
          satisfy `match` (a thread, the threads at a breakpoint id, ...).
          Nothing is resumed.
      continue_gpu_until(match)
          Resume the GPU until a later stop satisfies `match`. The current stop
          is never examined. Bounded by a continue count and a timeout.
      assert_gpu_stop_reason(reason), continue_gpu_until_stop_reason(reason)
          Shorthands for "a thread stopped for this reason".

    The verb says what happens to the GPU: `wait_for` and `assert` never resume
    it, `continue` does.

    All of them consume only GPU process state-changed events, so CPU events
    stay queued for whoever waits on the CPU process.
    """

    NO_DEBUG_INFO_TESTCASE = True

    # TODO: the waits below assume the CPU keeps running while the
    # GPU is stopped, which holds only because lit.local.cfg sets
    # NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1. Once the plugin stops the CPU on a
    # GPU stop, continue_gpu_until must resume the CPU (or drain its stop event)
    # before resuming the GPU.

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

    def _next_gpu_state(self, listener: lldb.SBListener, timeout_seconds: int) -> int:
        """Pull the next GPU process state-changed event and return its state.

        Only events from the GPU process broadcaster are consumed, so CPU events
        stay queued. Stopped events flagged as "restarted" (a stop lldb resumed
        from on its own) are skipped the way lldbutil.expect_state_changes does.
        Fails the test on timeout.
        """
        # SBListener waits take a uint32 second count; round fractions up once here.
        num_seconds = math.ceil(timeout_seconds)
        broadcaster = self.gpu_process.GetBroadcaster()
        while True:
            event = lldb.SBEvent()
            if not listener.WaitForEventForBroadcasterWithType(
                num_seconds,
                broadcaster,
                lldb.SBProcess.eBroadcastBitStateChanged,
                event,
            ):
                self.fail(
                    f"timed out after {timeout_seconds} s waiting for the GPU process "
                    "to change state"
                )
            state = lldb.SBProcess.GetStateFromEvent(event)
            if state == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(
                event
            ):
                continue
            return state

    def wait_for_gpu_to_stop(self, timeout_seconds: int = 60) -> None:
        """Wait for the running GPU process to stop.

        Nothing is resumed. Running events (and restarted stops) are tolerated on
        the way to eStateStopped. Any other state fails the test, naming the exit
        status if the GPU process exited, so a kernel that never reaches the
        expected stop fails instead of hanging the test.
        """
        listener = self.dbg.GetListener()
        while True:
            state = self._next_gpu_state(listener, timeout_seconds)
            if state == lldb.eStateRunning:
                continue
            if state == lldb.eStateStopped:
                return
            if state == lldb.eStateExited:
                what = f"exited (status {self.gpu_process.GetExitStatus()})"
            else:
                what = f"entered state {lldb.SBDebugger.StateAsCString(state)}"
            self.fail(f"GPU process {what} before stopping")

    def continue_cpu_and_wait_for_gpu_to_stop(self, timeout_seconds: int = 60):
        """Resume the CPU process and wait for the GPU process to stop.

        The GPU target must already exist. This is the usual way to reach the
        first GPU stop: the CPU sits at a breakpoint before the kernel launch and
        resuming it lets the kernel run to its breakpoint or exception.
        """
        self.setAsync(True)
        error = self.cpu_process.Continue()
        self.assertSuccess(error, "continue CPU process")
        self.wait_for_gpu_to_stop(timeout_seconds)

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

    def assert_gpu_stop(
        self,
        match: StopMatcher,
        *,
        description: str = "matching the condition",
    ) -> Any:
        """The current GPU stop must satisfy `match`; return its result.

        Nothing is resumed: use this when the test already brought the GPU to the
        stop it wants to inspect. `description` names what was expected in the
        failure message.
        """
        found = match(self.gpu_process)
        if found is None:
            self.fail(
                f"current GPU stop is not {description}; "
                f"GPU threads:\n{self._describe_gpu_threads()}"
            )
        return found

    def continue_gpu_until(
        self,
        match: StopMatcher,
        *,
        timeout_seconds: int = 60,
        max_continues: int = 16,
        description: str = "matching the condition",
    ) -> Any:
        """Resume the GPU process until a later stop satisfies `match`; return its result.

        The current stop is never examined. If the GPU is already where the test
        wants it, call assert_gpu_stop instead.

        Two bounds keep a wrong expectation from hanging the test:

        - `timeout_seconds` is how long one resume may take to reach the next
          stop. It catches a kernel that runs on without stopping.
        - `max_continues` is how many resumes are made in total. It catches a
          sticky stop, such as a fatal exception: every resume stops again at
          the same place within milliseconds, so the timeout never fires and
          only the count ends the loop.

        More than one resume can be legitimate, because several warps hitting
        one breakpoint may arrive as several stops. Even so, give each check its
        own anchor (a breakpoint or function) and match on it, so a mismatch
        fails on the first resume instead of consuming the stops that belong to
        later checks.
        """
        self.setAsync(True)
        for _ in range(max_continues):
            error = self.gpu_process.Continue()
            self.assertSuccess(error, "continue GPU process")
            self.wait_for_gpu_to_stop(timeout_seconds)
            found = match(self.gpu_process)
            if found is not None:
                return found
        self.fail(
            f"no GPU stop {description} within {max_continues} continues; "
            f"GPU threads:\n{self._describe_gpu_threads()}"
        )

    @staticmethod
    def _stop_reason_matcher(stop_reason: int, thread_index: Optional[int]):
        """Build the (match, description) pair shared by the *_gpu_stop_reason helpers.

        When `thread_index` is given only that thread is checked (an index past
        the end yields an invalid thread whose stop reason is eStopReasonInvalid,
        which never matches); otherwise the first thread with the reason is
        returned.
        """

        def match(process: lldb.SBProcess) -> Optional[lldb.SBThread]:
            if thread_index is not None:
                thread = process.GetThreadAtIndex(thread_index)
                return thread if thread.GetStopReason() == stop_reason else None
            return next(
                (t for t in process.threads if t.GetStopReason() == stop_reason),
                None,
            )

        reason = lldbutil.stop_reason_to_str(stop_reason)
        where = f"thread {thread_index}" if thread_index is not None else "a thread"
        return match, f"with {where} stopped for {reason}"

    def assert_gpu_stop_reason(
        self, stop_reason: int, *, thread_index: Optional[int] = None
    ) -> lldb.SBThread:
        """The current GPU stop must have a thread stopped for `stop_reason`."""
        match, description = self._stop_reason_matcher(stop_reason, thread_index)
        return self.assert_gpu_stop(match, description=description)

    def continue_gpu_until_stop_reason(
        self,
        stop_reason: int,
        *,
        thread_index: Optional[int] = None,
        timeout_seconds: int = 60,
        max_continues: int = 16,
    ) -> lldb.SBThread:
        """Resume the GPU process until a thread stops for `stop_reason`.

        Bounds are as in continue_gpu_until.
        """
        match, description = self._stop_reason_matcher(stop_reason, thread_index)
        return self.continue_gpu_until(
            match,
            timeout_seconds=timeout_seconds,
            max_continues=max_continues,
            description=description,
        )
