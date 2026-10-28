import math
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import lldb
from lldbsuite.test import configuration, lldbplatformutil, lldbutil
from lldbsuite.test.lldbtest import builder_module, line_number
from lldbsuite.test.tools.gpu import nvgpu_decorators
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

# A stop matcher receives the stopped GPU process and returns what the caller
# wants from that stop (a thread, a list of threads, ...) or None when the stop
# is not the one expected. Return None, not an empty list, for "no match".
StopMatcher = Callable[[lldb.SBProcess], Optional[Any]]

# The programs build_shared has built in this run, by test class and EXE.
_shared_builds: Dict[Tuple[type, str], str] = {}


def _with_gpu_check(setup):
    """Wrap a test's setUp. A test that needs a GPU is skipped before setUp
    runs, on a machine that cannot run a CUDA program or whose GPU does not
    meet the requirements of the method's @skipUnlessNVGPU marks, so that the
    base setUp does not delete the build directory of the last run. After
    setUp, it fails if the configured nvcc cannot build for this machine's
    GPUs."""

    def checked_setUp(self):
        needs_gpu = nvgpu_decorators.needs_gpu(self)
        if needs_gpu:
            reason = nvgpu_decorators.skip_reason(self)
            if reason:
                self.skipTest(reason)
        setup(self)
        if needs_gpu:
            self.fail_on_unusable_nvcc()

    return checked_setUp


class NVGPUTestCaseBase(GpuTestCaseBase):
    """
    Base class for NVIDIA GPU tests. A test that reads core files uses its
    subclass NVGPUCoreTestCaseBase (nvgpu_core_testcase.py), which adds making
    and loading core files.

    run_to_cpu_line(source, marker) launches the built program and runs the CPU
    to a marked line, the usual first step of a test. Its breakpoint matches the
    line exactly, so the copy lldb makes of it in the GPU target cannot move to
    device code.

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
      wait_for_stop(process)
          wait_for_gpu_to_stop for either process. For the CPU: it kept running
          while the GPU stopped, and the test needs it at one of its own
          breakpoints too. Nothing is resumed.
      wait_for_exit(process)
          The test resumed both processes and expects them to finish. Nothing
          is resumed.
      assert_exit_status(status)
          The same for both processes, which must exit with `status`.

    The verb says what happens to the GPU: `wait_for` and `assert` never resume
    it, `continue` does.

    Each consumes only the state-changed events of the processes it waits on,
    so the other process's events stay queued for whoever waits on it.
    """

    NO_DEBUG_INFO_TESTCASE = True

    # TODO: the waits below assume the CPU keeps running while the
    # GPU is stopped, which holds only because lit.local.cfg sets
    # NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1. Once the plugin stops the CPU on a
    # GPU stop, continue_gpu_until must resume the CPU (or drain its stop event)
    # before resuming the GPU.

    # Tests on this base class debug a live CUDA program, so they need a GPU.
    # NVGPUCoreTestCaseBase sets this to False. A test class can set it too,
    # and @skipUnlessNVGPUDevice from nvgpu_decorators marks one method of a
    # class that needs no GPU. @skipUnlessNVGPU marks a method that needs more
    # of the GPU, such as a compute capability.
    NEEDS_GPU = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Many LLDB tests define setUp and call TestBase.setUp(self) directly,
        # which would bypass the check in setUp below, so wrap theirs too.
        if "setUp" in cls.__dict__:
            cls.setUp = _with_gpu_check(cls.__dict__["setUp"])

    @_with_gpu_check
    def setUp(self):
        super().setUp()

    def skip_without_gpu(self):
        """Skip this test on a machine that cannot run a CUDA program on a
        GPU, and on a remote platform."""
        reason = nvgpu_decorators.device_skip_reason()
        if reason:
            self.skipTest(reason)

    def fail_on_unusable_nvcc(self):
        """Fail this test if the configured nvcc cannot build a CUDA program for
        this machine's GPUs: a misconfigured machine must not look like one
        without a GPU."""
        error = nvgpu_decorators.nvcc_error()
        if error:
            self.fail(error)

    def build_shared(self, exe: str) -> str:
        """Build the test's program once for all the test methods of this
        class, with EXE set to `exe`, and return its path.

        The program is built in a directory of its own,
        `<Module>.<Class>.<exe>`, next to the per-method directories that
        build() uses. The first method of a run that asks for it builds it,
        after deleting what an earlier run left there; later methods of the
        class get the same path, since dotest runs a file's methods one after
        another in one process.

        Use this for a program that several test methods run unchanged, such
        as a copied sample. It takes no make variables: a test that builds
        variants of a program uses build() in each method. The program still
        runs in the method's own build directory, which stays empty, so a
        program that reads files from its working directory cannot use this as
        it is.
        """
        cls = type(self)
        path = _shared_builds.get((cls, exe))
        if path:
            return path

        build_dir_name = f"{cls.__module__}.{cls.__name__}.{exe}"
        build_dir = os.path.join(
            configuration.test_build_dir, self.mydir, build_dir_name
        )
        # Start from an empty directory, as TestBase.makeBuildDir does for a
        # method's directory.
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
        lldbutil.mkdir_p(build_dir)
        # No debug information variants: NO_DEBUG_INFO_TESTCASE runs each
        # method once.
        command = builder_module().getBuildCommand(
            debug_info=None,
            architecture=configuration.arch,
            dictionary=lldbplatformutil.finalize_build_dictionary({"EXE": exe}),
            testdir=self.mydir,
            testname=build_dir_name,
        )
        if command is None:
            raise Exception("Don't know how to build binary")
        self.runBuildCommand(command)
        path = os.path.join(build_dir, exe)
        _shared_builds[(cls, exe)] = path
        return path

    def run_to_cpu_line(
        self,
        source: str,
        marker: str,
        exe_name: Optional[str] = None,
        target: Optional[lldb.SBTarget] = None,
    ) -> Tuple[lldb.SBTarget, lldb.SBProcess, lldb.SBThread, lldb.SBBreakpoint]:
        """Launch the built program and run the CPU to the line of `source`
        that contains `marker`, as lldbutil.run_to_line_breakpoint does. Return
        (target, process, thread, breakpoint). `exe_name` is a file in the
        method's build directory, a.out by default, or an absolute path such
        as build_shared returns. A test that sets breakpoints before the run
        creates the target itself and passes it as `target` instead.

        The breakpoint matches that line exactly. When lldb creates the GPU
        target it copies every CPU breakpoint into it. A copy that may move to
        the nearest code resolves in a cubin only inside a function that begins
        at or before the marked line, with one line of slack, so a marker on
        the line directly above a kernel's definition would stop the GPU in
        that kernel. An exact match never resolves to device code, so the
        layout of the program does not matter.
        """
        if target is None:
            exe_name = exe_name or "a.out"
            # An absolute exe_name, such as one from build_shared, is used as is.
            target = lldbutil.run_to_breakpoint_make_target(
                self, exe_name, in_cwd=not os.path.isabs(exe_name)
            )
        else:
            self.assertIsNone(exe_name, "pass exe_name or target, not both")
        line = line_number(source, marker)
        breakpoint = target.BreakpointCreateByLocation(
            lldb.SBFileSpec(source), line, 0, 0, lldb.SBFileSpecList(), False
        )
        self.assertGreater(
            breakpoint.GetNumLocations(),
            0,
            f"no CPU code at {source}:{line}, the line marked {marker!r}",
        )
        return lldbutil.run_to_breakpoint_do_run(self, target, breakpoint)

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

    def _next_state(
        self,
        process: lldb.SBProcess,
        what: str,
        listener: lldb.SBListener,
        timeout_seconds: int,
    ) -> int:
        """Pull the next state-changed event of `process` and return its state.

        Only events from that process's broadcaster are consumed, so the other
        process's events stay queued. Stopped events flagged as "restarted" (a
        stop lldb resumed from on its own) are skipped the way
        lldbutil.expect_state_changes does. Fails the test on timeout, naming
        `what` ("CPU" or "GPU").
        """
        # SBListener waits take a uint32 second count; round fractions up once here.
        num_seconds = math.ceil(timeout_seconds)
        broadcaster = process.GetBroadcaster()
        while True:
            event = lldb.SBEvent()
            if not listener.WaitForEventForBroadcasterWithType(
                num_seconds,
                broadcaster,
                lldb.SBProcess.eBroadcastBitStateChanged,
                event,
            ):
                self.fail(
                    f"timed out after {timeout_seconds} s waiting for the {what} "
                    "process to change state"
                )
            state = lldb.SBProcess.GetStateFromEvent(event)
            if state == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(
                event
            ):
                continue
            return state

    def _wait_for_state(
        self,
        process: lldb.SBProcess,
        wanted: int,
        doing: str,
        timeout_seconds: int,
    ) -> None:
        """Pull the state-changed events of `process` until it enters `wanted`.

        Running events (and restarted stops) are skipped. Any other state fails
        the test, saying what the process did before `doing` ("stopping",
        "exiting"): with the exit status and its description if it exited, so
        that a lost connection to lldb-server (status -1) is told apart from
        the program ending. A program that never gets where the test expects
        fails instead of hanging the test.
        """
        what = "GPU" if process.GetTarget() == self.gpu_target else "CPU"
        listener = self.dbg.GetListener()
        while True:
            state = self._next_state(process, what, listener, timeout_seconds)
            if state == wanted:
                return
            if state == lldb.eStateRunning:
                continue
            if state == lldb.eStateExited:
                status = process.GetExitStatus()
                description = process.GetExitDescription()
                if description:
                    entered = f"exited (status {status}: {description})"
                else:
                    entered = f"exited (status {status})"
            else:
                entered = f"entered state {lldb.SBDebugger.StateAsCString(state)}"
            self.fail(f"{what} process {entered} before {doing}")

    def wait_for_stop(self, process: lldb.SBProcess, timeout_seconds: int = 60) -> None:
        """Wait for `process` (CPU or GPU), which is running, to stop. Nothing
        is resumed."""
        self._wait_for_state(process, lldb.eStateStopped, "stopping", timeout_seconds)

    def wait_for_exit(self, process: lldb.SBProcess, timeout_seconds: int = 60) -> int:
        """Wait for `process` (CPU or GPU) to exit and return its exit status.

        Nothing is resumed: the test resumes both processes first. A stop fails
        the test, so a program that stops again instead of finishing fails
        instead of hanging.
        """
        self._wait_for_state(process, lldb.eStateExited, "exiting", timeout_seconds)
        return process.GetExitStatus()

    def assert_exit_status(self, status: int, timeout_seconds: int = 60) -> None:
        """Both processes, which the test resumed, must exit with `status`.
        Nothing is resumed."""
        # The GPU first: if it stops instead, that fails here at once, while
        # the CPU would only wait for it until the timeout.
        for process, what in ((self.gpu_process, "GPU"), (self.cpu_process, "CPU")):
            self.assertEqual(
                self.wait_for_exit(process, timeout_seconds),
                status,
                f"{what} exit status ({process.GetExitDescription() or 'no description'})",
            )

    def read_stdout(self) -> str:
        """Return what the CPU process wrote to stdout that the test has not
        read yet. lldb collects it on a thread of its own, so output the
        program has just written may not be here yet."""
        chunks = []
        while chunk := self.cpu_process.GetSTDOUT(4096):
            chunks.append(chunk)
        return "".join(chunks)

    def wait_for_gpu_to_stop(self, timeout_seconds: int = 60) -> None:
        """Wait for the running GPU process to stop. Nothing is resumed."""
        self.wait_for_stop(self.gpu_process, timeout_seconds)

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

    @staticmethod
    def _breakpoint_matcher(breakpoint: lldb.SBBreakpoint):
        """Build the (match, description) pair shared by the *_breakpoint helpers.

        The match is the list of threads stopped at `breakpoint`.
        """

        def match(process: lldb.SBProcess) -> Optional[List[lldb.SBThread]]:
            threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
            return threads or None

        return match, f"at {breakpoint}"

    def assert_gpu_stop_at_breakpoint(
        self, breakpoint: lldb.SBBreakpoint
    ) -> List[lldb.SBThread]:
        """The current GPU stop must have threads stopped at `breakpoint`; return
        them."""
        match, description = self._breakpoint_matcher(breakpoint)
        return self.assert_gpu_stop(match, description=description)

    def continue_gpu_until_breakpoint(
        self,
        breakpoint: lldb.SBBreakpoint,
        *,
        timeout_seconds: int = 60,
        max_continues: int = 16,
    ) -> List[lldb.SBThread]:
        """Resume the GPU process until threads stop at `breakpoint`; return them.

        Bounds are as in continue_gpu_until.
        """
        match, description = self._breakpoint_matcher(breakpoint)
        return self.continue_gpu_until(
            match,
            timeout_seconds=timeout_seconds,
            max_continues=max_continues,
            description=description,
        )

    def assert_frame_at(
        self, frame: lldb.SBFrame, function: str, source: str, marker: str
    ) -> None:
        """`frame` must be in `function`, at the line of `source` that contains
        `marker`."""
        self.assertEqual(frame.GetFunctionName(), function, f"function of {frame}")
        line_entry = frame.GetLineEntry()
        self.assertEqual(
            line_entry.GetFileSpec().GetFilename(), source, f"source file of {frame}"
        )
        self.assertEqual(
            line_entry.GetLine(),
            line_number(source, marker),
            f"line of {frame}, which should be the line marked {marker!r}",
        )

    def read_vec3_register(self, frame: lldb.SBFrame, name: str) -> List[int]:
        """Read a 3-component (dim3/uint3) vector register as [x, y, z]."""
        reg = frame.FindRegister(name)
        self.assertTrue(reg.IsValid(), f"{name} should be a valid register")
        data = reg.GetData()
        vals = []
        for i in range(3):
            err = lldb.SBError()
            vals.append(data.GetUnsignedInt32(err, i * 4))
            self.assertTrue(
                err.Success(), f"reading {name}[{i}] failed: {err.GetCString()}"
            )
        return vals

    def thread_coords(
        self, thread: lldb.SBThread
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return a GPU thread's (blockIdx, threadIdx), each as (x, y, z), from
        the CUDA coordinates lldb-server reports for it. Tell threads apart by
        these, not by their names, which are for display."""

        def read(path: str) -> int:
            stream = lldb.SBStream()
            self.assertTrue(
                thread.GetInfoItemByPathAsString(path, stream),
                f"{thread.GetName()} has no {path}",
            )
            # The SB API writes integers in hex, as "0x1f". Base 0 reads that
            # and a decimal alike, instead of misreading a decimal as hex.
            return int(stream.GetData(), 0)

        block = tuple(read(f"cuda.block_idx.{axis}") for axis in "xyz")
        index = tuple(read(f"cuda.thread_idx.{axis}") for axis in "xyz")
        return block, index
