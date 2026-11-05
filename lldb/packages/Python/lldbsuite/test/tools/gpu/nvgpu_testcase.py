import json
import math
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import lldb
from lldbsuite.test import configuration, lldbplatformutil, lldbutil
from lldbsuite.test.lldbtest import builder_module, line_number
from lldbsuite.test.tools.gpu import nvgpu_decorators
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

# A stop matcher receives the stopped GPU process and returns what the caller
# wants from that stop (a thread, a list of threads, ...) or None when the stop
# is not the one expected. Return None, not an empty list, for "no match".
StopMatcher = Callable[[lldb.SBProcess], Optional[Any]]

# A GPU thread's CUDA coordinates, (blockIdx, threadIdx), each as (x, y, z):
# what thread_coords returns.
ThreadCoords = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

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
          wait_for_gpu_to_stop for either process. For the CPU: the test
          resumed it, and needs it at one of its own breakpoints too. Nothing
          is resumed, except the CPU from target-sync stops (see below).
      wait_for_exit(process)
          The test resumed both processes and expects them to finish. Nothing
          is resumed, except the CPU from target-sync stops (see below).
      assert_exit_status(status)
          The same for both processes, which must exit with `status`.

    The verb says what happens to the GPU: `wait_for` and `assert` never resume
    it, `continue` does.

    When the GPU stops at a breakpoint or an exception, lldb-server stops the
    CPU too, with eStopReasonTargetSync: a "target-sync stop".
    NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1 turns that off, and the helpers work
    either way. continue_gpu_until, wait_for_exit and wait_for_stop on the CPU
    resume the CPU from target-sync stops, since the program may need the CPU
    to run on: to launch a kernel, load a module or exit. continue_gpu_until
    still returns with the CPU stopped alongside the GPU.

    Each consumes only the state-changed events of the processes it waits on,
    so the other process's events stay queued for whoever waits on it, except
    that the helpers that resume the CPU consume its events too.
    """

    NO_DEBUG_INFO_TESTCASE = True

    # How long continue_gpu_until waits for the CPU's target-sync stop after a
    # GPU stop. The event is usually there already. The limit only matters for
    # a GPU stop that doesn't stop the CPU, one that isn't a breakpoint or an
    # exception.
    CPU_STOP_GRACE_SECONDS = 5

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

    def make_shared_target(
        self, exe: str, args: Optional[List[str]] = None
    ) -> lldb.SBTarget:
        """Build the program with build_shared(exe) and return a target for
        it, for a test that sets breakpoints before the run. Every launch of
        the target passes `args` to the program, as `settings set
        target.run-args` does: the launches of the helpers here, and `run` or
        `process launch` without arguments."""
        target = lldbutil.run_to_breakpoint_make_target(
            self, self.build_shared(exe), in_cwd=False
        )
        if args:
            launch_info = target.GetLaunchInfo()
            launch_info.SetArguments(args, False)
            target.SetLaunchInfo(launch_info)
        return target

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

    def launch_and_wait_for_gpu_target(
        self, target: lldb.SBTarget, timeout_seconds: int = 60
    ) -> lldb.SBProcess:
        """Launch the program of `target` and wait for lldb to create the GPU
        target, which it does when the program initializes CUDA. Return the
        CPU process, which keeps running.

        This is for a program that makes no CUDA call before its first kernel
        launch: no CPU line before the launch has the GPU target in place, so
        run_to_cpu_line cannot bring it to a point where GPU breakpoints can be
        set. Set them on the CPU target before the run instead; lldb copies
        them into the GPU target when it creates it (see gpu_breakpoint_copy).

        The CPU target announces the GPU target with the event that lldb-dap
        uses to open a session for it, once the GPU process is connected and
        the breakpoints are copied.
        """

        def launch() -> lldb.SBProcess:
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertTrue(process.IsValid(), "the program did not launch")
            return process

        return self._start_and_wait_for_gpu_target(target, launch, timeout_seconds)

    def continue_and_wait_for_gpu_target(
        self, timeout_seconds: int = 60
    ) -> lldb.SBProcess:
        """Resume the CPU process and wait for lldb to create the GPU target,
        as it does once a program that lldb attached to runs again and its CUDA
        driver attaches. Return the CPU process, which keeps running."""
        self.assertIsNone(self.gpu_target, "the GPU target exists already")
        process = self.cpu_process

        def resume() -> lldb.SBProcess:
            self.assertSuccess(process.Continue(), "continue CPU process")
            return process

        return self._start_and_wait_for_gpu_target(
            self.cpu_target, resume, timeout_seconds
        )

    @staticmethod
    def wait_until(predicate: Callable[[], bool], timeout_seconds: float = 60) -> bool:
        """Poll `predicate` until it returns true or `timeout_seconds` pass, and
        return whether it did. For what lldb has no event for, such as a file
        that the program creates."""
        deadline = time.monotonic() + timeout_seconds
        while not predicate():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def _start_and_wait_for_gpu_target(
        self,
        target: lldb.SBTarget,
        start: Callable[[], lldb.SBProcess],
        timeout_seconds: int,
    ) -> lldb.SBProcess:
        """Call `start`, which launches or resumes the CPU process of `target`
        and returns it, and wait for lldb to create the GPU target. Return the
        CPU process."""
        self.setAsync(True)
        # A listener of its own, which takes only this event, so that the
        # state-changed events the other helpers wait for stay queued.
        listener = lldb.SBListener("nvgpu-test-gpu-target-created")
        broadcaster = target.GetBroadcaster()
        created_bit = lldb.SBTarget.eBroadcastBitNewTargetCreated
        self.assertEqual(
            listener.StartListeningForEvents(broadcaster, created_bit),
            created_bit,
            "listening for the GPU target's creation",
        )
        try:
            process = start()
            event = lldb.SBEvent()
            if not listener.WaitForEvent(math.ceil(timeout_seconds), event):
                # A program that fails before it initializes CUDA has exited
                # by now: say so, with what it printed, rather than that lldb
                # made no target.
                state = self.pull_state(process)
                if state == lldb.eStateExited:
                    self.fail(
                        f"the program exited (status {process.GetExitStatus()}) "
                        "without lldb creating a GPU target; its stdout:\n"
                        f"{self.read_stdout()}\nits stderr:\n{self.read_stderr()}"
                    )
                self.fail(
                    f"lldb created no GPU target within {timeout_seconds} s; the "
                    f"CPU process is {lldb.SBDebugger.StateAsCString(state)}"
                )
        finally:
            listener.StopListeningForEvents(broadcaster, created_bit)
        created = lldb.SBTarget.GetCreatedTargetFromEvent(event)
        self.assertTrue(created.IsValid(), "the event names no target")
        self.assertEqual(created, self.gpu_target, "the created target")
        self.assertTrue(self.gpu_process.IsValid(), "the GPU target has no process")
        return process

    @staticmethod
    def _breakpoint_settings(
        breakpoint: lldb.SBBreakpoint,
    ) -> Optional[Dict[str, Any]]:
        """Return what `breakpoint` stops at and how, as lldb serializes it:
        its resolver, filter, options and names, but neither its ID nor its
        locations. Return None if it does not serialize."""
        stream = lldb.SBStream()
        if not breakpoint.SerializeToStructuredData().GetAsJSON(stream).Success():
            return None
        return json.loads(stream.GetData())

    def gpu_breakpoint_copy(self, breakpoint: lldb.SBBreakpoint) -> lldb.SBBreakpoint:
        """Return the GPU target's copy of `breakpoint`, a breakpoint set on
        the CPU target before lldb created the GPU target, which copies every
        CPU breakpoint into it.

        The copy has the original's settings, names included, but not
        necessarily its ID: the GPU target numbers its breakpoints in the order
        it makes them, so once a CPU breakpoint has been deleted, the copy of
        one breakpoint has the ID of another. The copy is found by its settings
        instead, and must be the only GPU breakpoint with them;
        run_to_gpu_breakpoint names its breakpoint to tell it apart from an
        identical one.
        """
        settings = self._breakpoint_settings(breakpoint)
        self.assertIsNotNone(
            settings, f"{breakpoint} does not serialize, so its copy cannot be found"
        )
        copies = [
            candidate
            for candidate in self.gpu_target.breakpoint_iter()
            if self._breakpoint_settings(candidate) == settings
        ]
        self.assertEqual(
            len(copies),
            1,
            f"GPU breakpoints with the settings of {breakpoint}: "
            f"{[str(copy) for copy in copies]}",
        )
        return copies[0]

    def run_to_gpu_breakpoint(
        self, target: lldb.SBTarget, breakpoint: lldb.SBBreakpoint
    ) -> lldb.SBBreakpoint:
        """Launch the program of `target`, on which the test set `breakpoint`
        before the run, and wait for the GPU's first stop. Return the GPU
        target's copy of the breakpoint. This is what a user does with
        `breakpoint set` and `run`; see launch_and_wait_for_gpu_target for when
        a test needs it instead of run_to_cpu_line."""
        # The copy keeps the original's names, so a name of its own finds it
        # even beside an identical breakpoint.
        self.assertSuccess(
            breakpoint.AddNameWithErrorHandling(
                f"test_breakpoint_{breakpoint.GetID()}"
            ),
            f"naming {breakpoint}",
        )
        self.launch_and_wait_for_gpu_target(target)
        copy = self.gpu_breakpoint_copy(breakpoint)
        self.wait_for_gpu_to_stop()
        return copy

    def run_to_gpu_line(
        self,
        source: str,
        marker: str,
        exe_name: Optional[str] = None,
        target: Optional[lldb.SBTarget] = None,
        condition: Optional[str] = None,
    ) -> lldb.SBBreakpoint:
        """Set a breakpoint on the line of `source` that contains `marker`,
        with `condition` if one is given, before the run, and run the program
        to the GPU's first stop with run_to_gpu_breakpoint. Return the GPU
        target's copy of the breakpoint. `exe_name` and `target` are as for
        run_to_cpu_line.

        Use it for a line the CPU never runs, such as one in a __device__
        function. lldb also resolves a line of a kernel in the kernel's
        host-side stub, which stops the CPU after it launches the kernel.
        """
        # TODO: lldb should not stop the CPU in a kernel's host-side stub for a
        # breakpoint on a line of the kernel, as cuda-gdb does not. Once it
        # does not, a kernel's line works here too: drop the note above and
        # nvgpu-support.rst's advice to mark a line the CPU never runs.
        if target is None:
            exe_name = exe_name or "a.out"
            target = lldbutil.run_to_breakpoint_make_target(
                self, exe_name, in_cwd=not os.path.isabs(exe_name)
            )
        else:
            self.assertIsNone(exe_name, "pass exe_name or target, not both")
        breakpoint = target.BreakpointCreateByLocation(
            source, line_number(source, marker)
        )
        if condition:
            breakpoint.SetCondition(condition)
        return self.run_to_gpu_breakpoint(target, breakpoint)

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
        keep_cpu_running: bool = False,
    ) -> int:
        """Pull the next state-changed event of `process` and return its state.

        Only events from that process's broadcaster are consumed, so the other
        process's events stay queued. Stopped events flagged as "restarted" (a
        stop lldb resumed from on its own) are skipped the way
        lldbutil.expect_state_changes does. Fails the test on timeout, naming
        `what` ("CPU" or "GPU"). With `keep_cpu_running`, the CPU's events are
        consumed too.
        """
        # SBListener waits take a uint32 second count; round fractions up once here.
        num_seconds = math.ceil(timeout_seconds)
        broadcaster = process.GetBroadcaster()
        while True:
            event = lldb.SBEvent()
            if not self._wait_for_event(
                listener, broadcaster, num_seconds, event, keep_cpu_running
            ):
                self.fail(
                    f"timed out after {timeout_seconds} s waiting for the {what} "
                    "process to change state"
                    f"{self._describe_other_process(process)}"
                )
            state = lldb.SBProcess.GetStateFromEvent(event)
            if state == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(
                event
            ):
                continue
            return state

    def _wait_for_event(
        self,
        listener: lldb.SBListener,
        broadcaster: lldb.SBBroadcaster,
        num_seconds: int,
        event: lldb.SBEvent,
        keep_cpu_running: bool,
    ) -> bool:
        """Wait up to `num_seconds` for a state-changed event from `broadcaster`.

        With `keep_cpu_running`, wait in one-second slices, and resume the CPU
        if it sat in a target-sync stop for a whole slice with no GPU event.
        That happens when lldb resumed the GPU from a stop on its own: the
        CPU's stop then has no GPU stop to go with it.

        Don't resume the CPU sooner, or while a GPU event is queued. Its stop
        may belong to the GPU stop this wait is about to return, since the two
        events can arrive in either order, and resuming it would leave the CPU
        running past that stop. That can still happen when the GPU stops just
        as a slice ends, before its event reaches the listener.
        """
        mask = lldb.SBProcess.eBroadcastBitStateChanged
        if not keep_cpu_running:
            return listener.WaitForEventForBroadcasterWithType(
                num_seconds, broadcaster, mask, event
            )
        deadline = time.monotonic() + max(num_seconds, 1)
        cpu_was_held = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if cpu_was_held and not listener.PeekAtNextEventForBroadcasterWithType(
                broadcaster, mask, lldb.SBEvent()
            ):
                self._keep_cpu_running(math.ceil(remaining))
            cpu_was_held = self._cpu_held_for_target_sync()
            if listener.WaitForEventForBroadcasterWithType(1, broadcaster, mask, event):
                return True

    def cpu_stops_with_gpu(self) -> bool:
        """Whether lldb-server stops the CPU when the GPU stops. It does unless
        NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP is "1" in this process's
        environment, which lldb-server inherits."""
        return os.environ.get("NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP") != "1"

    def pull_state(self, process: lldb.SBProcess) -> int:
        """Pull the queued state-changed events of `process` (CPU or GPU)
        without waiting, and return its latest state. A restarted stop counts
        as running.

        GetState() is only updated as events are pulled, so until then it can
        be out of date. SBProcess.Stop() fails unless the process looks
        running, so pull the GPU's events before interrupting it.
        """
        listener = self.dbg.GetListener()
        broadcaster = process.GetBroadcaster()
        state = process.GetState()
        event = lldb.SBEvent()
        while listener.GetNextEventForBroadcasterWithType(
            broadcaster, lldb.SBProcess.eBroadcastBitStateChanged, event
        ):
            state = lldb.SBProcess.GetStateFromEvent(event)
            if state == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(
                event
            ):
                state = lldb.eStateRunning
        return state

    def _cpu_stopped_only_for_target_sync(self) -> bool:
        """Whether the stopped CPU is in a target-sync stop only: some thread
        reports target sync and none stopped for anything else."""
        reasons = {thread.GetStopReason() for thread in self.cpu_process.threads}
        return lldb.eStopReasonTargetSync in reasons and reasons <= {
            lldb.eStopReasonNone,
            lldb.eStopReasonTargetSync,
        }

    def _cpu_held_for_target_sync(self) -> bool:
        """Pull the CPU's queued events, then return whether it is in a
        target-sync stop only."""
        return (
            bool(self.cpu_process)
            and self.pull_state(self.cpu_process) == lldb.eStateStopped
            and self._cpu_stopped_only_for_target_sync()
        )

    def _settle_cpu(self, timeout_seconds: int) -> None:
        """After a GPU stop, wait up to `timeout_seconds` for the CPU to stop
        too, and pull its stop event. Not every GPU stop stops the CPU, so
        running out of time is not a failure.

        Without this, a CPU that has stopped can still look running. Teardown
        then interrupts it and waits for a stop that never comes, for the
        whole target.process.interrupt-timeout.
        """
        listener = self.dbg.GetListener()
        broadcaster = self.cpu_process.GetBroadcaster()
        deadline = time.monotonic() + timeout_seconds
        state = self.pull_state(self.cpu_process)
        while state == lldb.eStateRunning:
            remaining = deadline - time.monotonic()
            event = lldb.SBEvent()
            if remaining <= 0 or not listener.WaitForEventForBroadcasterWithType(
                math.ceil(remaining),
                broadcaster,
                lldb.SBProcess.eBroadcastBitStateChanged,
                event,
            ):
                return
            state = lldb.SBProcess.GetStateFromEvent(event)
            if state == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(
                event
            ):
                state = lldb.eStateRunning

    def _keep_cpu_running(self, timeout_seconds: int) -> None:
        """Resume the CPU if it is in a target-sync stop only. This doesn't
        wait for a stop that hasn't arrived yet; a later call resumes it."""
        if not self._cpu_held_for_target_sync():
            return
        # A continue in synchronous mode would wait for the next stop and
        # take its event, which the wait below expects.
        self.setAsync(True)
        cpu_process = self.cpu_process
        self.assertSuccess(cpu_process.Continue(), "continue CPU process")
        # Pull the running event now. Otherwise the CPU still looks stopped,
        # and the next call would try to resume it again.
        state = self._next_state(
            cpu_process, "CPU", self.dbg.GetListener(), timeout_seconds
        )
        if state != lldb.eStateRunning:
            self.fail(
                f"CPU process entered state {lldb.SBDebugger.StateAsCString(state)} "
                "instead of running"
            )

    def _wait_for_state(
        self,
        process: lldb.SBProcess,
        wanted: int,
        doing: str,
        timeout_seconds: int,
        keep_cpu_running: bool = False,
    ) -> None:
        """Pull the state-changed events of `process` until it enters `wanted`.

        Running events (and restarted stops) are skipped. With
        `keep_cpu_running`, the CPU's target-sync stops are resumed and skipped
        too. Any other state fails the test, saying what the process did before
        `doing` ("stopping", "exiting"): with the exit status and its
        description if it exited, so that a lost connection to lldb-server
        (status -1) is told apart from the program ending. A program that never
        gets where the test expects fails instead of hanging the test.
        """
        what = "GPU" if process.GetTarget() == self.gpu_target else "CPU"
        listener = self.dbg.GetListener()
        while True:
            state = self._next_state(
                process,
                what,
                listener,
                timeout_seconds,
                keep_cpu_running and what == "GPU",
            )
            if (
                keep_cpu_running
                and what == "CPU"
                and state == lldb.eStateStopped
                and self._cpu_stopped_only_for_target_sync()
            ):
                # The CPU stopped only because the GPU did.
                self._keep_cpu_running(timeout_seconds)
                continue
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
                if state == lldb.eStateStopped:
                    entered += self._describe_stop(process)
            self.fail(f"{what} process {entered} before {doing}")

    @staticmethod
    def _describe_stop(process: lldb.SBProcess) -> str:
        """Say why and where the stopped `process` stopped, for a failure
        message: the stop description and function of its first thread with
        a stop reason, or nothing if none has one."""
        for thread in process.threads:
            if thread.GetStopReason() in (
                lldb.eStopReasonNone,
                lldb.eStopReasonInvalid,
            ):
                continue
            return (
                f" ({thread.GetStopDescription(256)} "
                f"in {thread.frame[0].GetFunctionName()})"
            )
        return ""

    def _describe_other_process(self, process: lldb.SBProcess) -> str:
        """Say what the other process, the CPU's for the GPU and the GPU's for
        the CPU, is doing, for the message of a wait on `process` that timed
        out: the usual cause is that the other one holds it up, stopped at a
        breakpoint of its own. This pulls the other process's events, which
        only a test that is failing anyway may do."""
        if process.GetTarget() == self.gpu_target:
            other, what = self.cpu_process, "CPU"
        else:
            other, what = self.gpu_process, "GPU"
        if not other:
            return ""
        state = self.pull_state(other)
        if state == lldb.eStateExited:
            return f"; the {what} process exited (status {other.GetExitStatus()})"
        doing = lldb.SBDebugger.StateAsCString(state)
        if state == lldb.eStateStopped:
            doing += self._describe_stop(other)
        return f"; the {what} process is {doing}"

    def wait_for_stop(self, process: lldb.SBProcess, timeout_seconds: int = 60) -> None:
        """Wait for `process` (CPU or GPU), which is running, to stop.

        Nothing is resumed, except that a wait on the CPU resumes it from
        target-sync stops, since the test wants a stop of the CPU's own.
        """
        keep_cpu_running = (
            process.GetTarget() != self.gpu_target and self.cpu_stops_with_gpu()
        )
        self._wait_for_state(
            process, lldb.eStateStopped, "stopping", timeout_seconds, keep_cpu_running
        )

    def wait_for_exit(self, process: lldb.SBProcess, timeout_seconds: int = 60) -> int:
        """Wait for `process` (CPU or GPU) to exit and return its exit status.

        The test resumes the processes first; a CPU in a target-sync stop is
        resumed here. Any other stop fails the test, so a program that stops
        again instead of finishing fails instead of hanging.
        """
        keep_cpu_running = self.cpu_stops_with_gpu()
        # An earlier wait on the GPU may already have pulled the CPU's exit.
        if keep_cpu_running and process.GetState() == lldb.eStateExited:
            return process.GetExitStatus()
        if keep_cpu_running and process.GetTarget() == self.gpu_target:
            # The GPU is already resumed, so the CPU's target-sync stop is from
            # an earlier GPU stop and safe to resume now, which saves a slice in
            # _wait_for_event. Not before a wait on the CPU: this pulls the
            # CPU's events, and that wait needs its exit event.
            self._keep_cpu_running(timeout_seconds)
        self._wait_for_state(
            process, lldb.eStateExited, "exiting", timeout_seconds, keep_cpu_running
        )
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

    def continue_gpu_to_exit(
        self, status: int = 0, keep_breakpoints: bool = False
    ) -> None:
        """Let the program run to its end: delete the GPU target's
        breakpoints, unless `keep_breakpoints` is set, resume the GPU, and
        check that both processes exit with `status`. With the breakpoints
        kept, a stop at one of them fails the test."""
        if not keep_breakpoints:
            self.gpu_target.DeleteAllBreakpoints()
        self.setAsync(True)
        self.assertSuccess(self.gpu_process.Continue(), "continue GPU process")
        self.assert_exit_status(status)

    @staticmethod
    def _drain(read: Callable[[int], str]) -> str:
        """Return everything that `read`, GetSTDOUT or GetSTDERR of a process,
        has that the test has not read yet."""
        chunks = []
        while chunk := read(4096):
            chunks.append(chunk)
        return "".join(chunks)

    def read_stdout(self) -> str:
        """Return what the CPU process wrote to stdout that the test has not
        read yet. lldb collects it on a thread of its own, so output the
        program has just written may not be here yet."""
        return self._drain(self.cpu_process.GetSTDOUT)

    def read_stderr(self) -> str:
        """Return what the CPU process wrote to stderr that the test has not
        read yet, as read_stdout does for stdout."""
        return self._drain(self.cpu_process.GetSTDERR)

    def assert_stdout_contains(self, *texts: str) -> None:
        """What the CPU process has written to stdout since the test last read
        it must contain each of `texts`. A failure shows the whole output,
        which assertIn on its own cuts to 80 characters."""
        stdout = self.read_stdout()
        for text in texts:
            self.assertIn(text, stdout, f"in the program's output:\n{stdout}")

    def wait_for_gpu_to_stop(
        self, timeout_seconds: int = 60, *, keep_cpu_running: bool = False
    ) -> None:
        """Wait for the running GPU process to stop.

        Nothing is resumed, unless `keep_cpu_running` is set: then a CPU left
        in a target-sync stop with no GPU stop to go with it is resumed (see
        _wait_for_event).
        """
        self._wait_for_state(
            self.gpu_process,
            lldb.eStateStopped,
            "stopping",
            timeout_seconds,
            keep_cpu_running and self.cpu_stops_with_gpu(),
        )

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

        A CPU in a target-sync stop is resumed along with the GPU. After each
        GPU stop, this waits briefly for the CPU's matching stop, so that both
        processes are stopped when it returns.
        """
        self.setAsync(True)
        keep_cpu_running = self.cpu_stops_with_gpu()
        for _ in range(max_continues):
            if keep_cpu_running:
                self._keep_cpu_running(timeout_seconds)
            error = self.gpu_process.Continue()
            self.assertSuccess(error, "continue GPU process")
            self.wait_for_gpu_to_stop(
                timeout_seconds, keep_cpu_running=keep_cpu_running
            )
            if keep_cpu_running:
                self._settle_cpu(self.CPU_STOP_GRACE_SECONDS)
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

    def for_each_thread_at_breakpoint(
        self,
        breakpoint: lldb.SBBreakpoint,
        count: int,
        check: Optional[Callable[[lldb.SBThread], None]] = None,
    ) -> Set[ThreadCoords]:
        """Run `check` on each thread that the current GPU stop has at
        `breakpoint`, then resume the GPU until `count` threads in all have
        stopped there, running `check` on each as it stops. Return the
        threads' thread_coords.

        The threads that reach one breakpoint do not always stop at once: a
        warp can arrive in a stop of its own. A thread that stops there twice
        fails the test, since the program then went past where the test
        expected every thread to stop.
        """
        seen = set()
        stopped = self.assert_gpu_stop_at_breakpoint(breakpoint)
        while True:
            for thread in stopped:
                coords = self.thread_coords(thread)
                self.assertNotIn(
                    coords, seen, f"{thread.GetName()} stopped at {breakpoint} twice"
                )
                seen.add(coords)
                if check:
                    check(thread)
            if len(seen) >= count:
                return seen
            stopped = self.continue_gpu_until_breakpoint(breakpoint)

    def step_over_gpu_thread(
        self, thread: lldb.SBThread, expected_line: int, timeout_seconds: int = 30
    ) -> None:
        """Step `thread` over its source line, as `next` does, and wait for
        the GPU to stop. The step must complete with `thread` at
        `expected_line`."""
        step = f"{thread.GetName()} stepping over line "
        step += str(thread.frame[0].GetLineEntry().GetLine())
        # A step in synchronous mode would wait for the stop and take its
        # event, which the wait below expects.
        self.setAsync(True)
        error = lldb.SBError()
        thread.StepOver(lldb.eOnlyDuringStepping, error)
        self.assertSuccess(error, step)
        try:
            self.wait_for_gpu_to_stop(timeout_seconds)
        except self.failureException as failure:
            raise self.failureException(f"{step}: {failure}") from None
        self.assertStopReason(
            thread.GetStopReason(),
            lldb.eStopReasonPlanComplete,
            f"stop reason of {step}",
        )
        self.assertEqual(
            thread.frame[0].GetLineEntry().GetLine(),
            expected_line,
            f"line of {step}",
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

    @staticmethod
    def _describe_frame(frame: lldb.SBFrame) -> str:
        """Name `frame` for a failure message: its index, its function, and its
        thread, whose name gives a GPU thread's CUDA coordinates."""
        thread = frame.GetThread()
        thread_name = thread.GetName() or f"thread #{thread.GetIndexID()}"
        return (
            f"frame #{frame.GetFrameID()} ({frame.GetFunctionName()}) of {thread_name}"
        )

    def read_int_variable(self, frame: lldb.SBFrame, path: str) -> int:
        """Read the integer variable, or member or element of one, at `path`
        in `frame`, as `frame variable` does; fail if lldb cannot read it. To
        check one value, use assert_int_variable."""
        value = frame.GetValueForVariablePath(path)
        self.assertSuccess(
            value.GetError(), f"reading {path} in {self._describe_frame(frame)}"
        )
        return value.GetValueAsSigned()

    def assert_int_variable(
        self, frame: lldb.SBFrame, path: str, expected: int
    ) -> None:
        """The integer variable, or member or element of one, at `path` in
        `frame` must have the value `expected`. A failure names the variable,
        the frame and its thread."""
        self.assertEqual(
            self.read_int_variable(frame, path),
            expected,
            f"{path} in {self._describe_frame(frame)}",
        )

    def value_as_float(self, value: lldb.SBValue, what: str) -> float:
        """Return the number in `value`, a float or a double, which `what`
        names in a failure message."""
        self.assertIn(
            value.GetType().GetCanonicalType().GetBasicType(),
            (lldb.eBasicTypeFloat, lldb.eBasicTypeDouble),
            f"{what} is a float or a double, not {value.GetTypeName()}",
        )
        error = lldb.SBError()
        data = value.GetData()
        if value.GetByteSize() == 4:
            number = data.GetFloat(error, 0)
        else:
            number = data.GetDouble(error, 0)
        self.assertSuccess(error, f"reading {what}")
        return number

    def read_float_variable(self, frame: lldb.SBFrame, path: str) -> float:
        """Read the floating-point variable, or member or element of one, at
        `path` in `frame`, as `frame variable` does; fail if lldb cannot read
        it. To check one value, use assert_float_variable."""
        what = f"{path} in {self._describe_frame(frame)}"
        value = frame.GetValueForVariablePath(path)
        self.assertSuccess(value.GetError(), f"reading {what}")
        return self.value_as_float(value, what)

    def assert_float_variable(
        self, frame: lldb.SBFrame, path: str, expected: float, places: int = 6
    ) -> None:
        """The floating-point variable, or member or element of one, at `path`
        in `frame` must be `expected`, to `places` decimal places. A failure
        names the variable, the frame and its thread."""
        self.assertAlmostEqual(
            self.read_float_variable(frame, path),
            expected,
            places=places,
            msg=f"{path} in {self._describe_frame(frame)}",
        )

    def evaluate_expression(self, frame: lldb.SBFrame, expression: str) -> lldb.SBValue:
        """Evaluate `expression` in `frame`, as `expr` does, and return its
        value; fail, naming the expression, the frame and its thread, if lldb
        cannot evaluate it."""
        value = frame.EvaluateExpression(expression)
        self.assertSuccess(
            value.GetError(),
            f"evaluating {expression} in {self._describe_frame(frame)}",
        )
        return value

    def collect_errors(self) -> Callable[[], List[str]]:
        """Collect the errors that lldb reports outside a command from now
        on, such as a breakpoint condition that it cannot evaluate, and return
        a function that returns them, oldest first. From now on lldb prints
        none of these errors, so check them: the ones the test never asked
        for are printed when it ends, so that an unrelated error, such as a
        lost connection, is not lost with them.

        lldb prints an error that nothing listens for in the test's output. A
        condition that fails in every thread of a kernel prints an error for
        each thread at each stop: thousands of lines, which bury the failure
        when lit shows the output."""
        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitError
        )
        errors: List[str] = []
        checked = 0

        def collected() -> List[str]:
            nonlocal checked
            event = lldb.SBEvent()
            while listener.GetNextEventForBroadcaster(broadcaster, event):
                diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
                errors.append(diagnostic.GetValueForKey("message").GetStringValue(4096))
            checked = len(errors)
            return errors

        def print_unchecked() -> None:
            first = checked
            unchecked = collected()[first:]
            if unchecked:
                limit = 20
                print(
                    f"{len(unchecked)} errors lldb reported that the test did not "
                    f"check, the first {min(limit, len(unchecked))}:\n"
                    + "\n".join(unchecked[:limit])
                )

        self.addTearDownHook(print_unchecked)
        return collected

    def read_gpu_memory(
        self,
        address: int,
        space: str,
        size: int,
        thread: Optional[lldb.SBThread] = None,
    ) -> bytes:
        """Read `size` bytes at `address` in the GPU address space named
        `space`, such as "global", "shared" or "generic", as `memory read -p
        <space>` does. lldb asks lldb-server for them on every read. A shared,
        local or generic address means different memory to different blocks
        or threads: pass the `thread` to read it for."""
        error = lldb.SBError()
        space_id = self.gpu_process.GetAddressSpaceID(space, error)
        self.assertSuccess(error, f"the {space} address space")
        if thread is None:
            location = lldb.SBProcessAddress(address, space_id)
        else:
            self.assertTrue(thread.IsValid(), f"the thread to read {space} memory for")
            location = lldb.SBProcessAddress(address, space_id, thread)
        data = self.gpu_process.ReadMemory(location, size, error)
        self.assertSuccess(
            error, f"reading {size} bytes of {space} memory at {address:#x}"
        )
        return data

    def select_gpu_thread(
        self,
        blockidx: Optional[Sequence[int]] = None,
        threadidx: Optional[Sequence[int]] = None,
    ) -> lldb.SBThread:
        """Select a GPU thread by its CUDA block index and thread index, as
        `thread select --blockidx --threadidx` does, and return it. Each index
        is one to three coordinates, x first, and a coordinate left out
        matches any. Without `blockidx`, the thread is in the block of the
        selected thread; without `threadidx`, it has the selected thread's
        thread index. The GPU target becomes the selected target, as after the
        command."""
        self.assertTrue(blockidx or threadidx, "select_gpu_thread needs an index")
        command = "thread select"
        if blockidx:
            command += " --blockidx " + ",".join(str(i) for i in blockidx)
        if threadidx:
            command += " --threadidx " + ",".join(str(i) for i in threadidx)
        self.dbg.SetSelectedTarget(self.gpu_target)
        self.runCmd(command)
        selected = self.gpu_process.GetSelectedThread()
        for wanted, actual, what in zip(
            (blockidx, threadidx), self.thread_coords(selected), ("block", "thread")
        ):
            if wanted:
                self.assertEqual(
                    actual[: len(wanted)],
                    tuple(wanted),
                    f"{what} index of the thread `{command}` selected",
                )
        return selected

    def _thread_info_int(self, thread: lldb.SBThread, path: str) -> int:
        """Return the integer at `path`, such as "cuda.lane", in what
        lldb-server reports for a GPU thread."""
        stream = lldb.SBStream()
        self.assertTrue(
            thread.GetInfoItemByPathAsString(path, stream),
            f"{thread.GetName()} has no {path}",
        )
        # The SB API writes integers in hex, as "0x1f". Base 0 reads that and a
        # decimal alike, instead of misreading a decimal as hex.
        return int(stream.GetData(), 0)

    def thread_coords(self, thread: lldb.SBThread) -> ThreadCoords:
        """Return a GPU thread's (blockIdx, threadIdx), each as (x, y, z), from
        the CUDA coordinates lldb-server reports for it. Tell threads apart by
        these, not by their names, which are for display."""
        block = tuple(
            self._thread_info_int(thread, f"cuda.block_idx.{axis}") for axis in "xyz"
        )
        index = tuple(
            self._thread_info_int(thread, f"cuda.thread_idx.{axis}") for axis in "xyz"
        )
        return block, index

    def thread_physical_coords(
        self, thread: lldb.SBThread
    ) -> Tuple[int, int, int, int]:
        """Return where a GPU thread runs, as lldb-server reports it: its
        device, its SM, its warp on that SM and its lane in the warp. CUDA
        fixes a thread's warp and lane in its block by the thread's index, but
        not which of the SM's warps runs that warp, and a warp can move to
        another during execution (the PTX ISA says so of %warpid): compare
        SMs and warps only within one stop."""
        return tuple(
            self._thread_info_int(thread, f"cuda.{item}")
            for item in ("device", "sm", "warp", "lane")
        )
