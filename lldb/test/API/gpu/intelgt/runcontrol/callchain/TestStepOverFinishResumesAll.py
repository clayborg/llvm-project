"""
Test the observed behaviour of finish (step-out) and step-over-a-call when
multiple EU hardware threads are stopped at a breakpoint: BOTH the focus EU
thread and its sibling EU threads advance (unlike a single instruction step,
which advances only the stepped EU thread — see runcontrol/TestSchedulerLocking).

This asserts the OBSERVED end state only. The mechanism lives in core LLDB and
is fully traced (gdb-remote packets confirm the GPU target receives per-thread
vCont;s steps for EACH stopped EU thread, then a continue, during the finish):

  - finish/step-out is created with stop_others=false
    (CommandObjectThread.cpp: bool_stop_other_threads = (m_step_type !=
    eStepTypeOut)), so ThreadPlanStepOut::StopOthers() == false.
  - With no plan demanding StopOthers, ThreadList::WillResume leaves
    thread_to_run == null and takes the "everybody runs" branch, which calls
    SetupToStepOverBreakpointIfNeeded on EVERY non-suspended thread.
  - Both EU threads sit on the SAME breakpoint instruction, so each gets a
    private ThreadPlanStepOverBreakpoint, whose GetPlanRunState() == eStateStepping
    => a vCont;s:TID for each. (Clearing the bp instruction is a correctness
    requirement, independent of RunMode/scheduler-locking; the plan's
    SupportsResumeOthers() == false.)
  - Then the StepOut plan's own continue leg (GetPlanRunState() == eStateRunning)
    runs the device to A's return address -> the $c.

By contrast stepi builds a StopOthers()==true plan, so thread_to_run IS set and
only that one EU thread is set up; siblings are suspended (see
runcontrol/TestSchedulerLocking). That asymmetry is core-LLDB behaviour, not a
plugin policy. We assert only the end state so a future change is noticed.

Kernel (callchain_1d.cpp): a 32-work-item parallel_for that calls mid()->leaf()
(both noinline), so the hardware uses >= 2 EU threads and there is a real call
stack to step over / finish out of.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtStepOverFinishResumesAllTestCase(IntelGtTestCaseBase):
    """step-over-a-call and finish resume all EU threads."""

    SOURCE = "callchain_1d.cpp"
    EXE = "callchain_1d"

    @staticmethod
    def _eu_of(t):
        return t.GetThreadID() & ~0xFF

    def _launch_at(self, pattern):
        self.build()
        exe = self.getBuildArtifact(self.EXE)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")
        bp = target.BreakpointCreateBySourceRegex(
            pattern, lldb.SBFileSpec(self.SOURCE))
        self.assertGreater(bp.GetNumLocations(), 0,
                           f"breakpoint '{pattern}' should resolve")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        launch_info.SetEnvironmentEntries(
            [f"{k}={v}" for k, v in os.environ.items()], True)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(),
                        f"Launch should succeed: {error.GetCString()}")
        self._wait_stopped(process)
        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target expected")
        return self.dbg.GetTargetAtIndex(1).GetProcess()

    def _wait_stopped(self, process, timeout=30):
        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(timeout, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()
        return state

    def _two_eu_focus(self, gpu):
        """Return (a, b): the breakpoint-reporting lane of two distinct EU
        threads, asserting there are at least two."""
        groups = {}
        for t in gpu.threads:
            if t.GetThreadID() == 1:
                continue
            groups.setdefault(self._eu_of(t), []).append(t)
        self.assertGreaterEqual(
            len(groups), 2,
            f"need >= 2 EU threads; saw {len(groups)}")
        bases = sorted(groups)

        def focus(group):
            return next((t for t in group
                         if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                        group[0])
        return focus(groups[bases[0]]), focus(groups[bases[1]])

    def test_finish_resumes_all_eu_threads(self):
        """finish (step-out) advances the other EU thread too."""
        gpu = self._launch_at("leaf-line")
        a, b = self._two_eu_focus(gpu)
        a_tid, b_tid = a.GetThreadID(), b.GetThreadID()
        a_line0 = a.GetFrameAtIndex(0).GetLineEntry().GetLine()
        b_pc0 = b.GetFrameAtIndex(0).GetPC()

        gpu.SetSelectedThread(a)
        error = lldb.SBError()
        a.StepOut(error)
        self.assertTrue(error.Success(), f"finish failed: {error.GetCString()}")
        self._wait_stopped(gpu)

        a2 = gpu.GetThreadByID(a_tid)
        b2 = gpu.GetThreadByID(b_tid)
        self.assertTrue(a2.IsValid() and b2.IsValid())
        # A finished out of leaf() into mid().
        self.assertNotEqual(a2.GetFrameAtIndex(0).GetLineEntry().GetLine(),
                            a_line0, "finished EU thread A should leave leaf()")
        # B advanced too (finish ran the device to A's return address).
        self.assertNotEqual(
            b2.GetFrameAtIndex(0).GetPC(), b_pc0,
            "finish should also advance sibling EU thread B (device resumes)")

    def test_step_over_call_keeps_sibling_eu_thread(self):
        """step-over a function call keeps sibling EU thread B frozen.

        With eStopReasonBreakpoint on all lanes and eOnlyDuringStepping (stop_others=True),
        only the stepped EU thread (A) advances through the call. Sibling EU thread B
        stays frozen — consistent with scheduler-locking on the GPU. finish (step-out)
        still advances B (see test_finish_resumes_all_eu_threads), because finish uses
        stop_others=False."""
        gpu = self._launch_at("call-line")
        a, b = self._two_eu_focus(gpu)
        a_tid, b_tid = a.GetThreadID(), b.GetThreadID()
        a_pc0 = a.GetFrameAtIndex(0).GetPC()
        b_pc0 = b.GetFrameAtIndex(0).GetPC()

        gpu.SetSelectedThread(a)
        error = lldb.SBError()
        a.StepOver(lldb.eOnlyDuringStepping, error)
        self.assertTrue(error.Success(),
                        f"step-over failed: {error.GetCString()}")
        self._wait_stopped(gpu)

        a2 = gpu.GetThreadByID(a_tid)
        b2 = gpu.GetThreadByID(b_tid)
        self.assertTrue(a2.IsValid() and b2.IsValid())
        self.assertNotEqual(a2.GetFrameAtIndex(0).GetPC(), a_pc0,
                            "stepped EU thread A should advance")
        self.assertEqual(
            b2.GetFrameAtIndex(0).GetPC(), b_pc0,
            "sibling EU thread B should NOT advance under eOnlyDuringStepping StepOver "
            "(stop_others=True keeps B frozen)")
