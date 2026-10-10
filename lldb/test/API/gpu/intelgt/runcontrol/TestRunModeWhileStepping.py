"""
Test how the stepping run modes affect GPU sibling EU threads on StepOver.

LLDB exposes three run modes for stepping plans (RunMode in
lldb-enumerations.h): eOnlyThisThread (this-thread), eAllThreads (all-threads)
and eOnlyDuringStepping (while-stepping).

OBSERVED behaviour on IntelGT (verified, not assumed): a source-level StepOver
of a non-call line advances the SIBLING EU thread too, in ALL THREE run modes --
including this-thread. The run mode is effectively not honoured for StepOver on
the GPU. The reason is the forced step-over-breakpoint: when multiple EU threads
sit on the same breakpoint, core LLDB must single-step every one of them off the
breakpoint instruction before the process can make progress (see
runcontrol/callchain/TestStepOverFinishResumesAll and the scheduler-locking
notes). That step-over-breakpoint is independent of run mode.

Contrast: a pure instruction step (StepInstruction / stepi) DOES isolate to the
stepped EU thread -- that is verified in runcontrol/TestSchedulerLocking. So the
isolation difference on the GPU is StepInstruction vs StepOver, not the run-mode
argument.

This test pins that observed behaviour for all three run modes so a future
change to run-mode handling is noticed. Kernel: parallel_1d.cpp (32 work-items)
is split across >= 2 EU threads.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtRunModeWhileSteppingTestCase(RunControlTestCaseBase):
    """StepOver advances siblings in every run mode; only stepi isolates."""

    def _two_eu_focus(self, lanes):
        """Return (a, b): the breakpoint-reporting lane of two distinct EU
        threads, asserting there are at least two."""
        groups = {}
        for t in lanes:
            groups.setdefault(self.eu_of(t), []).append(t)
        self.assertGreaterEqual(
            len(groups), 2,
            f"need >= 2 EU threads; saw {len(groups)} "
            f"(bases {sorted(hex(b) for b in groups)})")
        bases = sorted(groups)

        def focus(group):
            return next((t for t in group
                         if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                        group[0])
        return focus(groups[bases[0]]), focus(groups[bases[1]])

    def _step_over_in_mode(self, run_mode, mode_name):
        """StepOver EU thread A in *run_mode*; return (a_advanced, b_advanced)."""
        gpu_process, lanes = self.launch_and_stop("first-line")
        a, b = self._two_eu_focus(lanes)
        a_tid, b_tid = a.GetThreadID(), b.GetThreadID()
        line_a0 = a.GetFrameAtIndex(0).GetLineEntry().GetLine()
        pc_b0 = b.GetFrameAtIndex(0).GetPC()
        self.assertGreater(line_a0, 0, "EU thread A should have a source line")

        gpu_process.SetSelectedThread(a)
        error = lldb.SBError()
        a.StepOver(run_mode, error)
        self.assertTrue(error.Success(),
                        f"StepOver({mode_name}) should succeed: "
                        f"{error.GetCString()}")
        self._wait_stopped(gpu_process)

        a_after = gpu_process.GetThreadByID(a_tid)
        b_after = gpu_process.GetThreadByID(b_tid)
        self.assertTrue(a_after.IsValid() and b_after.IsValid())
        a_advanced = (a_after.GetFrameAtIndex(0).GetLineEntry().GetLine()
                      > line_a0)
        b_advanced = (b_after.GetFrameAtIndex(0).GetPC() != pc_b0)
        return a_advanced, b_advanced

    def test_while_stepping_step_over_advances_sibling(self):
        """while-stepping: StepOver of a non-call line keeps sibling EU thread B put.

        With eStopReasonBreakpoint on all lanes, eOnlyDuringStepping (stop_others=True)
        correctly keeps sibling EU thread B frozen — only the stepped EU thread advances.
        This is analogous to how stepi isolates (see TestSchedulerLocking).
        The eAllThreads case where B also advances is verified in
        test_all_threads_step_over_advances_sibling."""
        a_adv, b_adv = self._step_over_in_mode(
            lldb.eOnlyDuringStepping, "while-stepping")
        self.assertTrue(a_adv, "stepped EU thread A should advance")
        self.assertFalse(
            b_adv,
            "sibling EU thread B should NOT advance under while-stepping "
            "(stop_others=True keeps sibling EU thread frozen)")

    def test_this_thread_step_over_keeps_sibling_put(self):
        """this-thread: StepOver keeps sibling EU thread B frozen.

        With eStopReasonBreakpoint on all lanes, eOnlyThisThread (stop_others=True)
        correctly keeps sibling EU thread B frozen. Only eAllThreads lets siblings
        advance on StepOver."""
        a_adv, b_adv = self._step_over_in_mode(
            lldb.eOnlyThisThread, "this-thread")
        self.assertTrue(a_adv, "stepped EU thread A should advance")
        self.assertFalse(
            b_adv,
            "sibling EU thread B should NOT advance under this-thread StepOver "
            "(stop_others=True keeps sibling EU thread frozen)")

    def test_all_threads_step_over_advances_sibling(self):
        """all-threads: StepOver advances the sibling EU thread (default)."""
        a_adv, b_adv = self._step_over_in_mode(
            lldb.eAllThreads, "all-threads")
        self.assertTrue(a_adv, "stepped EU thread A should advance")
        self.assertTrue(b_adv, "sibling EU thread B advances under all-threads")
