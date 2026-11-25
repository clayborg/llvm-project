"""
Test stepping a NON-focus (sibling) SIMD lane of a stopped EU thread.

When an EU thread stops at a breakpoint, all active lanes of the same EU thread
report eStopReasonBreakpoint — they are genuinely all stopped at the same
instruction in lockstep. Previously only the focus lane reported
eStopReasonBreakpoint while siblings got eStopReasonNone, which caused
Thread::ShouldStop() to bail out before consulting the step plan (it returns
false immediately for threads with no stop reason), so stepping a sibling lane
resumed the whole process instead of stepping that lane.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtStepSiblingLaneTestCase(RunControlTestCaseBase):
    """Stepping a sibling (eStopReasonBreakpoint) lane completes and is
    attributed to that lane."""

    def _focus_and_sibling(self, lanes):
        """Return (focus_thread, sibling_thread) on the same EU thread.
        Both report eStopReasonBreakpoint (all lanes stopped at the same
        breakpoint in lockstep). Focus is the one with the lowest lane id."""
        focus = next((t for t in lanes
                      if t.GetStopReason() == lldb.eStopReasonBreakpoint), None)
        self.assertIsNotNone(focus, "a lane should report the breakpoint")
        eu = self.eu_of(focus)
        sibling = next(
            (t for t in lanes
             if self.eu_of(t) == eu
             and t.GetThreadID() != focus.GetThreadID()
             and t.GetStopReason() == lldb.eStopReasonBreakpoint),
            None)
        return focus, sibling

    def test_all_lanes_report_breakpoint(self):
        """All active lanes of a stopped EU thread report eStopReasonBreakpoint.

        They stopped together in lockstep at the same breakpoint instruction.
        Reporting a real stop reason on all lanes (rather than eStopReasonNone
        on siblings) allows Thread::ShouldStop to engage the step plan when
        the user selects any lane and issues a step.
        """
        _, lanes = self.launch_and_stop()
        focus, sibling = self._focus_and_sibling(lanes)
        self.assertIsNotNone(
            sibling,
            "the EU thread should have a sibling lane also reporting "
            "eStopReasonBreakpoint")
        # Both are at the same source line (stopped together in lockstep).
        self.assertEqual(
            focus.GetFrameAtIndex(0).GetLineEntry().GetLine(),
            sibling.GetFrameAtIndex(0).GetLineEntry().GetLine(),
            "focus and sibling lane should be at the same source line")

    def test_step_sibling_lane_completes(self):
        """Stepping a sibling lane completes (no hang, no runaway) and advances
        its PC.

        The step-complete stop must be attributed to the stepped sibling lane:
        after StepInstruction the sibling's PC advances and it reports a
        trace/plan-complete stop reason, proving the client's step on that
        lane was satisfied.
        """
        gpu_process, lanes = self.launch_and_stop()
        focus, sibling = self._focus_and_sibling(lanes)
        self.assertIsNotNone(sibling, "need a sibling lane to step")

        sibling_tid = sibling.GetThreadID()
        pc0 = sibling.GetFrameAtIndex(0).GetPC()
        self.assertNotEqual(pc0, lldb.LLDB_INVALID_ADDRESS)

        gpu_process.SetSelectedThread(sibling)
        error = lldb.SBError()
        sibling.StepInstruction(False, error)
        self.assertTrue(error.Success(),
                        f"stepping the sibling lane should succeed: "
                        f"{error.GetCString()}")
        self._wait_stopped(gpu_process)

        # Re-fetch the sibling lane (same stable TID) and verify it advanced
        # and carries the step-complete stop reason.
        stepped = gpu_process.GetThreadByID(sibling_tid)
        self.assertTrue(stepped.IsValid(),
                        "sibling lane should still exist after the step")
        pc1 = stepped.GetFrameAtIndex(0).GetPC()
        self.assertGreater(pc1, pc0,
                           f"sibling lane PC should advance (0x{pc0:x} -> "
                           f"0x{pc1:x})")
        self.assertIn(
            stepped.GetStopReason(),
            (lldb.eStopReasonTrace, lldb.eStopReasonPlanComplete),
            "stepped sibling lane should report a step-complete stop reason")
