"""
Test SIMD lane divergence handling on Intel GT.

A kernel with an if/else on (gid % 2) makes even and odd lanes take different
branches, so the EU thread's execution mask (CE) is half-masked inside each
branch. We verify:

  * At a breakpoint inside a branch body, only the lanes active for that branch
    are presented, the execution mask is half-masked, and the focus thread is
    an active lane (the plugin refreshes active lanes from CE on every stop).

  * (XFAIL) Following a single focus lane THROUGH a divergent branch via
    instruction stepping keeps the focus on that lane until it re-converges
    (GDB's re-enable-PC mechanism). The IntelGT plugin does not implement this
    yet: stepping the EU thread advances all lanes in lockstep and, once the
    focus lane goes inactive, there is no re-enable-PC follow to keep tracking
    just that lane.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtLaneDivergenceTestCase(IntelGtTestCaseBase):
    """Divergent-branch execution-mask and stepping behaviour."""

    SOURCE = "diverge_1d.cpp"
    EXE = "diverge_1d"

    def _launch_at(self, pattern):
        """Build, launch, stop at *pattern*, return the GPU process."""
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

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(30, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped, "host process should stop")

        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target expected")
        gpu = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertTrue(gpu.IsValid(), "GPU process should be valid")
        return gpu

    @staticmethod
    def _lane(t):
        return t.GetThreadID() & 0xFF

    @staticmethod
    def _eu(t):
        return t.GetThreadID() & ~0xFF

    def _ce(self, thread):
        ce = thread.GetFrameAtIndex(0).FindRegister("ce")
        if not ce.IsValid():
            return None
        return ce.GetValueAsUnsigned()

    def test_even_branch_masks_odd_lanes(self):
        """In the even branch only even lanes are active and focus is active."""
        gpu = self._launch_at("even-branch")
        focus = gpu.GetSelectedThread()
        self.assertEqual(focus.GetStopReason(), lldb.eStopReasonBreakpoint,
                         "focus thread should be stopped at the breakpoint")

        focus_lane = self._lane(focus)
        self.assertEqual(focus_lane % 2, 0,
                         f"focus lane {focus_lane} should be even in even branch")

        ce = self._ce(focus)
        self.assertIsNotNone(ce, "CE should be readable")
        # Even lanes active => 0x...5555 pattern (bit 0,2,4,... set).
        self.assertEqual(ce & 0x3, 0x1,
                         f"CE 0x{ce:x} should have lane 0 active, lane 1 inactive")

        # Every presented lane carrying the breakpoint stop reason must be
        # active in CE (the plugin refreshes active lanes from CE on stop).
        eu = self._eu(focus)
        for t in gpu.threads:
            if t.GetThreadID() == 1 or self._eu(t) != eu:
                continue
            if t.GetStopReason() == lldb.eStopReasonBreakpoint:
                lane = self._lane(t)
                self.assertTrue(ce & (1 << lane),
                                f"stopped lane {lane} should be active in "
                                f"CE 0x{ce:x}")

    def test_odd_branch_masks_even_lanes(self):
        """In the odd branch the focus lane is odd and active."""
        gpu = self._launch_at("odd-branch")
        focus = gpu.GetSelectedThread()
        self.assertEqual(focus.GetStopReason(), lldb.eStopReasonBreakpoint)
        focus_lane = self._lane(focus)
        self.assertEqual(focus_lane % 2, 1,
                         f"focus lane {focus_lane} should be odd in odd branch")
        ce = self._ce(focus)
        self.assertIsNotNone(ce)
        self.assertTrue(ce & (1 << focus_lane),
                        f"focus lane {focus_lane} should be active in CE 0x{ce:x}")

    @expectedFailureAll(
        bugnumber="IntelGT: no re-enable-PC follow for a diverged focus lane")
    def test_step_follows_focus_lane_through_divergence(self):
        """Stepping a focus lane through a divergent branch keeps tracking it.

        GDB sets a breakpoint at the lane's re-enable (reconvergence) PC and
        resumes so the user keeps following the SAME lane across the branch it
        does not take. The IntelGT plugin has no such mechanism: stepping
        advances the whole EU thread in lockstep and, when the focus lane goes
        inactive, the focus is re-selected to a DIFFERENT (active) lane instead
        of following the original one. This XFAIL documents that gap.
        """
        gpu = self._launch_at("branch-line")
        # Focus an odd lane (lane 1): it will take the ELSE branch. As we step
        # the EU thread, the IF (even) lanes execute first while lane 1 is
        # inactive. A correct re-enable-PC follow would keep the focus on lane
        # 1 until it becomes active again; instead the focus moves to lane 0.
        odd = gpu.GetThreadByID((gpu.GetSelectedThread().GetThreadID() & ~0xFF) | 1)
        self.assertTrue(odd.IsValid(), "odd lane 1 should exist at the branch")
        gpu.SetSelectedThread(odd)
        self.assertEqual(self._lane(gpu.GetSelectedThread()), 1)

        # Step until the execution mask first changes (the divergence point).
        prev_ce = self._ce(odd)
        for _ in range(60):
            gpu.GetSelectedThread().StepInstruction(False)
            listener = self.dbg.GetListener()
            event = lldb.SBEvent()
            st = gpu.GetState()
            while st == lldb.eStateRunning:
                if not listener.WaitForEvent(30, event):
                    break
                st = gpu.GetState()
            if gpu.GetState() == lldb.eStateExited:
                break
            cur = gpu.GetSelectedThread()
            ce = self._ce(cur)
            if ce is not None and ce != prev_ce:
                # Divergence happened. The focus must still be lane 1.
                self.assertEqual(
                    self._lane(cur), 1,
                    "focus should still be the original lane 1 after the "
                    "divergent branch (requires re-enable-PC follow)")
                return
            prev_ce = ce
        self.fail("did not reach the divergence point within the step budget")
