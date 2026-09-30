"""
Test conditional breakpoints across multiple EU threads / SIMD lanes.

This is the core motivation for stopping all EU threads on a breakpoint: a
breakpoint condition may only match work-items that live in an EU thread other
than the one that physically trapped. If sibling EU threads were not stopped
and presented, a conditional hit in those lanes would be silently missed.

The kernel launches 32 work-items spread across multiple EU threads. We set a
condition matching work-items in BOTH EU threads (gid in the low half and the
high half) and verify the matching lanes are present at the stop.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtConditionalBreakpointTestCase(RunControlTestCaseBase):
    """Conditional breakpoint must see matching lanes in every EU thread."""

    # Two work-items that (for a SIMD16/2-EU split of 32 items) fall in
    # different EU threads: 3 in the low half, 20 in the high half.
    MATCH_GIDS = [3, 20]

    def test_condition_matches_lanes_in_multiple_eu_threads(self):
        """A condition matching work-items across EU threads stops on them.

        Without all-EU-thread gathering, a gid that lives in the non-trapping
        EU thread would be missed; with it, every matching lane is visible.
        """
        cond = " || ".join(f"gid == {g}" for g in self.MATCH_GIDS)
        target, _ = self._create_target_with_breakpoint("inside-kernel")

        # Replace the plain breakpoint with a conditional one on the same line.
        target.DeleteAllBreakpoints()
        bp = target.BreakpointCreateBySourceRegex(
            "inside-kernel", lldb.SBFileSpec("parallel_1d.cpp"))
        self.assertGreater(bp.GetNumLocations(), 0)
        bp.SetCondition(cond)

        self._launch(target)
        gpu_process = self._gpu_process()
        lanes = self._lane_threads(gpu_process)

        # Collect gids of lanes that satisfy the condition and are presented.
        present_match_gids = set()
        eus_with_match = set()
        for t in lanes:
            gid = self.read_int(t, "gid")
            if gid in self.MATCH_GIDS:
                present_match_gids.add(gid)
                eus_with_match.add(self.eu_of(t))

        # Every matching work-item must be visible at the stop...
        self.assertEqual(
            present_match_gids, set(self.MATCH_GIDS),
            f"all condition-matching work-items {self.MATCH_GIDS} should be "
            f"present, got {sorted(present_match_gids)}")

        # ...and they must span more than one EU thread, which is the case the
        # all-EU-thread gather exists to handle.
        self.assertGreater(
            len(eus_with_match), 1,
            "matching work-items should span multiple EU threads "
            f"(saw {len(eus_with_match)})")
