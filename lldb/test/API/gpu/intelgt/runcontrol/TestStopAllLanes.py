"""
Test that a breakpoint hit stops and presents ALL EU threads / SIMD lanes.

When a breakpoint inside a parallel_for kernel is hit, the work-items are
spread across multiple EU hardware threads. The Level Zero driver only reports
the EU thread(s) that trapped and does NOT stop the siblings on its own. The
IntelGT plugin must interrupt all running EU threads and gather them before
reporting the stop, so that every work-item (and every SIMD lane, for
conditional breakpoints) is visible to LLDB.

The kernel launches 32 work-items, which the hardware splits across more than
one EU thread. This test verifies all 32 are present at the stop.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtStopAllLanesTestCase(RunControlTestCaseBase):
    """Verify all EU threads / lanes stop together on a breakpoint."""

    def test_multiple_eu_threads_stop(self):
        """More than one EU thread is presented at the breakpoint."""
        _, lanes = self.launch_and_stop()
        eus = {self.eu_of(t) for t in lanes}
        self.assertGreater(
            len(eus), 1,
            f"breakpoint should stop multiple EU threads, saw {len(eus)} "
            f"(bases {sorted(hex(e) for e in eus)})")

    def test_all_workitems_present(self):
        """Every work-item 0..N-1 is visible across all stopped EU threads.

        gid is read per lane-thread; within each EU thread gid increases 1:1
        with the lane index (gid - lane is constant per EU thread).
        """
        _, lanes = self.launch_and_stop()

        gids = set()
        base_by_eu = {}
        for t in lanes:
            lane = self.lane_of(t)
            gid = self.read_int(t, "gid")
            self.assertIsNotNone(gid, f"lane {lane}: gid should be readable")
            gids.add(gid)
            eu = self.eu_of(t)
            base = gid - lane
            if eu in base_by_eu:
                self.assertEqual(
                    base_by_eu[eu], base,
                    f"EU 0x{eu:x}: gid-lane should be constant within an EU "
                    f"thread (saw {base_by_eu[eu]} and {base})")
            else:
                base_by_eu[eu] = base

        self.assertEqual(
            gids, set(range(self.NUM_WORK_ITEMS)),
            f"all work-items 0..{self.NUM_WORK_ITEMS - 1} should be present, "
            f"got {sorted(gids)}")

    def test_stopped_thread_list(self):
        """The full GPU thread list reports every active lane as stopped.

        Mirrors gdb.sycl/intelgt-info-threads: at an all-stop breakpoint every
        presented lane-thread is in the stopped state and has a valid frame 0
        in the kernel source.
        """
        gpu_process, lanes = self.launch_and_stop()

        # At least one lane per EU thread reports the breakpoint stop reason;
        # the rest are siblings stopped at the same location.
        bp_lanes = [t for t in lanes
                    if t.GetStopReason() == lldb.eStopReasonBreakpoint]
        self.assertGreater(len(bp_lanes), 0,
                           "at least one lane should report the breakpoint")

        for t in lanes:
            frame = t.GetFrameAtIndex(0)
            self.assertTrue(frame.IsValid(),
                            f"lane 0x{t.GetThreadID():x} should have a frame")
            le = frame.GetLineEntry()
            self.assertEqual(
                le.GetFileSpec().GetFilename(), "parallel_1d.cpp",
                f"lane 0x{t.GetThreadID():x} frame 0 should be in the kernel")
