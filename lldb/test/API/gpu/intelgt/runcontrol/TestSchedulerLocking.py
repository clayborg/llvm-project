"""
Test scheduler-locking semantics across EU threads (all-stop model).

The IntelGT plugin is all-stop. Scheduler-locking on the GPU is *partial* and
depends on the mechanism of the run-control operation:

  - A pure single instruction step (stepi) advances ONLY the stepped EU thread
    (CR0 single-step of one EU thread); sibling EU threads stay put. This is
    verified here and mirrors GDB's scheduler-locking-on-step.

  - Operations that internally CONTINUE the target — step-over a function call,
    and finish/step-out — resume the WHOLE device, so sibling EU threads also
    advance. Those are verified in TestStepOverFinishResumesAll (which needs a
    call-chain kernel).

The kernel (parallel_1d.cpp, 32 work-items) is split by the hardware across
more than one EU thread, so a breakpoint stop presents >= 2 EU threads.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtSchedulerLockingTestCase(RunControlTestCaseBase):
    """Stepping one EU thread does not advance the others."""

    def _eu_groups(self, lanes):
        """Group lane-threads by EU-thread base TID."""
        groups = {}
        for t in lanes:
            groups.setdefault(self.eu_of(t), []).append(t)
        return groups

    def _focus(self, group):
        """The breakpoint-reporting lane of an EU group, else the first."""
        return next((t for t in group
                     if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                    group[0])

    def test_step_one_eu_thread_leaves_others_put(self):
        """Single-stepping one EU thread advances only that EU thread."""
        gpu_process, lanes = self.launch_and_stop()

        groups = self._eu_groups(lanes)
        self.assertGreaterEqual(
            len(groups), 2,
            f"need >= 2 EU threads for scheduler-locking; saw {len(groups)} "
            f"(bases {sorted(hex(b) for b in groups)})")

        bases = sorted(groups)
        eu_a, eu_b = bases[0], bases[1]
        a = self._focus(groups[eu_a])
        b = self._focus(groups[eu_b])

        a_tid, b_tid = a.GetThreadID(), b.GetThreadID()
        pc_a0 = a.GetFrameAtIndex(0).GetPC()
        pc_b0 = b.GetFrameAtIndex(0).GetPC()
        self.assertNotEqual(pc_a0, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(pc_b0, lldb.LLDB_INVALID_ADDRESS)

        # Step EU thread A's focus lane.
        gpu_process.SetSelectedThread(a)
        error = lldb.SBError()
        a.StepInstruction(False, error)
        self.assertTrue(error.Success(),
                        f"stepping EU thread A should succeed: "
                        f"{error.GetCString()}")
        self._wait_stopped(gpu_process)

        a_after = gpu_process.GetThreadByID(a_tid)
        b_after = gpu_process.GetThreadByID(b_tid)
        self.assertTrue(a_after.IsValid(), "EU thread A should still exist")
        self.assertTrue(b_after.IsValid(),
                        "EU thread B should still be present (stopped)")

        pc_a1 = a_after.GetFrameAtIndex(0).GetPC()
        pc_b1 = b_after.GetFrameAtIndex(0).GetPC()

        # A advanced; B did not.
        self.assertGreater(pc_a1, pc_a0,
                           f"stepped EU thread A should advance "
                           f"(0x{pc_a0:x} -> 0x{pc_a1:x})")
        self.assertEqual(pc_b1, pc_b0,
                         f"non-stepped EU thread B must stay put "
                         f"(was 0x{pc_b0:x}, now 0x{pc_b1:x})")
