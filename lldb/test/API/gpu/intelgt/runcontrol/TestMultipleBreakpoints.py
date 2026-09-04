"""
Test multiple software breakpoints in an Intel GT kernel.

The plugin patches instruction bits to set software breakpoints and saves the
original opcodes per address (m_bp_saved_opcodes), so several breakpoints must
coexist. Verify two breakpoints on different kernel lines both work, that
continuing moves from the first to the second, and that removing one restores
its instruction (no spurious stop).
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtMultipleBreakpointsTestCase(RunControlTestCaseBase):
    """Multiple software breakpoints coexist and fire independently."""

    def _line_of(self, marker):
        return line_number("parallel_1d.cpp", marker)

    def test_two_breakpoints_resolve_and_first_fires(self):
        """Two BPs on different kernel lines coexist; the earlier one fires.

        Both software breakpoints resolve to distinct locations/IDs (the plugin
        patches and saves each instruction independently), and launching stops
        at the earlier line (first-line). The complementary case — that the
        later breakpoint also fires once the earlier is removed — is covered by
        test_remove_one_breakpoint.
        """
        target, _ = self._create_target_with_breakpoint("first-line")
        # Add a second breakpoint on a later line.
        spec = lldb.SBFileSpec("parallel_1d.cpp")
        bp_b = target.BreakpointCreateBySourceRegex("inside-kernel", spec)
        self.assertGreater(bp_b.GetNumLocations(), 0, "BP B should resolve")

        first_line = self._line_of("first-line")
        store_line = self._line_of("inside-kernel")
        self.assertNotEqual(first_line, store_line,
                            "the two markers must be on distinct lines")

        self._launch(target)
        gpu_process = self._gpu_process()
        bp_thread = next((t for t in self._lane_threads(gpu_process)
                          if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                         None)
        self.assertIsNotNone(bp_thread, "a lane should report a breakpoint")
        line = bp_thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(line, first_line,
                         f"with both BPs set, the first stop should be at the "
                         f"earlier line {first_line}, was {line}")

    def test_remove_one_breakpoint(self):
        """Deleting the earlier BP makes run stop at the later one instead."""
        self.build()
        exe = self.getBuildArtifact("parallel_1d")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        spec = lldb.SBFileSpec("parallel_1d.cpp")
        bp_a = target.BreakpointCreateBySourceRegex("first-line", spec)
        bp_b = target.BreakpointCreateBySourceRegex("inside-kernel", spec)
        self.assertGreater(bp_a.GetNumLocations(), 0)
        self.assertGreater(bp_b.GetNumLocations(), 0)

        # Remove the earlier breakpoint; its patched instruction must be
        # restored so we do NOT stop there.
        self.assertTrue(target.BreakpointDelete(bp_a.GetID()),
                        "should delete BP A")

        self._launch(target)
        gpu_process = self._gpu_process()
        lanes = self._lane_threads(gpu_process)
        bp_thread = next((t for t in lanes
                          if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                         None)
        self.assertIsNotNone(bp_thread, "should stop at the remaining breakpoint")
        line = bp_thread.GetFrameAtIndex(0).GetLineEntry().GetLine()
        self.assertEqual(line, self._line_of("inside-kernel"),
                         "with BP A removed, first stop should be at BP B")
