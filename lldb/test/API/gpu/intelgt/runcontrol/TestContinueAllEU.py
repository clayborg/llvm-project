"""
Test continuing from an all-EU-thread breakpoint stop.

After the IntelGT plugin gathers all EU threads at a breakpoint, a continue
must resume the whole device (all EU threads together, all-stop semantics) and
let the kernel run to completion. This exercises the nresumed = nthreads
re-seed on a continue-all and confirms no EU thread is left stranded.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtContinueAllEUTestCase(RunControlTestCaseBase):
    """Continuing from a multi-EU stop resumes everything and finishes."""

    def test_continue_runs_to_completion(self):
        """Continue from the all-stopped breakpoint; kernel finishes cleanly."""
        target, _ = self._create_target_with_breakpoint("inside-kernel")
        process = self._launch(target)
        gpu_process = self._gpu_process()

        lanes = self._lane_threads(gpu_process)
        eus = {self.eu_of(t) for t in lanes}
        self.assertGreater(len(eus), 1,
                           "precondition: multiple EU threads stopped")

        # Delete the breakpoint so the continue is unobstructed, then resume
        # the whole device. The gather fix must not leave any EU thread
        # stranded: the continue should succeed and the process should leave
        # the stopped state (run to completion rather than hang or error).
        target.DeleteAllBreakpoints()
        self.dbg.SetSelectedTarget(self.dbg.GetTargetAtIndex(0))
        cpu_process = self.dbg.GetTargetAtIndex(0).GetProcess()
        error = cpu_process.Continue()
        self.assertTrue(error.Success(),
                        f"continue should succeed: {error.GetCString()}")

        state = self._wait_stopped(cpu_process)
        # All EU threads were resumed together; the process must make progress
        # past the breakpoint to termination (not remain stopped / hung).
        self.assertEqual(state, lldb.eStateExited,
                         "process should run to completion after continue")
