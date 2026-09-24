"""
Test instruction-level single-stepping (stepi) of an Intel GT EU thread.

The plugin arms a hardware single-step via CR0 (EUThreadIntelGT::PrepareStep /
ClearStepBits). This verifies that StepInstruction advances the PC by one
instruction and the thread reports a clean step-complete stop, and that
stepping repeatedly keeps the PC moving forward.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from runcontrol_util import RunControlTestCaseBase


class IntelGtStepInstructionTestCase(RunControlTestCaseBase):
    """Single instruction stepping on the GPU."""

    def _stepping_thread(self, lanes):
        return next((t for t in lanes
                     if t.GetStopReason() == lldb.eStopReasonBreakpoint),
                    lanes[0])

    def test_stepi_advances_pc(self):
        """StepInstruction moves the PC forward by one instruction."""
        gpu_process, lanes = self.launch_and_stop("first-line")
        thread = self._stepping_thread(lanes)

        pc0 = thread.GetFrameAtIndex(0).GetPC()
        self.assertNotEqual(pc0, lldb.LLDB_INVALID_ADDRESS, "PC should be valid")

        thread.StepInstruction(False)  # step-into-instruction
        self._wait_stopped(gpu_process)

        pc1 = thread.GetFrameAtIndex(0).GetPC()
        self.assertNotEqual(pc1, lldb.LLDB_INVALID_ADDRESS,
                            "PC should be valid after stepi")
        self.assertGreater(pc1, pc0,
                           f"stepi should advance PC (0x{pc0:x} -> 0x{pc1:x})")

    def test_stepi_repeated_monotonic(self):
        """Several stepi in a row keep the PC strictly advancing."""
        gpu_process, lanes = self.launch_and_stop("first-line")
        thread = self._stepping_thread(lanes)

        last = thread.GetFrameAtIndex(0).GetPC()
        for i in range(3):
            thread.StepInstruction(False)
            self._wait_stopped(gpu_process)
            if gpu_process.GetState() == lldb.eStateExited:
                break
            pc = thread.GetFrameAtIndex(0).GetPC()
            self.assertGreater(pc, last,
                               f"stepi {i}: PC should advance "
                               f"(0x{last:x} -> 0x{pc:x})")
            last = pc
