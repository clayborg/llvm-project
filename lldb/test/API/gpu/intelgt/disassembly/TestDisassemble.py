"""
Test disassembly of Intel GT GPU kernel code.

At a kernel breakpoint, `disassemble` (via the DisassemblerIntelGT plugin) must
decode the EU instructions around the current PC. We assert structure
(instructions decoded, addresses present and ordered, current PC covered)
rather than exact mnemonics, which vary by device generation.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtDisassembleTestCase(IntelGtTestCaseBase):
    """Disassemble GPU kernel code at a stop."""

    SOURCE = "disasm_kernel.cpp"
    EXE = "disasm_kernel"

    def _stop_at_kernel(self):
        """Build, launch, stop at the kernel breakpoint, return the stopped
        GPU thread."""
        self.build()
        exe = self.getBuildArtifact(self.EXE)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex(
            "inside-kernel", lldb.SBFileSpec(self.SOURCE))
        self.assertGreater(bp.GetNumLocations(), 0,
                           "breakpoint should resolve")

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
        thread = next((t for t in gpu.threads
                       if t.GetStopReason() == lldb.eStopReasonBreakpoint), None)
        self.assertIsNotNone(thread, "a GPU thread should hit the breakpoint")
        return thread

    def test_disassemble_current_frame(self):
        """frame.Disassemble() returns non-empty instructions with addresses."""
        thread = self._stop_at_kernel()
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "frame 0 should be valid")

        text = frame.Disassemble()
        self.assertTrue(text and text.strip(),
                        "frame.Disassemble() should return text")
        self.assertIn("0x", text, "disassembly should contain addresses")
        nlines = [l for l in text.splitlines() if "0x" in l]
        self.assertGreater(len(nlines), 0,
                           f"expected decoded instructions, got:\n{text}")

    def test_disassemble_function_instructions(self):
        """SBFunction/SBSymbol disassembly yields ordered instructions."""
        thread = self._stop_at_kernel()
        frame = thread.GetFrameAtIndex(0)
        target = self.dbg.GetTargetAtIndex(1)

        insts = None
        symbol = frame.GetSymbol()
        if symbol and symbol.IsValid():
            insts = symbol.GetInstructions(target)
        if insts is None or insts.GetSize() == 0:
            func = frame.GetFunction()
            if func and func.IsValid():
                insts = func.GetInstructions(target)

        self.assertIsNotNone(insts, "should obtain an instruction list")
        n = insts.GetSize()
        self.assertGreater(n, 0, "kernel function should disassemble to >0 insts")

        prev = None
        for i in range(n):
            inst = insts.GetInstructionAtIndex(i)
            addr = inst.GetAddress().GetLoadAddress(target)
            self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS,
                                f"inst {i} should have a load address")
            if prev is not None:
                self.assertGreaterEqual(addr, prev,
                                        "instruction addresses should be ordered")
            prev = addr

    def test_disassemble_pc_is_covered(self):
        """The current PC matches a disassembled instruction address."""
        thread = self._stop_at_kernel()
        frame = thread.GetFrameAtIndex(0)
        target = self.dbg.GetTargetAtIndex(1)
        pc = frame.GetPC()
        self.assertNotEqual(pc, lldb.LLDB_INVALID_ADDRESS, "PC should be valid")

        func = frame.GetFunction()
        insts = func.GetInstructions(target) if func and func.IsValid() else None
        if insts is None or insts.GetSize() == 0:
            sym = frame.GetSymbol()
            insts = sym.GetInstructions(target) if sym and sym.IsValid() else None
        self.assertIsNotNone(insts, "should obtain instructions")

        addrs = {insts.GetInstructionAtIndex(i).GetAddress().GetLoadAddress(target)
                 for i in range(insts.GetSize())}
        self.assertIn(pc, addrs,
                      f"current PC 0x{pc:x} should be a disassembled instruction")
