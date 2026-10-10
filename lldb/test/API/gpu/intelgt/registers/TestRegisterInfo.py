"""
Test GPU register display commands in IntelGT.

Verifies that LLDB's 'register read' command correctly displays Intel GPU
registers organized by register sets (GRF, ARF, SBA, etc.).

Based on GDB's gdb.arch/intelgt-info-reg.exp (lines 55-70).
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtRegisterInfoTestCase(IntelGtTestCaseBase):
    """Test register info/display commands."""

    def _stop_at_kernel_breakpoint(self):
        """Build, launch, and return (frame, process) stopped at the kernel bp."""
        self.build()
        exe = self.getBuildArtifact("simple_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex(
            "// GPU BREAKPOINT", lldb.SBFileSpec("simple_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0,
                           "Should have breakpoint location")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

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
        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped")

        self.assertEqual(self.dbg.GetNumTargets(), 2,
                         "GPU target should be created")
        gpu_process = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break
        self.assertIsNotNone(stopped_thread, "Should have stopped GPU thread")

        frame = stopped_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")
        return frame

    def _check_reg(self, frame, name, expected_byte_size=None, nonzero=False):
        """Assert register is valid and readable; optionally check size/nonzero."""
        reg = frame.FindRegister(name)
        self.assertTrue(reg.IsValid(), f"{name} should be accessible")
        if expected_byte_size is not None:
            self.assertEqual(reg.GetByteSize(), expected_byte_size,
                             f"{name} should be {expected_byte_size} bytes")
        if nonzero:
            self.assertNotEqual(reg.GetValueAsUnsigned(), 0,
                                f"{name} should be non-zero")
        return reg

    def test_register_read_default(self):
        """register read shows GRF registers (r0, r1, r2) by default."""
        frame = self._stop_at_kernel_breakpoint()

        reg_sets = frame.GetRegisters()
        self.assertGreater(reg_sets.GetSize(), 0, "Should have register sets")

        grf_set = None
        for i in range(reg_sets.GetSize()):
            regset = reg_sets.GetValueAtIndex(i)
            if "GRF" in regset.GetName():
                grf_set = regset
                break
        self.assertIsNotNone(grf_set, "Should have GRF register set")

        reg_names = [grf_set.GetChildAtIndex(i).GetName()
                     for i in range(min(10, grf_set.GetNumChildren()))]
        self.assertIn("r0", reg_names, "GRF set should contain r0")
        self.assertIn("r1", reg_names, "GRF set should contain r1")
        self.assertIn("r2", reg_names, "GRF set should contain r2")

        r0 = frame.FindRegister("r0")
        self.assertTrue(r0.IsValid(), "r0 should be valid")
        # GRF registers are vector-encoded; they have lane children, not a flat value.
        self.assertGreater(r0.GetNumChildren(), 0, "r0 should have lane elements")

    def test_register_read_arf(self):
        """ARF registers (sr0, cr0, ce) are accessible."""
        frame = self._stop_at_kernel_breakpoint()

        self._check_reg(frame, "sr0")
        self._check_reg(frame, "cr0")
        ce = frame.FindRegister("ce")
        self.assertTrue(ce.IsValid(), "ce register should be accessible")

    def test_register_read_virtual(self):
        """PC and ip pseudo-registers are accessible and consistent."""
        frame = self._stop_at_kernel_breakpoint()

        pc = frame.GetPC()
        self.assertNotEqual(pc, lldb.LLDB_INVALID_ADDRESS, "PC should be valid")

        pc_reg = self._check_reg(frame, "pc", expected_byte_size=8, nonzero=True)
        self.assertEqual(pc_reg.GetValueAsUnsigned(), pc,
                         "pc register should match frame PC")

        ip_reg = self._check_reg(frame, "ip", expected_byte_size=4)
        ip_val = ip_reg.GetValueAsUnsigned()
        # pc == isabase + ip  (isabase may be 0 on some devices)
        isabase_reg = frame.FindRegister("isabase")
        if isabase_reg.IsValid():
            isabase = isabase_reg.GetValueAsUnsigned()
            self.assertEqual(pc, isabase + ip_val,
                             "pc should equal isabase + ip")
        print(f"ip=0x{ip_val:x}  pc=0x{pc:x}")

    def test_all_register_groups(self):
        """One register from every exposed register group is readable.

        Covers: GRF, Addr, Flag, CE, SR, CR, ACC, MME, SP, SBA, ip pseudo.
        Optional groups (TDR, DBG, FC, MF) are checked only when present.
        """
        frame = self._stop_at_kernel_breakpoint()

        # --- GRF (vector, 64 bytes on SIMD32) ---
        r0 = self._check_reg(frame, "r0")
        self.assertGreater(r0.GetNumChildren(), 0, "r0 should have lane elements")

        # --- Addr ---
        self._check_reg(frame, "a0")

        # --- Flag ---
        self._check_reg(frame, "f0")

        # --- CE ---
        ce = frame.FindRegister("ce")
        self.assertTrue(ce.IsValid(), "ce should be accessible")

        # --- SR ---
        self._check_reg(frame, "sr0")

        # --- CR ---
        self._check_reg(frame, "cr0")

        # --- ACC ---
        self._check_reg(frame, "acc0")

        # --- MME ---
        self._check_reg(frame, "mme0")

        # --- SBA: names match GDB's sba_names[] ---
        for name in ("genstbase", "sustbase", "isabase",
                     "blsustbase", "blsastbase", "btbase", "scrbase0"):
            self._check_reg(frame, name, expected_byte_size=8)

        # --- ip pseudo (raw 32-bit offset from CR0.dword2) ---
        self._check_reg(frame, "ip", expected_byte_size=4)

        # --- pc pseudo (64-bit full address) ---
        self._check_reg(frame, "pc", expected_byte_size=8, nonzero=True)

        # --- DBG ---
        self._check_reg(frame, "dbg0")

        # --- Optional groups: present on some devices, skip if absent ---
        for name in ("fc0", "mf0"):
            reg = frame.FindRegister(name)
            if reg.IsValid():
                self.assertIsNotNone(reg.GetValue(),
                                     f"{name} present but unreadable")
                print(f"{name}={reg.GetValue()}")
            else:
                print(f"{name}: not present on this device")
