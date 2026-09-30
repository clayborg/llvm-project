"""
Test GPU register read/write operations in IntelGT.

Verifies that LLDB can:
1. Read register values from stopped GPU threads
2. Modify register values
3. Verify modified values persist and affect execution

Based on GDB's gdb.arch/intelgt-read-register.exp and
gdb.arch/intelgt-write-register.exp.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtRegisterReadWriteTestCase(IntelGtTestCaseBase):
    """Test register read and write operations."""

    def test_register_read_value(self):
        """Test reading register values from GPU thread.

        Verifies that we can read actual values from GRF and ARF registers
        when GPU thread is stopped at a breakpoint.
        """
        self.build()

        exe = self.getBuildArtifact("read_write_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        # Set breakpoint
        bp = target.BreakpointCreateBySourceRegex("read-write-breakpoint",
            lldb.SBFileSpec("read_write_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0, "Should have breakpoint location")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")
        self.assertTrue(process.IsValid(), "Process should be valid")

        # Wait for breakpoint
        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        timeout_seconds = 30

        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(timeout_seconds, event):
                self.fail(f"Timeout waiting for breakpoint")
            state = process.GetState()

        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped")

        # Get GPU thread
        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target should be created")
        gpu_target = self.dbg.GetTargetAtIndex(1)
        self.assertTrue(gpu_target.IsValid(), "GPU target should be valid")

        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break

        self.assertIsNotNone(stopped_thread, "Should have stopped GPU thread")

        frame = stopped_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        # Test: Read r0 register value (GRF is eEncodingVector, 32 or 64 bytes depending on HW)
        r0 = frame.FindRegister("r0")
        self.assertTrue(r0.IsValid(), "r0 should be valid")
        self.assertIn(r0.GetByteSize(), [32, 64], "r0 should be 32 bytes (SIMD16) or 64 bytes (SIMD32)")

        # Vector registers expose children (elements), not a flat string value
        self.assertGreater(r0.GetNumChildren(), 0, "r0 should have child elements")
        r0_elem0 = r0.GetChildAtIndex(0)
        self.assertTrue(r0_elem0.IsValid(), "r0[0] element should be valid")
        print(f"r0[0] value: {r0_elem0.GetValue()}")

        # Test: Read ARF register (sr0)
        sr0 = frame.FindRegister("sr0")
        self.assertTrue(sr0.IsValid(), "sr0 should be valid")

        sr0_value = sr0.GetValue()
        self.assertIsNotNone(sr0_value, "sr0 should have a value")
        print(f"sr0 value: {sr0_value}")

        # Test: Read virtual register (PC)
        pc_reg = frame.FindRegister("pc")
        self.assertTrue(pc_reg.IsValid(), "pc should be valid")

        pc_value = pc_reg.GetValueAsUnsigned()
        self.assertNotEqual(pc_value, 0, "PC should be non-zero")

        # PC should match frame's PC
        frame_pc = frame.GetPC()
        self.assertEqual(pc_value, frame_pc, "PC register should match frame PC")
        print(f"PC: 0x{pc_value:x}")

    def test_register_write_grf(self):
        """Test writing to GRF register.

        GRF registers are 32 bytes (SIMD16) or 64 bytes (SIMD32). Write is done
        via SBData using CreateDataFromUInt64Array sized to match the register.
        """
        self.build()

        exe = self.getBuildArtifact("read_write_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex("read-write-breakpoint",
            lldb.SBFileSpec("read_write_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        timeout_seconds = 30

        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(timeout_seconds, event):
                self.fail(f"Timeout waiting for breakpoint")
            state = process.GetState()

        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped")

        gpu_target = self.dbg.GetTargetAtIndex(1)
        gpu_process = gpu_target.GetProcess()

        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break

        self.assertIsNotNone(stopped_thread, "Should have stopped GPU thread")

        frame = stopped_thread.GetFrameAtIndex(0)

        # Test: Write to r127 (last GRF register)
        r127 = frame.FindRegister("r127")
        self.assertTrue(r127.IsValid(), "r127 should be valid")

        # Read original value via first child element (vector register)
        self.assertGreater(r127.GetNumChildren(), 0, "r127 should have child elements")
        r127_elem0 = r127.GetChildAtIndex(0)
        self.assertTrue(r127_elem0.IsValid(), "r127[0] should be valid")
        original_elem0 = r127_elem0.GetValue()
        print(f"r127[0] original value: {original_elem0}")

        # GRF size is 32 bytes (SIMD16) or 64 bytes (SIMD32). Use SBData with
        # the right number of uint64 values to match the register's actual size.
        num_words = r127.GetByteSize() // 8
        new_words = [0x1122334455667788] * num_words
        data = lldb.SBData.CreateDataFromUInt64Array(
            lldb.eByteOrderLittle, 8, new_words)

        # Disable synthetic value so SetData reaches the actual register,
        # not a synthetic host-memory copy.
        r127.SetPreferSyntheticValue(False)
        error = lldb.SBError()
        success = r127.SetData(data, error)
        self.assertTrue(success, f"Should be able to write to r127: {error.GetCString()}")

        # Read back and verify the value changed
        r127.Clear()
        r127 = frame.FindRegister("r127")
        new_elem0 = r127.GetChildAtIndex(0).GetValue()
        self.assertNotEqual(new_elem0, original_elem0,
                           f"r127[0] should have changed after write, got {new_elem0}")
        print(f"r127[0] new value: {new_elem0}")

    def test_register_write_pc(self):
        """Test that PC register is read-only.

        Virtual registers like PC should not be writable.
        """
        self.build()

        exe = self.getBuildArtifact("read_write_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex("read-write-breakpoint",
            lldb.SBFileSpec("read_write_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        timeout_seconds = 30

        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(timeout_seconds, event):
                self.fail(f"Timeout waiting for breakpoint")
            state = process.GetState()

        gpu_target = self.dbg.GetTargetAtIndex(1)
        gpu_process = gpu_target.GetProcess()

        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break

        self.assertIsNotNone(stopped_thread, "Should have stopped GPU thread")

        frame = stopped_thread.GetFrameAtIndex(0)

        # Test: Attempt to write PC should fail (or be ignored)
        pc_reg = frame.FindRegister("pc")
        self.assertTrue(pc_reg.IsValid(), "pc should be valid")

        original_pc = pc_reg.GetValueAsUnsigned()

        # Try to write a different value
        success = pc_reg.SetValueFromCString("0x1234567890")

        # Either write should fail, or PC should remain unchanged
        current_pc = pc_reg.GetValueAsUnsigned()
        if success:
            # If write "succeeded", PC should not actually have changed
            self.assertEqual(current_pc, original_pc,
                           "PC should not change even if write appears successful")
        else:
            print("PC write correctly rejected")
