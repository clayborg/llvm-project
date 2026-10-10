"""
Test GPU illegal opcode exception detection in IntelGT.

Verifies that executing 0x00 byte (illegal opcode per ISA) triggers an illegal
opcode exception that is correctly detected and reported by LLDB.

The test:
1. Sets a breakpoint in the kernel
2. Runs until breakpoint is hit
3. Overwrites instruction at PC with 0x00 byte (illegal opcode)
4. Continues execution
5. Verifies illegal opcode exception is caught with SIGILL signal
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtIllegalOpcodeExceptionTestCase(IntelGtTestCaseBase):
    """Test illegal opcode exception detection."""

    @expectedFailureAll(bugnumber="UMD does not enable illegal opcode exceptions (CR0.1 bit 12)")
    def test_illegal_opcode_exception(self):
        """Test that GPU illegal opcode is detected with SIGILL signal.

        Kernel is stopped at a breakpoint, then the instruction at PC is
        overwritten with 0x00 (illegal opcode per Intel GPU ISA).
        On continue, the GPU should raise an illegal opcode exception.

        Expected behavior:
        1. Breakpoint is hit in kernel
        2. Memory at PC is overwritten with 0x00 byte (illegal opcode)
        3. GPU process continues and hits exception (CR0.1 bit 28 set)
        4. At least one GPU thread reports eStopReasonException
        5. Signal is SIGILL and stop description contains "Illegal opcode"
        """
        self.build()

        exe = self.getBuildArtifact("illegal_opcode")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        # Set breakpoint on the line with "int x = 42"
        bp = target.BreakpointCreateBySourceRegex("int x = 42",
            lldb.SBFileSpec("illegal_opcode.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0, "Should have at least one breakpoint location")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")
        self.assertTrue(process.IsValid(), "Process should be valid")

        # Wait for breakpoint hit
        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        timeout_seconds = 30

        if not listener.WaitForEvent(timeout_seconds, event):
            self.fail(f"Timeout waiting for breakpoint after {timeout_seconds}s")

        state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped,
                        f"Process should be stopped at breakpoint, got state {state}")

        # Find GPU target and process
        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target should be created")
        gpu_target = self.dbg.GetTargetAtIndex(1)
        self.assertTrue(gpu_target.IsValid(), "GPU target should be valid")

        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        # Find the stopped GPU thread
        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break

        self.assertIsNotNone(stopped_thread, "Should have a GPU thread stopped at breakpoint")

        # Get PC (program counter) from the stopped thread
        frame = stopped_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")
        pc = frame.GetPC()
        self.assertNotEqual(pc, lldb.LLDB_INVALID_ADDRESS, "PC should be valid")

        # Overwrite kernel memory with 0x00 bytes (illegal opcode)
        # According to ISA: "The byte value of the illegal opcode is 0x00"
        invalid_bytes = bytes([0x00] * 1024)
        error = lldb.SBError()
        bytes_written = gpu_process.WriteMemory(pc, invalid_bytes, error)

        self.assertTrue(error.Success(),
                       f"Memory write should succeed: {error.GetCString()}")
        self.assertEqual(bytes_written, 1024,
                        f"Should write 1024 bytes, wrote {bytes_written}")

        # Remove the breakpoint so we don't hit it again
        target.BreakpointDelete(bp.GetID())

        # From here the GPU kernel code is corrupted (0x00 bytes). Whatever
        # happens, we must KILL the process before returning so we never leave
        # a GPU running corrupted code — that would wedge the device/driver
        # session and poison subsequent tests in the same lit worker.
        try:
            # Continue to execute the illegal opcode at current PC
            gpu_process.Continue()

            # Wait for exception (does NOT occur because the UMD doesn't enable
            # illegal opcode exceptions — this is the expected-failure path).
            if not listener.WaitForEvent(timeout_seconds, event):
                self.fail(f"Timeout waiting for exception after {timeout_seconds}s")

            # Find threads with exception stop reason
            exception_threads = []
            for thread in gpu_process.threads:
                stop_reason = thread.GetStopReason()
                if stop_reason == lldb.eStopReasonException:
                    exception_threads.append(thread)

            self.assertGreater(len(exception_threads), 0,
                              "At least one GPU thread should report exception stop reason")

            exception_thread = exception_threads[0]
            stop_desc = exception_thread.GetStopDescription(256)
            self.assertTrue("Illegal opcode" in stop_desc,
                           f"Stop description should mention 'Illegal opcode', got: '{stop_desc}'")
        finally:
            # Tear down the corrupted GPU run so it cannot leak to the next test.
            process.Kill()
