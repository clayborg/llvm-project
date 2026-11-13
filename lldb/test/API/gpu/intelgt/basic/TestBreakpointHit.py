"""
Test that we can hit a GPU breakpoint in IntelGT.

This test attempts to actually HIT a GPU breakpoint and stop execution,
not just SET it. This is the critical functionality needed for debugging.
"""

import lldb
import os
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtBreakpointHitTestCase(IntelGtTestCaseBase):
    """Test GPU breakpoint hit functionality."""

    def test_gpu_breakpoint_actually_hits(self):
        """Test that a GPU breakpoint actually stops execution.

        This is the MVP test case - if this works, IntelGT debugging is functional.

        The working approach for IntelGT:
        1. Set GPU breakpoint BEFORE running (not after kernel submit)
        2. Run the program
        3. GPU breakpoint will hit automatically
        """
        self.build()

        # Create target
        exe = self.getBuildArtifact("simple_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        # Set GPU breakpoint BEFORE running
        # For IntelGT, the breakpoint will remain pending until the GPU target/module
        # is created during launch. Resolution is verified after the GPU target exists.
        source_spec = lldb.SBFileSpec("simple_kernel.cpp", False)
        gpu_bkpt = target.BreakpointCreateBySourceRegex("// GPU BREAKPOINT", source_spec)
        self.assertTrue(gpu_bkpt.IsValid(), "GPU breakpoint should be created")

        # Set up environment
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        # Run the program - it will automatically create GPU target and hit the breakpoint
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")
        self.assertTrue(process.IsValid(), "Process should be valid")

        # Wait for process to stop
        # The GPU target is created and GPU breakpoint hit happens automatically
        state = process.GetState()
        if state == lldb.eStateRunning:
            # Wait for stop
            listener = self.dbg.GetListener()
            event = lldb.SBEvent()
            if not listener.WaitForEvent(30, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()

        # Process should be stopped
        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped")

        # Verify GPU target was created
        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target should be created")
        gpu_target = self.dbg.GetTargetAtIndex(1)
        self.assertTrue(gpu_target.IsValid(), "GPU target should be valid")

        # Get GPU process
        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        # Verify GPU process stopped at breakpoint
        self.assertEqual(gpu_process.GetState(), lldb.eStateStopped,
                        "GPU process should be stopped at breakpoint")

        # Find thread stopped at breakpoint
        stopped_threads = [t for t in gpu_process.threads
                          if t.GetStopReason() == lldb.eStopReasonBreakpoint]
        self.assertGreater(len(stopped_threads), 0, "At least one GPU thread should hit breakpoint")

        gpu_thread = stopped_threads[0]
        self.assertTrue(gpu_thread.IsValid(), "GPU thread should be valid")

        # Verify we're at the right location
        frame = gpu_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Frame should be valid")

        line_entry = frame.GetLineEntry()
        self.assertTrue(line_entry.IsValid(), "Line entry should be valid")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "simple_kernel.cpp",
                        "Should be in simple_kernel.cpp")
        # Line 18 is the GPU BREAKPOINT line
        self.assertEqual(line_entry.GetLine(), 18, "Should stop at GPU breakpoint line 18")

        # Verify thread info
        thread_name = gpu_thread.GetName()
        self.assertTrue("IntelGT EU" in thread_name, f"Thread should be IntelGT EU thread: {thread_name}")

        print(f"SUCCESS: Hit GPU breakpoint at {line_entry.GetFileSpec().GetFilename()}:{line_entry.GetLine()}")
        print(f"  Thread: {thread_name}")
        print(f"  Stop reason: breakpoint")
