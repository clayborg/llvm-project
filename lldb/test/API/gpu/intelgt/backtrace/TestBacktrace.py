"""
Test GPU thread backtrace / stack unwinding in IntelGT.

Verifies that LLDB can unwind GPU call frames and that frame 0 reports
the correct source location. Based on the manual test
lldb/test/Shell/gpu/intelgt/manual-tests/test-backtrace.exp.

The kernel has two non-inlined helper functions (middle -> inner) so
that bt can produce multiple frames when the compiler honours noinline.
The test only requires frame 0 to be valid; additional frames are
checked when present.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtBacktraceTestCase(IntelGtTestCaseBase):
    """Test GPU thread backtrace / stack frame unwinding."""

    def _stop_at_kernel_breakpoint(self, pattern):
        """Build, launch, and return the GPU thread stopped at *pattern*."""
        self.build()
        exe = self.getBuildArtifact("backtrace_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex(
            pattern, lldb.SBFileSpec("backtrace_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0, "Breakpoint should have a location")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{k}={v}" for k, v in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(30, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped, "Host process should be stopped")

        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target should be created")
        gpu_process = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        stopped_thread = None
        for thread in gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_thread = thread
                break
        self.assertIsNotNone(stopped_thread, "Should have a GPU thread stopped at breakpoint")
        return stopped_thread

    # ------------------------------------------------------------------

    def test_backtrace_frame0_location(self):
        """frame 0 reports the correct source file and line."""
        thread = self._stop_at_kernel_breakpoint("backtrace-inner")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "frame 0 should be valid")

        line_entry = frame.GetLineEntry()
        self.assertTrue(line_entry.IsValid(), "frame 0 line entry should be valid")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "backtrace_kernel.cpp",
                         "frame 0 should be in backtrace_kernel.cpp")
        print(f"frame 0: {line_entry.GetFileSpec().GetFilename()}:{line_entry.GetLine()}")

    def test_backtrace_has_frames(self):
        """bt produces multiple frames via framedesc unwinding.

        The GPU unwinder reads r127 (framedesc register) to reconstruct the
        call chain. With noinline functions the full chain inner -> middle ->
        kernel lambda should be visible.
        """
        thread = self._stop_at_kernel_breakpoint("backtrace-inner")

        num_frames = thread.GetNumFrames()
        self.assertGreater(num_frames, 1, "Thread should have more than one frame")
        print(f"Total frames: {num_frames}")

        for i in range(num_frames):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid(), f"frame {i} should be valid")
            func = frame.GetFunctionName()
            line_entry = frame.GetLineEntry()
            loc = (f"{line_entry.GetFileSpec().GetFilename()}:{line_entry.GetLine()}"
                   if line_entry.IsValid() else "<no location>")
            print(f"  frame #{i}: {func} at {loc}")

        func_names = [thread.GetFrameAtIndex(i).GetFunctionName() or ""
                      for i in range(num_frames)]
        self.assertTrue(any("inner" in f for f in func_names), "inner should appear in bt")
        self.assertTrue(any("middle" in f for f in func_names), "middle should appear in bt")

    def test_backtrace_frame_select(self):
        """Selecting frame 0 returns a valid frame with a live PC."""
        thread = self._stop_at_kernel_breakpoint("backtrace-inner")

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "frame 0 should be selectable")
        pc = frame0.GetPC()
        self.assertNotEqual(pc, lldb.LLDB_INVALID_ADDRESS, "frame 0 PC should be valid")
        print(f"frame 0 PC: 0x{pc:x}")

    def test_thread_backtrace(self):
        """thread backtrace (all frames) returns consistent results."""
        thread = self._stop_at_kernel_breakpoint("backtrace-top")

        # GetNumFrames() and iterating frames must be consistent.
        num_frames = thread.GetNumFrames()
        self.assertGreater(num_frames, 0, "Thread should have frames")

        # All frames must be valid.
        for i in range(num_frames):
            self.assertTrue(thread.GetFrameAtIndex(i).IsValid(),
                            f"frame {i} should be valid in thread backtrace")
        print(f"thread backtrace: {num_frames} frame(s)")

    def test_finish_from_inner(self):
        """finish command works correctly from innermost frame."""
        thread = self._stop_at_kernel_breakpoint("backtrace-inner")

        # Should be stopped in inner()
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIn("inner", frame0.GetFunctionName())

        # Execute finish - should step out to middle()
        process = thread.GetProcess()
        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Success(), f"finish failed: {error.GetCString()}")

        # Wait for thread to stop
        state = process.GetState()
        listener = process.GetTarget().GetDebugger().GetListener()
        event = lldb.SBEvent()
        timeout = 10  # seconds
        if state == lldb.eStateRunning:
            if listener.WaitForEvent(timeout, event):
                state = process.GetState()

        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped after finish")

        # Should now be in middle()
        frame0 = thread.GetFrameAtIndex(0)
        func_name = frame0.GetFunctionName()
        print(f"After finish, stopped in: {func_name}")
        self.assertIn("middle", func_name, "finish should stop in middle()")

    def test_finish_through_multiple_frames(self):
        """finish command works through multiple frames without stopping early."""
        thread = self._stop_at_kernel_breakpoint("backtrace-inner")

        initial_frames = thread.GetNumFrames()
        print(f"Starting with {initial_frames} frames")

        # Should be in inner()
        self.assertIn("inner", thread.GetFrameAtIndex(0).GetFunctionName())

        # First finish: inner -> middle
        process = thread.GetProcess()
        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Success(), f"First finish failed: {error.GetCString()}")

        # Wait for stop
        state = process.GetState()
        listener = process.GetTarget().GetDebugger().GetListener()
        event = lldb.SBEvent()
        if state == lldb.eStateRunning:
            listener.WaitForEvent(10, event)
            state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped)

        # Verify we're in middle()
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIn("middle", frame0.GetFunctionName())
        print(f"After first finish: {frame0.GetFunctionName()}")

        # Second finish: middle -> lambda
        thread.StepOut(error)
        self.assertTrue(error.Success(), f"Second finish failed: {error.GetCString()}")

        # Wait for stop
        if process.GetState() == lldb.eStateRunning:
            listener.WaitForEvent(10, event)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        # Should be in lambda (or beyond)
        frame0 = thread.GetFrameAtIndex(0)
        func_name = frame0.GetFunctionName()
        print(f"After second finish: {func_name}")
        self.assertNotIn("inner", func_name, "Should have finished past inner()")
        self.assertNotIn("middle", func_name, "Should have finished past middle()")
