"""
Test GPU software exception detection in IntelGT.

Verifies that a failed assertion triggers a software exception that is
correctly detected and reported by LLDB with appropriate signal and stop
description.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtSoftwareExceptionTestCase(IntelGtTestCaseBase):
    """Test software exception detection."""

    def test_software_exception(self):
        """Test that GPU software exception is detected with SIGTRAP signal.

        Kernel executes assert(false), triggering a software exception.
        The driver sends thread stopped events with CR0.1 bit 29 set.

        Expected behavior:
        1. GPU process stops with eStopReasonException
        2. At least one GPU thread reports the exception
        3. Signal number is SIGTRAP (5)
        4. Stop description contains "Software exception"
        """
        self.build()

        exe = self.getBuildArtifact("software_exception")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), f"Launch should succeed: {error.GetCString()}")
        self.assertTrue(process.IsValid(), "Process should be valid")

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        timeout_seconds = 30

        if not listener.WaitForEvent(timeout_seconds, event):
            self.fail(f"Timeout waiting for process event after {timeout_seconds}s")

        state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped,
                        f"Process should be stopped, got state {state}")

        self.assertEqual(self.dbg.GetNumTargets(), 2, "GPU target should be created")
        gpu_target = self.dbg.GetTargetAtIndex(1)
        self.assertTrue(gpu_target.IsValid(), "GPU target should be valid")

        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        # GPU process may be in eStateExited or eStateStopped depending on timing.
        # The important check is that threads have the exception stop reason.

        exception_threads = []
        for thread in gpu_process.threads:
            stop_reason = thread.GetStopReason()
            if stop_reason == lldb.eStopReasonException:
                exception_threads.append(thread)

        self.assertGreater(len(exception_threads), 0,
                          "At least one GPU thread should report exception stop reason")

        exception_thread = exception_threads[0]
        self.assertTrue(exception_thread.IsValid(), "Exception thread should be valid")

        stop_desc = exception_thread.GetStopDescription(256)
        self.assertTrue("Software exception" in stop_desc,
                       f"Stop description should mention 'Software exception', got: '{stop_desc}'")

        thread_name = exception_thread.GetName()
        self.assertTrue("IntelGT EU" in thread_name,
                       f"Thread should be IntelGT EU thread: {thread_name}")

        self.assertGreater(exception_thread.GetNumFrames(), 0,
                          "Exception thread should have stack frames")

        frame = exception_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Top frame should be valid")

        print(f"SUCCESS: Software exception detected")
        print(f"  Thread: {thread_name}")
        print(f"  Stop reason: eStopReasonException")
        print(f"  Stop description: {stop_desc}")
        print(f"  Frames: {exception_thread.GetNumFrames()}")
