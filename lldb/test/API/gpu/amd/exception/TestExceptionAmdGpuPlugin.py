"""
Exception tests for the AMDGPU plugin.
Tests that GPU memory violations are reported as eStopReasonException.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


class ExceptionAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_gpu_exception(self):
        """Launch the kernel and wait for the GPU to stop with an exception."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch. The GPU plugin
        # stops the CPU when GPU modules are loaded (auto_resume_native=false).
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be valid")
        self.assertTrue(self.gpu_process.IsValid(), "GPU process should be valid")

        # Run asynchronously so we can manage both CPU and GPU processes.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue the GPU process first (it is waiting).
        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        # Continue the CPU process to launch the kernel.
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        # Wait for the GPU process to stop due to the memory violation.
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateStopped]
        )

        # At least one GPU thread should be stopped with an exception.
        exception_threads = []
        for thread in self.gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonException:
                exception_threads.append(thread)

        self.assertGreater(
            len(exception_threads),
            0,
            "At least one GPU thread should be stopped with eStopReasonException",
        )

    def test_gpu_exception_description(self):
        """Test that the exception description contains useful information."""
        self.run_to_gpu_exception()

        # Find a thread with an exception stop reason.
        exception_thread = None
        for thread in self.gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonException:
                exception_thread = thread
                break

        self.assertIsNotNone(
            exception_thread, "Should find a thread stopped with an exception"
        )

        # The stop description should indicate a memory access violation.
        stop_description = exception_thread.GetStopDescription(256)
        self.assertIn(
            "Memory access violation",
            stop_description,
            f"Exception stop description should contain 'Memory access violation', "
            f"got: '{stop_description}'",
        )

    def test_gpu_exception_backtrace(self):
        """Test that we can get a backtrace from the excepting GPU thread."""
        self.run_to_gpu_exception()

        # Find a thread with an exception stop reason.
        exception_thread = None
        for thread in self.gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonException:
                exception_thread = thread
                break

        self.assertIsNotNone(
            exception_thread, "Should find a thread stopped with an exception"
        )

        # We should be able to get at least one frame from the backtrace.
        self.assertGreater(
            exception_thread.GetNumFrames(),
            0,
            "Exception thread should have at least one frame in backtrace",
        )

        # The top frame should be in the kernel function.
        frame = exception_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Top frame should be valid")

        # Verify the top frame is in the memory_violation_kernel function.
        frame_name = frame.GetFunctionName()
        self.assertIsNotNone(frame_name, "Top frame should have a function name")
        self.assertIn(
            "memory_violation_kernel",
            frame_name,
            f"Top frame should be in 'memory_violation_kernel', "
            f"got: '{frame_name}'",
        )
