"""
Exception tests for the AMDGPU plugin.
Tests that GPU exceptions are reported as eStopReasonException with correct
descriptions for memory violations, assert traps, and illegal instructions.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


class ExceptionAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_gpu_exception(self, scenario=None):
        """Launch the kernel and wait for the GPU to stop with an exception.
        If scenario is provided, it is passed as a command-line argument to
        select which kernel to launch (memory_violation, assert_trap,
        illegal_instruction).
        """
        self.build()

        target = self.createTestTarget()
        args = [scenario] if scenario else None
        process = target.LaunchSimple(args, None, self.get_process_working_directory())
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

        # Wait for the GPU process to stop due to the exception.
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateStopped]
        )

    def find_exception_thread(self):
        """Find and return a GPU thread stopped with eStopReasonException."""
        for thread in self.gpu_process.threads:
            if thread.GetStopReason() == lldb.eStopReasonException:
                return thread
        return None

    def verify_exception_with_description(self, scenario, expected_description):
        """Run a scenario and verify eStopReasonException with the expected
        description string."""
        self.run_to_gpu_exception(scenario)

        exception_thread = self.find_exception_thread()
        self.assertIsNotNone(
            exception_thread, "Should find a thread stopped with an exception"
        )

        stop_description = exception_thread.GetStopDescription(256)
        self.assertIn(
            expected_description,
            stop_description,
            f"Exception description should contain '{expected_description}', "
            f"got: '{stop_description}'",
        )

    def test_gpu_memory_violation(self):
        """Test that a GPU memory violation reports the correct description."""
        self.verify_exception_with_description(
            "memory_violation", "Memory access violation"
        )

    def test_gpu_assert_trap(self):
        """Test that a GPU assert trap reports the correct description."""
        self.verify_exception_with_description("assert_trap", "Assert trap")

    def test_gpu_illegal_instruction(self):
        """Test that a GPU illegal instruction reports the correct description."""
        self.verify_exception_with_description(
            "illegal_instruction", "Illegal instruction"
        )

    def test_gpu_exception_backtrace(self):
        """Test that we can get a backtrace from an excepting GPU thread."""
        self.run_to_gpu_exception("memory_violation")

        exception_thread = self.find_exception_thread()
        self.assertIsNotNone(
            exception_thread, "Should find a thread stopped with an exception"
        )

        self.assertGreater(
            exception_thread.GetNumFrames(),
            0,
            "Exception thread should have at least one frame in backtrace",
        )

        frame = exception_thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Top frame should be valid")

        frame_name = frame.GetFunctionName()
        self.assertIsNotNone(frame_name, "Top frame should have a function name")
        self.assertIn(
            "memory_violation_kernel",
            frame_name,
            f"Top frame should be in 'memory_violation_kernel', "
            f"got: '{frame_name}'",
        )
