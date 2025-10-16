"""
Test NVGPU stack unwinding in various scenarios.
"""

import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUUnwind(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_backtrace(self, test_name, expected_frames):
        """Verify that the backtrace contains the expected frames.

        expected_frames: list of (func_name, file_name, expected_line) tuples
        """
        thread = self.gpu_process.thread[0]

        self.assertGreaterEqual(
            thread.GetNumFrames(),
            len(expected_frames),
            f"{test_name}: Expected at least {len(expected_frames)} frames",
        )

        for i, (expected_func, expected_file, expected_line) in enumerate(expected_frames):
            frame = thread.GetFrameAtIndex(i)
            self.assertIsNotNone(frame, f"{test_name}: Frame {i} should exist")

            # Skip validation for frames without debug info
            if expected_func is None:
                continue

            actual_func = frame.GetFunctionName()
            self.assertIsNotNone(
                actual_func,
                f"{test_name}: Frame {i} function name is None",
            )
            self.assertIn(
                expected_func,
                actual_func,
                f"{test_name}: Frame {i} expected '{expected_func}', got '{actual_func}'",
            )

            if expected_file:
                actual_file = frame.GetLineEntry().GetFileSpec().GetFilename()
                self.assertEqual(
                    expected_file,
                    actual_file,
                    f"{test_name}: Frame {i} file mismatch",
                )

            if expected_line is not None:
                actual_line = frame.GetLineEntry().GetLine()
                self.assertEqual(
                    expected_line,
                    actual_line,
                    f"{test_name}: Frame {i} expected line {expected_line}, got {actual_line}",
                )

        # Verify backtrace terminates properly
        self.assertLess(thread.GetNumFrames(), 20, f"{test_name}: Too many frames")

    def test_unwinding(self):
        """Test NVGPU stack unwinding in various scenarios."""
        self.killCPUOnTeardown()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd(f"file {exe}")
        source = "unwind.cu"
        cpu_bp_line = line_number(source, "// cpu breakpoint")
        gpu_bp_line = line_number(source, "// gpu breakpoint")
        exit_bp_line = line_number(source, "// breakpoint before exit")

        self.runCmd(f"b {gpu_bp_line}")
        self.runCmd(f"b {cpu_bp_line}")
        self.runCmd(f"b {exit_bp_line}")
        self.runCmd("r")

        # Wait for first GPU breakpoint to be hit
        self.continue_cpu_and_wait_for_gpu_to_stop()

        # Verify GPU target was created
        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.select_gpu()
        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

        expected_frames = [
            ("breakpoint", "unwind.cu", line_number(source, "// gpu breakpoint, frame_breakpoint")),
            ("level3", "unwind.cu", line_number(source, "// frame_level3")),
            ("level2", "unwind.cu", line_number(source, "// frame_level2")),
            ("level1", "unwind.cu", line_number(source, "// frame_level1")),
            ("level0", "unwind.cu", line_number(source, "// frame_level0")),
            ("unwind_test_kernel", "unwind.cu", line_number(source, "// frame_kernel_test1")),
        ]
        self.check_backtrace("test_unwind_no_arguments", expected_frames)

        # Verify RA register is composite of R20/R21 (while we're here)
        frame = self.gpu_process.thread[0].frame[0]
        ra_reg = frame.FindRegister("RA")
        self.assertTrue(ra_reg.IsValid())
        self.assertEqual(ra_reg.GetByteSize(), 8)

        r20_reg = frame.FindRegister("R20")
        r21_reg = frame.FindRegister("R21")
        self.assertTrue(r20_reg.IsValid())
        self.assertTrue(r21_reg.IsValid())

        ra_value = ra_reg.GetValueAsUnsigned()
        r20_value = r20_reg.GetValueAsUnsigned()
        r21_value = r21_reg.GetValueAsUnsigned()
        expected_ra = r20_value | (r21_value << 32)
        self.assertEqual(ra_value, expected_ra)

        # Reset async mode
        self.dbg.SetAsync(False)

        # Test 2: Unwind with arguments
        self.gpu_process.Continue()
        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

        expected_frames = [
            ("breakpoint", "unwind.cu", line_number(source, "// gpu breakpoint, frame_breakpoint")),
            ("level3_with_arguments", "unwind.cu", line_number(source, "// frame_level3_with_arguments")),
            ("level2_with_arguments", "unwind.cu", line_number(source, "// frame_level2_with_arguments")),
            ("level1_with_arguments", "unwind.cu", line_number(source, "// frame_level1_with_arguments")),
            ("level0_with_arguments", "unwind.cu", line_number(source, "// frame_level0_with_arguments")),
            ("unwind_test_kernel", "unwind.cu", line_number(source, "// frame_kernel_test2")),
        ]
        self.check_backtrace("test_unwind_with_arguments", expected_frames)

        # Test 3: Unwind with divergent control flow
        self.gpu_process.Continue()
        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

        # fmt: off
        expected_frames = [
            ("breakpoint", "unwind.cu", line_number(source, "// gpu breakpoint, frame_breakpoint")),
            ("level3_with_divergent_control_flow", "unwind.cu", line_number(source, "// frame_level3_with_divergent_control_flow")),
            ("level2_with_divergent_control_flow", "unwind.cu", line_number(source, "// frame_level2_with_divergent_control_flow")),
            ("level1_with_divergent_control_flow", "unwind.cu", line_number(source, "// frame_level1_with_divergent_control_flow")),
            ("level0_with_divergent_control_flow", "unwind.cu", line_number(source, "// frame_level0_with_divergent_control_flow")),
            ("unwind_test_kernel", "unwind.cu", line_number(source, "// frame_kernel_test3")),
        ]
        # fmt: on
        self.check_backtrace("test_unwind_divergent_control_flow", expected_frames)

        # Test 4: Unwind with null function pointer (no breakpoint - hits fault directly)
        # Note: This must be the last test because it hits a fault and stops the process
        self.gpu_process.Continue()
        self.select_gpu()
        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)

        expected_frames = [
            (None, None, None),  # Invalid address (no debug info)
            ("call_function", "unwind.cu", line_number(source, "// frame_call_function")),
            ("level3_null_call", "unwind.cu", line_number(source, "// frame_level3_null_call")),
            ("level2_null_call", "unwind.cu", line_number(source, "// frame_level2_null_call")),
            ("level1_null_call", "unwind.cu", line_number(source, "// frame_level1_null_call")),
            ("level0_null_call", "unwind.cu", line_number(source, "// frame_level0_null_call")),
            ("unwind_test_kernel", "unwind.cu", line_number(source, "// frame_kernel_test4")),
        ]
        self.check_backtrace("test_unwind_null_function_ptr", expected_frames)
