"""
Basic tests for the Mock GPU Plugin.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

CPU_BEFORE_BREAKPOINT_COMMENT = "// CPU BREAKPOINT - BEFORE INITIALIZE"
CPU_AFTER_BREAKPOINT_COMMENT = "// CPU BREAKPOINT - AFTER INITIALIZE"
GPU_BREAKPOINT_COMMENT = "// MOCK GPU BREAKPOINT"
SOURCE_FILE = "hello_world.cpp"


class BasicMockGpuTestCase(GpuTestCaseBase):
    def common_setup(self):
        """Build the test program and run to the CPU breakpoint."""
        super().setUp()
        self.build()
        self.source_spec = lldb.SBFileSpec(SOURCE_FILE, False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = lldbutil.run_to_source_breakpoint(
            self, CPU_BEFORE_BREAKPOINT_COMMENT, self.source_spec
        )

    def test_mock_gpu_two_targets(self):
        """
        Verify that two targets exist: one CPU and one mock GPU.
        Ensures the GPU thread is correctly named.
        """
        self.common_setup()
        # Check that there is one CPU target before GPU is initialized.
        self.assertEqual(self.dbg.GetNumTargets(), 1, "There is one CPU target")

        # Continue to the breakpoint after GPU is initialized.
        lldbutil.continue_to_source_breakpoint(self, self.cpu_process, CPU_AFTER_BREAKPOINT_COMMENT, self.source_spec)

        # Check that there are two targets.
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")

        # Check the CPU target.
        self.assertIsNotNone(self.cpu_target, "CPU target should exist")
        self.assertTrue(self.cpu_target.GetProcess().IsValid(), "CPU process is valid")

        # Check the GPU target.
        self.assertIsNotNone(self.gpu_target, "GPU target should exist")
        self.assertTrue(self.gpu_process.IsValid(), "GPU process is valid")
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        self.assertEqual(
            gpu_thread.GetName(),
            "Mock GPU Thread Name",
            "GPU thread has the right name",
        )

    def test_mock_gpu_register_read(self):
        """
        Test that we can read registers from the mock GPU target
        and the "fake" register values are correct.
        """
        self.common_setup()
        # Continue to the breakpoint after GPU is initialized.
        lldbutil.continue_to_source_breakpoint(self, self.cpu_process, CPU_AFTER_BREAKPOINT_COMMENT, self.source_spec)

        # Switch to the GPU target and read the registers.
        self.select_gpu()
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        gpu_frame = gpu_thread.GetFrameAtIndex(0)
        gpu_registers_raw = lldbutil.get_registers(gpu_frame, "general purpose")

        # Convert the registers to a dictionary for easier checking.
        gpu_registers = {reg.GetName(): reg.GetValueAsUnsigned() for reg in gpu_registers_raw}

        expected_registers = {
            "R0": 0x0,
            "R1": 0x1,
            "R2": 0x2,
            "R3": 0x3,
            "R4": 0x4,
            "R5": 0x5,
            "R6": 0x6,
            "R7": 0x7,
            "SP": 0x8,
            "FP": 0x9,
            "PC": 0xA,
            "Flags": 0xB,
        }
        for reg_name, expected_value in expected_registers.items():
            self.assertIn(reg_name, gpu_registers)
            self.assertEqual(
                gpu_registers[reg_name],
                expected_value,
                f"Register {reg_name} value mismatch",
            )

    def test_mock_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint on the gpu target."""
        # Switch to the GPU target and set a breakpoint.
        self.common_setup()
        self.select_gpu()
        (gpu_target, gpu_process, gpu_thread, gpu_bkpt) = lldbutil.run_to_source_breakpoint(
            self,
            GPU_BREAKPOINT_COMMENT,
            self.source_spec,
        )

        # Check the breakpoint was hit.
        self.assertEqual(
            gpu_bkpt.GetHitCount(),
            1,
            "Breakpoint should have been hit once",
        )

    def test_copy_cpu_breakpoints_during_attaching(self):
        """Test that we can copy CPU breakpoints during attaching."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd(f"file {exe}")
        gpu_breakpoint_comment_line = line_number(SOURCE_FILE, GPU_BREAKPOINT_COMMENT)

        self.runCmd(f"b {gpu_breakpoint_comment_line}")
        self.runCmd("r")

        self.assertTrue(self.gpu_process.IsValid())

    def test_mock_gpu_target_switch(self):
        """Test that we can switch targets between CPU and GPU."""

        self.common_setup()
        # Continue to the breakpoint after GPU is initialized.
        lldbutil.continue_to_source_breakpoint(self, self.cpu_process, CPU_AFTER_BREAKPOINT_COMMENT, self.source_spec)

        self.select_gpu()

        self.runCmd("target switch")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.cpu_target)

        self.runCmd("target switch")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.gpu_target)

        self.runCmd("target switch")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.cpu_target)

        self.runCmd("target switch mock-gpu")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.gpu_target)

        self.runCmd("target switch mock-gpu")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.gpu_target)

        self.runCmd("target switch cpu")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.cpu_target)

        self.runCmd("target switch cpu")
        self.assertTrue(self.dbg.GetSelectedTarget() == self.cpu_target)
    def test_mock_gpu_first_breakpoint_hit(self):
        """
        Test that we can hit the first breakpoint on the cpu target,
        Making sure we handle the GPUActions on the first stop-reply.
        """
        self.common_setup()

        self.select_cpu()
        self.expect(
            "breakpoint list --internal",
            patterns=[r"name = 'gpu_first_stop'.*hit count = 1"],
        )
