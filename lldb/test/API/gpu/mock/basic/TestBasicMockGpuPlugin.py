"""
Basic tests for the Mock GPU Plugin.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase


class BasicMockGpuTestCase(GpuTestCaseBase):
    def test_mock_gpu_two_targets(self):
        """
        Test that there are two targets:
        one for the native process and one for the mock GPU process.
        """
        self.build()

        # There should be no targets before running the program.
        self.assertEqual(self.dbg.GetNumTargets(), 0, "There are no targets")

        # Set a breakpoint in the CPU source and run to it.
        source_spec = lldb.SBFileSpec("hello_world.cpp", False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = (
            lldbutil.run_to_source_breakpoint(
                self, "// CPU BREAKPOINT - BEFORE LAUNCH", source_spec
            )
        )
        self.assertEqual(self.cpu_target, cpu_target)

        # GPU target is created, has the default thread, and has the correct name.
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")
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

        self.build()

        # Set a breakpoint in the CPU source and run to it.
        source_spec = lldb.SBFileSpec("hello_world.cpp", False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = (
            lldbutil.run_to_source_breakpoint(
                self, "// CPU BREAKPOINT - BEFORE LAUNCH", source_spec
            )
        )

        # Switch to the GPU target and read the registers.
        self.select_gpu()
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        gpu_frame = gpu_thread.GetFrameAtIndex(0)
        gpu_registers = lldbutil.get_registers(gpu_frame, "general purpose")

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
            # Find the register by name in the gpu_registers list
            reg = next((r for r in gpu_registers if r.GetName() == reg_name), None)
            self.assertIsNotNone(reg, f"Register {reg_name} not found")
            self.assertEqual(
                reg.GetValueAsUnsigned(),
                expected_value,
                f"Register {reg_name} value mismatch",
            )

    def test_mock_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint on the gpu target."""

        self.build()

        # Set a breakpoint in the CPU source and run to it.
        source_spec = lldb.SBFileSpec("hello_world.cpp", False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = (
            lldbutil.run_to_source_breakpoint(
                self, "// CPU BREAKPOINT - BEFORE LAUNCH", source_spec
            )
        )

        # Switch to the GPU target and set a breakpoint.
        self.select_gpu()
        (gpu_target, gpu_process, gpu_thread, gpu_bkpt) = (
            lldbutil.run_to_source_breakpoint(
                self,
                "// MOCK GPU BREAKPOINT",
                source_spec,
            )
        )

        # Check the breakpoint was hit.
        self.assertEqual(
            gpu_bkpt.GetHitCount(),
            1,
            "Breakpoint should have been hit once",
        )
