"""
Basic tests for the Mock GPU Plugin.
"""

import json
import re

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

CPU_BEFORE_BREAKPOINT_COMMENT = "// CPU BREAKPOINT - BEFORE INITIALIZE"
CPU_AFTER_BREAKPOINT_COMMENT = "// CPU BREAKPOINT - AFTER INITIALIZE"
GPU_BREAKPOINT_COMMENT = "// MOCK GPU BREAKPOINT"
SOURCE_FILE = "hello_world.cpp"


def get_sent_symbols(log_file):
    """Return the name and value of each symbol that lldb sent in a
    jGPUPluginSymbols packet, in the order sent, from a gdb-remote packet log."""
    sent = []
    with open(log_file) as log:
        for line in log:
            _, found, packet = line.partition("send packet: $jGPUPluginSymbols:")
            if not found:
                continue
            # Drop the checksum, then undo the escaping of special characters
            # as '}' followed by the character XOR 0x20.
            payload = re.sub(
                r"\}(.)",
                lambda match: chr(ord(match.group(1)) ^ 0x20),
                packet.rpartition("#")[0],
            )
            args = json.loads(payload)
            sent.extend((s["name"], s.get("value")) for s in args["symbol_values"])
    return sent


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

    def test_mock_gpu_thread_extended_info(self):
        """
        Test that the jThreadExtendedInfo packet returns the JSON supplied by
        ThreadMockGPU::GetExtendedInfo().
        """
        self.common_setup()
        # Continue to the breakpoint after GPU is initialized.
        lldbutil.continue_to_source_breakpoint(self, self.cpu_process, CPU_AFTER_BREAKPOINT_COMMENT, self.source_spec)

        self.select_gpu()
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)

        expected_items = {
            "name": "Mock GPU Thread Name",
            # ProcessMockGPU::UpdateThreads() makes its thread inactive.
            "is_active": "false",
            "dispatch.kernel_name": "mock_gpu_kernel",
        }
        for path, expected_value in expected_items.items():
            stream = lldb.SBStream()
            self.assertTrue(
                gpu_thread.GetInfoItemByPathAsString(path, stream),
                f'extended thread info has a "{path}" item',
            )
            self.assertEqual(stream.GetData(), expected_value)

    def test_mock_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint in the mock GPU kernel."""
        self.common_setup()
        # The mock GPU process can't set breakpoints, and its kernel is CPU
        # code, so the breakpoint goes in the CPU target. Reaching it means
        # continuing past the plug-in's breakpoints in gpu_initialize,
        # gpu_shlib_load and gpu_third_stop.
        gpu_bkpt = self.cpu_target.BreakpointCreateBySourceRegex(
            GPU_BREAKPOINT_COMMENT, self.source_spec
        )
        threads = lldbutil.continue_to_breakpoint(self.cpu_process, gpu_bkpt)
        self.assertTrue(threads, "The CPU process stopped at the breakpoint")

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

    def test_mock_gpu_lane_and_simd_ids(self):
        """
        Test that a GPU thread reports the lane and SIMD group it belongs to,
        and that a CPU thread reports neither.
        """
        self.common_setup()

        # Continue to the breakpoint after GPU is initialized.
        lldbutil.continue_to_source_breakpoint(
            self, self.cpu_process, CPU_AFTER_BREAKPOINT_COMMENT, self.source_spec
        )

        # ProcessMockGPU::UpdateThreads() creates a single thread with these
        # values, so they should come back with these values
        self.select_gpu()
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        self.assertTrue(gpu_thread.IsValid())
        self.assertEqual(gpu_thread.GetThreadID(), 3456)
        self.assertEqual(gpu_thread.GetLaneID(), 0)
        self.assertEqual(gpu_thread.GetSIMD(), 1)

        # A CPU thread has no lane or SIMD group, so both should come back
        # invalid.
        self.select_cpu()
        cpu_thread = self.cpu_process.GetThreadAtIndex(0)
        self.assertTrue(cpu_thread.IsValid())
        self.assertEqual(cpu_thread.GetLaneID(), lldb.LLDB_INVALID_LANE_ID)
        self.assertEqual(cpu_thread.GetSIMD(), lldb.LLDB_INVALID_SIMD_ID)

    def test_mock_gpu_requested_symbols(self):
        """
        Test that lldb sends the plug-in each symbol it asks for once, with its
        load address, and doesn't send names that don't resolve.
        """
        log_file = self.getBuildArtifact("packets.log")
        self.runCmd(f"log enable -f {log_file} gdb-remote packets")
        self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))
        self.build()
        target, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, CPU_BEFORE_BREAKPOINT_COMMENT, lldb.SBFileSpec(SOURCE_FILE, False)
        )

        # The plug-in's initialize actions ask for "gpu_shlib_load", before
        # lldb knows where the executable is loaded, and for
        # "invalid_name_that_wont_be_found". Its reply to "gpu_shlib_load"
        # asks for "gpu_kernel".
        expected = []
        for name in ("gpu_shlib_load", "gpu_kernel"):
            symbols = target.FindSymbols(name)
            self.assertEqual(symbols.GetSize(), 1, f'one symbol is named "{name}"')
            address = symbols.GetContextAtIndex(0).GetSymbol().GetStartAddress()
            expected.append((name, address.GetLoadAddress(target)))
        self.assertEqual(get_sent_symbols(log_file), expected)
