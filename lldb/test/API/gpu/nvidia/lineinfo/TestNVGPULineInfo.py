"""
Test a program built with line tables (-lineinfo) instead of device debug
information (-G).
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPULineInfo(NVGPUTestCaseBase):
    SOURCE = "main.cu"

    def stop_in_square(self) -> lldb.SBFrame:
        """Build, run to the launch, and stop at a name breakpoint in the
        kernel. Return the stopped thread's frame."""
        self.build()
        self.run_to_cpu_line(self.SOURCE, "// breakpoint before launch")
        breakpoint = self.gpu_target.BreakpointCreateByName("square")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].frame[0]
        self.assertEqual(frame.GetFunctionName(), "square(int*)")
        return frame

    def test_stop_in_kernel_with_line_table_only(self):
        """Test a stop in a kernel whose cubin has a line table and no debug
        information."""
        module = self.stop_in_square().GetModule()
        self.assertTrue(
            module.FindSection(".debug_line").IsValid(),
            f"no .debug_line in {module.GetFileSpec().GetFilename()}",
        )
        self.assertFalse(
            module.FindSection(".debug_info").IsValid(),
            f".debug_info in {module.GetFileSpec().GetFilename()}",
        )

    @expectedFailureAll(
        "lldb has no line entry for a cubin with .debug_line but no .debug_info"
    )
    def test_line_entry_from_line_table(self):
        """Test that the stop in the kernel has a line entry from the line
        table, in the kernel's source. Expected to fail until lldb reads the
        line table of a cubin that has no .debug_info; the mark comes off with
        the change that makes it pass."""
        line_entry = self.stop_in_square().GetLineEntry()
        self.assertTrue(line_entry.IsValid(), "no line entry at the stop")
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), self.SOURCE)
        self.assertGreaterEqual(
            line_entry.GetLine(),
            line_number(self.SOURCE, "__global__ void square(int *data) {"),
        )
