"""
Disassembly tests for the AMDGPU plugin.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


class DisassembleAmdGpuTestCase(AmdGpuTestCaseBase):
    def test_disassembly(self):
        """Test that we can disassemble on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "disasm.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        frame = gpu_threads[0].GetFrameAtIndex(0)
        disassembly = frame.Disassemble()
        self.assertIn("s_mov_b32 s33, 0", disassembly)
