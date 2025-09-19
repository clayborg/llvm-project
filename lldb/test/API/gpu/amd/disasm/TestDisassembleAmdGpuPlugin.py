"""
Disassembly tests for the AMDGPU plugin.
"""

import lldb
import re
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


def normalize_whitespace(s: str) -> str:
    # Replace all runs of (space, tab) with a single space
    return re.sub(r"[\t ]+", " ", s).strip()


class DisassembleAmdGpuTestCase(AmdGpuTestCaseBase):
    def test_disassembly(self):
        """Test that we can disassemble on the gpu target."""
        self.build()

        # Run to the gpu breakpoint
        source = "disasm.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        # Disassemble the current frame
        frame = gpu_threads[0].GetFrameAtIndex(0)
        disassembly = frame.Disassemble()
        disassembly = normalize_whitespace(disassembly)

        # Make sure we see the expected instructions from the inline assembly.
        expected_instructions = [
            "s_not_b32 s27, s28",
            "s_add_u32 s23, s24, s25",
            "s_add_u32 s23, s24, 0x12345678",
            "s_cmp_eq_u32 s29, s30",
            "s_movk_i32 s26, 0x1234",
            "s_nop 1",
            "v_accvgpr_mov_b32 a3, a5",
            "v_add_f32_e32 v17, v18, v19",
            "v_add_f32_e64 v17, v18, 1.0",
            "v_fma_f32 v20, v21, v22, v23",
            "v_accvgpr_read_b32 v17, a3",
            "v_cmp_eq_f32_e32 vcc, v24, v25",
            "v_cmp_eq_f32_e64 vcc, v24, 1.0",
        ]
        for instruction in expected_instructions:
            self.assertIn(instruction, disassembly)
