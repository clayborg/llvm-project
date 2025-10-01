import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDisass(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_disass_elf_v7(self):
        """Test that we can disassemble an ELFv7 binary."""
        self.runCmd("file " + self.getSourcePath("elfv7.cubin"))
        self.expect(
            "disassemble -a 0x00007fffd7243b00",
            patterns=[
                "elfv7.cubin\\[0x7fffd7243b00\\].*<\\+0>:.*MOV.*R1,c\\[0x0\\]\\[0x28\\]",
                "elfv7.cubin\\[0x7fffd7243b10\\].*<\\+16>:.*S2R.*R2,SR_TID.X",
            ],
        )

    def test_disass_elf_v8(self):
        """Test that we can disassemble an ELFv8 binary."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        self.expect(
            "disassemble -a 0x00007fffcf280300",
            patterns=[
                "elfv8.cubin\\[0x7fffcf280300\\].*<\\+0>:.*MOV.*R4,R4",
                "elfv8.cubin\\[0x7fffcf280310\\].*<\\+16>:.*FADD.*R3,-RZ,|R4|",
            ],
        )

    def test_disass_live_program(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
        self.assertIn("CUDA Exception(12): Warp - Assert", str(self.gpu_process.thread[0]))

        # Now let's test that the disass can print at least one entry
        self.expect("disassemble", patterns=[".*cuda_elf.*\\.cubin`.*:.*"])
