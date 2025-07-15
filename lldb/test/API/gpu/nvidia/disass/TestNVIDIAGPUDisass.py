from lldbsuite.test.lldbtest import TestBase


class TestNVIDIAGPUDisass(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_disass_elf_v7(self):
        """Test that we can disassemble an ELFv7 binary."""
        self.runCmd("file " + self.getSourcePath("elfv7.cubin"))
        self.expect(
            "disassemble -a 0x00007fffd7243b00", substrs=["elfv7.cubin[0x7fffd7243b00] <+0>:", "MOV", "R1,c[0x0][0x28]"]
        )

    def test_disass_elf_v8(self):
        """Test that we can disassemble an ELFv8 binary."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        self.expect("disassemble -a 0x00007fffcf280300", substrs=["elfv8.cubin[0x7fffcf280300] <+0>:", "MOV", "R4,R4"])
