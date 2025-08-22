"""
Register tests for the AMDGPU plugin.
"""

import lldb
from amdgpu_testcase import AmdGpuTestCaseBase
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

# Expected size of a wavefront in threads.
WAVE_SIZE = 64
NUM_VECTOR_REGISTERS = 256
NUM_ACCUM_REGISTERS = NUM_VECTOR_REGISTERS
NUM_SCALAR_REGISTERS = 102


class RegisterAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_reg_gpu_breakpoint(self, reg_base):
        source = "reg.hip"
        gpr = f"{reg_base.upper()}GPR"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, f"// GPU BREAKPOINT - {gpr}", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")
        self.select_gpu()

    def test_vgrp_read_write(self):
        self.do_reg_read_write_test("v", NUM_VECTOR_REGISTERS)

    def test_sgrp_read_write(self):
        self.do_reg_read_write_test("s", NUM_SCALAR_REGISTERS)

    @expectedFailureAll("Read of initial AGPR register valuee is working yet")
    def test_agrp_read_write(self):
        self.do_reg_read_write_test("a", NUM_ACCUM_REGISTERS)

    def test_vgrp_lane_write(self):
        self.do_lane_read_write_test("v", NUM_VECTOR_REGISTERS)

    def test_agrp_lane_write(self):
        self.do_lane_read_write_test("a", NUM_ACCUM_REGISTERS)

    def do_reg_read_write_test(self, reg_base, num_regs):
        """Verify we can read and write the whole register value"""
        self.build()
        self.run_to_reg_gpu_breakpoint(reg_base)

        for i in range(num_regs):
            reg, known_value, new_value = self.get_test_values(reg_base, i, num_regs)
            self.verify_reg_read_and_write(reg, known_value, new_value)

    def do_lane_read_write_test(self, reg_base, num_regs):
        """Verify we can read and write the individual lanes of a register"""
        self.build()
        self.run_to_reg_gpu_breakpoint(reg_base)

        for i in range(num_regs):
            reg = f"{reg_base}{i}"
            new_value = [0] * WAVE_SIZE
            self.verify_reg_write(reg, new_value)

            for lane in range(WAVE_SIZE):
                new_value[lane] = self.replicate_byte(lane + 1)
                self.verify_reg_write(reg, new_value)

    def replicate_byte(self, value, minval=0):
        """Replicate the byte across the 4 bytes of a 32-bit value."""
        value = max(value & 0xFF, minval)
        return (value << 24) | (value << 16) | (value << 8) | value

    def get_test_values(self, reg_base, i, num_regs):
        """Return the register to read, the known initialized value of that register, and a new value to write."""
        # For a register v{i} we replicate the byte {i} across the 4 bytes of the register.
        # For the new value we want to make write a non-zero value that is different
        # from the original value.  We do this by replicating the byte max({num_regs - i}, 1)
        # across the 4 bytes of the register.
        reg = f"{reg_base}{i}"
        original_value = self.replicate_byte(i)
        new_value = self.replicate_byte(num_regs - i, minval=1)

        if self.is_vector_reg(reg_base):
            original_value = [original_value] * WAVE_SIZE
            new_value = [new_value] * WAVE_SIZE
        return reg, original_value, new_value

    def verify_reg_read_and_write(self, reg, original_value, new_value):
        """Verify that we can read and write the register with known values."""
        # First read the register and verify the expected value.
        self.verify_reg_read(reg, original_value)

        # Now write the new value back the register.
        self.verify_reg_write(reg, new_value)

    def verify_reg_read(self, gpr, expected):
        """Read a value from a register and verify the result."""
        reg_value_str = f"{gpr} = " + self.get_reg_value_str(gpr, expected)
        self.expect(f"register read {gpr}", substrs=[reg_value_str])

    def verify_reg_write(self, gpr, value):
        """Write the value to a register and verify we read back the same value."""
        self.reg_write(gpr, value)
        self.verify_reg_read(gpr, value)

    def reg_write(self, gpr, value):
        """Write a value to a register."""
        reg_value = self.get_reg_value_str(gpr, value)
        self.runCmd(f'register write {gpr} "{reg_value}"')

    def is_vector_reg(self, gpr):
        """Return true if the register is a vector register."""
        return gpr.startswith("v") or gpr.startswith("a")

    def get_reg_value_str(self, gpr, value):
        """Return the string representation used by lldb for the value in the register.
        Expands the value to the correct width if needed for vector registers.
        """
        if self.is_vector_reg(gpr):
            return self.get_vector_reg_str(value)
        else:
            return self.get_scalar_reg_str(value)

    def get_vector_reg_str(self, lanes):
        """Return the string representation used for a vector register class."""
        self.assertTrue(isinstance(lanes, list), "lanes must be a list")
        self.assertTrue(len(lanes) == WAVE_SIZE, "lanes must be WAVE_SIZE long")
        reg_value = "{"
        reg_value += " ".join([self.convert_32bit_value_to_bytes(v) for v in lanes])
        reg_value += "}"

        return reg_value

    def get_scalar_reg_str(self, value):
        return f"0x{value:08x}"

    def convert_32bit_value_to_bytes(self, value):
        """Convert to a space separated list of bytes in little-endian order."""
        le_byte_values = [(value >> (8 * i)) & 0xFF for i in range(4)]
        return " ".join([f"0x{b:02x}" for b in le_byte_values])
