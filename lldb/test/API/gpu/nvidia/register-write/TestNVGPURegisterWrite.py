from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


SOURCE = "register_write.cu"


class TestNVGPURegisterWrite(NVGPUTestCaseBase):
    def test_register_writes(self):
        self.build()
        breakpoint = self.run_to_gpu_line(SOURCE, "// register write breakpoint")
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]

        # Round-trip each writable register class.
        self.write_register(thread, "R0", 0xDEAD)
        self.write_register(thread, "R3", 0xBEEF)

        # SP and FP are R1 and R2 under other names.
        for alias, register, value in (("SP", "R1", 0x5150), ("FP", "R2", 0x4650)):
            self.write_register(thread, alias, value)
            self.assertEqual(
                self.read_register(thread.GetFrameAtIndex(0), register),
                value,
                f"{register} after writing {alias}",
            )

        # Exercise the server-side composite RA write directly.
        self.write_register_behind_lldb(thread, "RA", 0x123456789ABCDEF0)

        # Write the value that P0 does not hold already.
        p0 = self.read_register(thread.GetFrameAtIndex(0), "P0")
        self.write_register(thread, "P0", p0 ^ 1)

        # Reject writes to architectural and synthesized read-only registers.
        for name, value in (
            ("pc", 0),
            ("errorpc", 0),
            ("RZ", 5),
            ("URZ", 5),
            ("warpSize", 8),
        ):
            self.expect(
                f"register write {name} {value}",
                error=True,
                substrs=["Failed to write register"],
            )

        # A write to a register that the function does not use fails.
        frame = thread.GetFrameAtIndex(0)
        unused = next(
            f"R{n}"
            for n in range(255)
            if frame.FindRegister(f"R{n}").GetValue() is None
        )
        self.expect(
            f"register write {unused} 1",
            error=True,
            substrs=["Failed to write register"],
        )
