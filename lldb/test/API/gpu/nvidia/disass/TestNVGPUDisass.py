import re

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDisass(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Start of acosf in elfv8.cubin.
    ELFV8_ACOSF = 0x00007FFFCF280300
    SASS_INSTRUCTION_SIZE = 16

    @staticmethod
    def disassembled_addresses(output):
        """The address of each instruction in `disassemble` output."""
        return [
            int(addr, 16) for addr in re.findall(r"(0x[0-9a-f]+)\]? <\+\d+>:", output)
        ]

    @staticmethod
    def disassembled_encodings(output):
        """The encoding of each instruction in `disassemble -b` output."""
        return [
            bytes.fromhex(encoding)
            for encoding in re.findall(r" <\+\d+>: +((?:[0-9a-f]{2} )+)", output)
        ]

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

    def test_disass_count_elf_v8(self):
        """`disassemble -c N` shows exactly N instructions. The odd count
        matters because nvdisasm rejects input that ends partway through an
        instruction."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        for count in (8, 3):
            self.runCmd(f"disassemble -s {self.ELFV8_ACOSF:#x} -c {count}")
            self.assertEqual(
                self.disassembled_addresses(self.res.GetOutput()),
                [
                    self.ELFV8_ACOSF + i * self.SASS_INSTRUCTION_SIZE
                    for i in range(count)
                ],
            )

    def test_disass_bytes_elf_v8(self):
        """`disassemble -b` shows each instruction's 16-byte encoding."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        self.expect(
            f"disassemble -s {self.ELFV8_ACOSF:#x} -c 2 -b",
            patterns=[
                "elfv8.cubin\\[0x7fffcf280300\\] <\\+0>: +"
                "02 72 04 00 04 00 00 00 00 0f 00 00 00 de 3f 00 +MOV +R4,R4",
                "elfv8.cubin\\[0x7fffcf280310\\] <\\+16>: +"
                "21 72 03 ff 04 00 00 40 00 01 00 00 00 de 3f 00 +FADD",
            ],
        )

    def test_disass_unaligned_end_elf_v8(self):
        """A range that ends partway through an instruction disassembles the
        whole instructions it covers."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        self.runCmd(
            f"disassemble -s {self.ELFV8_ACOSF:#x} -e {self.ELFV8_ACOSF + 0x18:#x}"
        )
        self.assertEqual(
            self.disassembled_addresses(self.res.GetOutput()), [self.ELFV8_ACOSF]
        )

    def test_disass_partial_instruction_elf_v8(self):
        """Bytes that hold no whole instruction, or that start off an
        instruction boundary, are refused with the reason."""
        self.runCmd("file " + self.getSourcePath("elfv8.cubin"))
        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitError
        )
        start = self.ELFV8_ACOSF
        for command, reason in (
            (f"disassemble -s {start:#x} -e {start + 8:#x}", "are 16 bytes"),
            (f"disassemble -s {start + 8:#x} -c 2", "are 16-byte aligned"),
        ):
            self.expect(command, error=True)
            event = lldbutil.fetch_next_event(self, listener, broadcaster)
            diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
            self.assertIn(
                reason, diagnostic.GetValueForKey("message").GetStringValue(256)
            )

    def test_read_instructions_elf_v8(self):
        """SBTarget.ReadInstructions(addr, N) returns N instructions, each one
        16 bytes long and holding the bytes it was decoded from."""
        target = self.createTestTarget(self.getSourcePath("elfv8.cubin"))
        self.assertEqual(target.GetMinimumOpcodeByteSize(), self.SASS_INSTRUCTION_SIZE)
        self.assertEqual(target.GetMaximumOpcodeByteSize(), self.SASS_INSTRUCTION_SIZE)

        count = 8
        start = target.ResolveFileAddress(self.ELFV8_ACOSF)
        error = lldb.SBError()
        code = target.ReadMemory(start, count * self.SASS_INSTRUCTION_SIZE, error)
        self.assertSuccess(error)

        instructions = target.ReadInstructions(start, count)
        self.assertEqual(instructions.GetSize(), count)
        for i, inst in enumerate(instructions):
            offset = i * self.SASS_INSTRUCTION_SIZE
            self.assertEqual(
                inst.GetAddress().GetFileAddress(), self.ELFV8_ACOSF + offset
            )
            self.assertEqual(inst.GetByteSize(), self.SASS_INSTRUCTION_SIZE)
            self.assertEqual(
                bytes(inst.GetData(target).uint8s),
                code[offset : offset + self.SASS_INSTRUCTION_SIZE],
            )

    def test_disass_live_program(self):
        """Disassemble at a live GPU assert. The stop shows
        `stop-disassembly-count` instructions from the PC, and
        `disassemble --pc -c N -b` shows N instructions, each with the 16-byte
        encoding the cubin holds at its address."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
        # Look the asserting lane up by its stop reason rather than assuming it is
        # thread 0: faulting-lane selection is not guaranteed.
        thread = self.find_thread_by_stop_reason(lldb.eStopReasonException)
        self.assertIn("CUDA Exception(12): Warp Assert", str(thread))

        self.select_gpu()
        self.gpu_process.SetSelectedThread(thread)
        # Now let's test that the disass can print at least one entry
        self.expect("disassemble", patterns=[".*cuda_elf.*\\.cubin`.*:.*"])

        target = self.gpu_target
        size = self.SASS_INSTRUCTION_SIZE
        frame = thread.GetFrameAtIndex(0)
        pc = frame.GetPC()
        self.assertFalse(
            frame.GetLineEntry().IsValid(),
            "the stop only shows disassembly for a frame without line info",
        )
        self.runCmd("settings set stop-disassembly-display no-debuginfo")
        self.runCmd("settings set stop-disassembly-count 4")
        self.runCmd("frame select 0")
        output = self.res.GetOutput()
        self.assertEqual(
            self.disassembled_addresses(output), [pc + i * size for i in range(4)]
        )
        self.assertRegex(output, rf"(?m)^-> +{pc:#x} <\+\d+>:")

        count = 3
        section = frame.GetPCAddress().GetSection()
        offset = pc - section.GetLoadAddress(target)
        code = bytes(section.GetSectionData(offset, count * size).uint8s)
        expected = [code[i * size : (i + 1) * size] for i in range(count)]
        self.assertNotIn(bytes(size), expected)

        self.runCmd(f"disassemble --pc -c {count} -b")
        output = self.res.GetOutput()
        self.assertEqual(
            self.disassembled_addresses(output), [pc + i * size for i in range(count)]
        )
        self.assertEqual(self.disassembled_encodings(output), expected)

        self.assertEqual(target.GetMinimumOpcodeByteSize(), size)
        self.assertEqual(target.GetMaximumOpcodeByteSize(), size)
        # TODO: DTCLLDB-321: Read by count once ReadInstructions(start, count)
        # works on live GPU targets; it forces a live read of code memory, which
        # fails there.
        end = lldb.SBAddress(pc + count * size, target)
        instructions = target.ReadInstructions(frame.GetPCAddress(), end, None)
        self.assertEqual([inst.GetByteSize() for inst in instructions], [size] * count)
        self.assertEqual(
            [bytes(inst.GetData(target).uint8s) for inst in instructions], expected
        )
