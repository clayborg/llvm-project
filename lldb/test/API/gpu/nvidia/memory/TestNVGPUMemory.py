import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUMemory(NVGPUTestCaseBase):
    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.killCPUOnTeardown()

        self.build()
        source = "memory.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        d_arr_addr = self.cpu_process.thread[0].frame[0].FindVariable("d_arr").GetValueAsAddress()
        if self.TraceOn:
            print(f"d_arr_addr: {d_arr_addr}")

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.select_gpu()

        # This doesn't require a thread for reading.
        self.expect(
            f"memory read -p global {d_arr_addr} --format x --size 4 -c 16",
            substrs=[
                "0x00000000 0x00000001 0x00000002 0x00000003",
                "0x00000004 0x00000005 0x00000006 0x00000007",
                "0x00000008 0x00000009 0x0000000a 0x0000000b",
                "0x0000000c 0x0000000d 0x0000000e 0x0000000f",
            ],
        )

        # This requires a thread for reading.
        self.expect(
            f"memory read -p generic {d_arr_addr} --format x --size 4 -c 16",
            substrs=[
                "0x00000000 0x00000001 0x00000002 0x00000003",
                "0x00000004 0x00000005 0x00000006 0x00000007",
                "0x00000008 0x00000009 0x0000000a 0x0000000b",
                "0x0000000c 0x0000000d 0x0000000e 0x0000000f",
            ],
        )

    def test_read_larger_than_one_packet(self):
        """A read of global memory that is larger than one memory-read packet
        returns all of it, as `memory read -p global` of a large buffer asks
        for. The same read with no address space does."""
        self.build()
        source = "memory.cu"
        _, _, thread, _ = self.run_to_cpu_line(source, "// big buffer filled")
        address = thread.frame[0].FindVariable("d_big").GetValueAsAddress()
        # lldb-server sometimes learns of the fault only after the program has
        # run on to its exit, which ends the GPU process too.
        exit_breakpoint = self.cpu_target.BreakpointCreateByLocation(
            lldb.SBFileSpec(source), line_number(source, "// breakpoint before exit")
        )
        self.assertGreater(
            exit_breakpoint.GetNumLocations(), 0, "the breakpoint before the exit"
        )
        # Memory is read at a GPU stop: the kernel's fault.
        self.continue_cpu_and_wait_for_gpu_to_stop()

        # What the program wrote, see memory.cu.
        size = 1 << 20
        expected = bytes(i % 251 for i in range(size))

        error = lldb.SBError()
        data = self.gpu_process.ReadMemory(address, size, error)
        self.assertSuccess(error, f"reading {size} bytes with no address space")
        self.assertEqual(len(data), size, "bytes read with no address space")
        self.assertTrue(data == expected, "the bytes read with no address space")

        data = self.read_gpu_memory(address, "global", size)
        self.assertEqual(len(data), size, "bytes read from the global space")
        self.assertTrue(data == expected, "the bytes read from the global space")
