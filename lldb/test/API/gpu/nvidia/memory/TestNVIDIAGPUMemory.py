import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvidiagpu_testcase import NvidiaGpuTestCaseBase


class TestNVIDIAGPUMemory(NvidiaGpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "memory.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

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
