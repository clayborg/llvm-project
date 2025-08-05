import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvidiagpu_testcase import NvidiaGpuTestCaseBase


class TestNVIDIAGPUErrorPC(NvidiaGpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_showing_error_pc(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "errorpc.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
        self.assertIn("CUDA Exception(6): Warp - Misaligned address at 0x", str(self.gpu_process.thread[0]))

        errorpc = self.gpu_process.thread[0].frame[0].FindRegister("errorpc").GetValueAsAddress()
        self.assertNotEqual(errorpc, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(errorpc, 0)
