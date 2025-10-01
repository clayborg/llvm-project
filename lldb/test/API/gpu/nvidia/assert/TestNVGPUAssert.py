import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

class TestNVGPUAssert(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
        self.assertIn("CUDA Exception(12): Warp - Assert", str(self.gpu_process.thread[0]))

        frame = self.gpu_process.thread[0].frame[0]

        # We don't expect to see an errorpc set
        self.assertNotIn("CUDA Exception(12): Warp - Assert at 0x", str(self.gpu_process.thread[0]))
        errorpc = frame.FindRegister("errorpc").GetValueAsAddress()
        self.assertEqual(errorpc, lldb.LLDB_INVALID_ADDRESS)

        # As our kernel crashes in the prologue of assert, the RA register should be the same as the PC.
        self.assertEqual(
            frame.FindRegister("RA").GetValueAsAddress(),
            frame.FindRegister("PC").GetValueAsAddress(),
        )
