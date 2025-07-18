import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import TestBase, line_number


class TestNVIDIAGPUErrorPC(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_showing_error_pc(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "errorpc.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        cpu = self.dbg.GetTargetAtIndex(0)
        gpu = self.dbg.GetTargetAtIndex(1)

        # We switch to async mode to wait for state changes in the GPU target while the CPU resumes.
        self.setAsync(True)
        listener = self.dbg.GetListener()
        cpu.process.Continue()
        lldbutil.expect_state_changes(self, listener, gpu.process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(gpu.process.state, lldb.eStateStopped)
        self.assertIn("CUDA Exception(6): Warp - Misaligned address at 0x", str(gpu.process.thread[0]))

        errorpc = gpu.process.thread[0].frame[0].FindRegister("errorpc").GetValueAsAddress()
        self.assertNotEqual(errorpc, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(errorpc, 0)
