import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import TestBase, line_number

class TestNVIDIAGPUAssert(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "assert.cu"
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
        self.assertIn("NVIDIA GPU Thread Stopped by Exception", str(gpu.process.thread[0]))

        # Now let's test that the disass can print at least one entry
        self.expect("disassemble", patterns=[".*cuda_elf.*cubin.* <.*>: .*"])
