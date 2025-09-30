import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvidiagpu_testcase import NvidiaGpuTestCaseBase


class TestNVIDIAGPUThreads(NvidiaGpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_thread_in_a_single_block(self):
        """Test that we can read all threads in a single block, and one of them has excepted."""
        self.killCPUOnTeardown()

        self.build()
        source = "threads.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)
        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.select_gpu()

        self.assertEqual(len(self.gpu_process.threads), 512)

        thread_with_exception = self.find_thread_by_name("threadIdx(x=5 y=0 z=0)")
        self.assertEqual(thread_with_exception.GetStopReason(), lldb.eStopReasonException)

        thread_without_exception = self.find_thread_by_name("threadIdx(x=511 y=0 z=0)")
        self.assertEqual(thread_without_exception.GetStopReason(), lldb.eStopReasonNone)
