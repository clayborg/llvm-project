import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUThreads(NVGPUTestCaseBase):
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

    def test_thread_list_aggregation(self):
        """Test that `thread list` aggregates GPU threads by source line and that
        `-v` falls back to the legacy per-PC layout."""
        self.killCPUOnTeardown()

        self.build()
        source = "threads.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)
        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.select_gpu()

        # Default (aggregated) mode: groups are summarized by source line, the
        # raw "pc=0x" prefix is suppressed in favor of "module`function at
        # file:line", and large thread counts are visible in the summary line.
        self.expect(
            "thread list",
            substrs=[
                "thread(s):",
                f"at {source}:",
            ],
            matching=True,
        )
        self.expect("thread list", substrs=["pc=0x"], matching=False)

        # Verbose mode opts out of GPU-specific aggregation and renders one
        # row per thread, like `thread list` on a CPU target. Each row uses
        # the standard `thread #N: tid = ...` prefix and embeds the CUDA
        # coordinates in the thread name.
        self.expect(
            "thread list -v",
            substrs=["thread #", "tid = 0x", "name = 'blockIdx"],
            matching=True,
        )
        self.expect("thread list -v", substrs=["thread(s):"], matching=False)

        # --stop-reason exception (default rendering): filters down to threads
        # whose stop reason is a CUDA exception. The thread that dereferenced
        # 0x03 must still be there; threads from non-faulting warps are folded
        # into a trailing summary row.
        self.expect(
            "thread list -s exception",
            substrs=["thread(s):", "hidden by --stop-reason filter"],
            matching=True,
        )
        self.expect(
            "thread list -s exception",
            substrs=["threadIdx(x=511 y=0 z=0), stop reason"],
            matching=False,
        )

        # --stop-reason SIGTRAP: substring match against the description, used
        # here to limit the output to the SIGTRAP-stopped threads in the
        # warp.
        self.expect(
            "thread list -s SIGTRAP",
            substrs=["thread(s):"],
            matching=True,
        )

        # Combining -v and --stop-reason: per-thread rendering, filtered to
        # threads matching the filter expression.
        self.expect(
            "thread list -v -s exception",
            substrs=["thread #"],
            matching=True,
        )
