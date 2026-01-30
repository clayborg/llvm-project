import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUBreakpoints(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_before_kernel_launch(self):
        """Test that we can set a breakpoint before the kernel launch."""
        self.killCPUOnTeardown()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd(f"file {exe}")
        source = "breakpoints.cu"
        cpu_bp_line: int = line_number(source, "// cpu breakpoint")
        gpu_bp_line: int = line_number(source, "// gpu breakpoint 1")
        gpu_bp_line_2: int = line_number(source, "// gpu breakpoint 2")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        self.runCmd(f"b {gpu_bp_line}")
        self.runCmd(f"b {gpu_bp_line_2}")
        self.runCmd(f"b {cpu_bp_line}")
        self.runCmd(f"b {exit_bp_line}")
        self.runCmd("r")

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.select_gpu()

        # All 16 threads should be at the first breakpoint, condensed into one line
        self.expect(
            "thread list",
            substrs=[
                "16 thread(s)",
                "blockIdx(x=[0...3] y=0 z=0) threadIdx(x=[0...3] y=0 z=0)",
                f"at {source}:{gpu_bp_line}",
            ],
        )

        self.dbg.SetAsync(False)
        self.select_gpu()
        self.gpu_process.Continue()
        # Only threadIdx.x == 0 threads hit the second breakpoint (4 threads, one per block)
        # The other 12 threads are at __syncthreads()
        self.expect(
            "thread list",
            substrs=[
                "blockIdx(x=0 y=0 z=0) threadIdx(x=0 y=0 z=0)",
                f"at {source}:{gpu_bp_line_2}",
            ],
        )
