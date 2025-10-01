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
        gpu_bp_line: int = line_number(source, "// gpu breakpoint")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        self.runCmd(f"b {gpu_bp_line}")
        self.runCmd(f"b {cpu_bp_line}")
        self.runCmd(f"b {exit_bp_line}")
        self.runCmd("r")

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.select_gpu()

        self.expect(
            "thread list",
            substrs=[
                f"at {source}:{gpu_bp_line}, name = 'blockIdx(x=0 y=0 z=0) threadIdx(x=0 y=0 z=0)', stop reason = breakpoint",
                f"at {source}:{gpu_bp_line}, name = 'blockIdx(x=1 y=0 z=0) threadIdx(x=0 y=0 z=0)', stop reason = breakpoint",
                f"at {source}:{gpu_bp_line}, name = 'blockIdx(x=2 y=0 z=0) threadIdx(x=0 y=0 z=0)', stop reason = breakpoint",
                f"at {source}:{gpu_bp_line}, name = 'blockIdx(x=3 y=0 z=0) threadIdx(x=0 y=0 z=0)', stop reason = breakpoint",
            ],
        )
