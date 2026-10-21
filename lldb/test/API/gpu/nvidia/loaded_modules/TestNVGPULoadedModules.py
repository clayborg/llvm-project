"""
Test breakpoints in modules that the program loads at run time: a fatbin, and
PTX that the driver compiles.
"""

from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPULoadedModules(NVGPUTestCaseBase):
    def test_stop_in_loaded_fatbin_and_ptx(self):
        """Test stops in kernels from a fatbin and from PTX that the driver
        compiles, both loaded with cuModuleLoad."""
        self.build()
        # The program loads these from its working directory.
        for name in ["fatbin_kernel.fatbin", "ptx_kernel.ptx"]:
            lldbutil.install_to_target(self, self.getBuildArtifact(name))
        self.run_to_cpu_line("main.cu", "// breakpoint before loading")

        fatbin_breakpoint = self.gpu_target.BreakpointCreateByName("fatbin_kernel")
        ptx_breakpoint = self.gpu_target.BreakpointCreateByName("ptx_kernel")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        thread = self.assert_gpu_stop_at_breakpoint(fatbin_breakpoint)[0]
        self.assert_frame_at(
            thread.frame[0],
            "fatbin_kernel",
            "fatbin_kernel.cu",
            "// breakpoint in fatbin_kernel",
        )

        thread = self.continue_gpu_until_breakpoint(ptx_breakpoint)[0]
        self.assert_frame_at(
            thread.frame[0],
            "ptx_kernel",
            "ptx_kernel.cu",
            "// breakpoint in ptx_kernel",
        )
