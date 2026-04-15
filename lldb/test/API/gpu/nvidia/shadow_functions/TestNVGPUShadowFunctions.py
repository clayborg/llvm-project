import json
import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUShadowFunctions(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def assertBreakpointLocationsDisabled(self, breakpoint):
        num_locations = breakpoint.GetNumLocations()
        self.assertGreater(
            num_locations,
            0)
        for i in range(num_locations):
            loc = breakpoint.GetLocationAtIndex(i)
            self.assertFalse(loc.IsEnabled())

    def test_shadow_function_breakpoint_disabled(self):
        """Test that breakpoints set on kernel function names have their CPU-side
        shadow wrapper locations disabled after module load."""
        self.killCPUOnTeardown()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd(f"file {exe}")

        source = "shadow_functions.cu"
        gpu_bp_line: int = line_number(source, "// gpu breakpoint")
        # NOTE: this CPU breakpoint isn't really important to this test.
        # However, it seems that in the test runner context, we need to set some
        # CPU breakpoint before the GPU target launch otherwise the test will
        # hang during "r".
        cpu_bp_line: int = line_number(source, "// cpu breakpoint")

        # These breakpoints should all not be hit. They point to the GPU
        # function in different ways.
        cpu_target_kernel_bp_by_name = self.cpu_target.BreakpointCreateByName(
            "my_kernel"
        )
        self.assertTrue(cpu_target_kernel_bp_by_name.IsValid())
        cpu_target_kernel_bp_by_file_line = (
            self.cpu_target.BreakpointCreateByLocation(
                "shadow_functions.cu", gpu_bp_line
            )
        )
        self.assertTrue(cpu_target_kernel_bp_by_file_line.IsValid())

        self.runCmd(f"b {cpu_bp_line}")
        self.runCmd(f"b {gpu_bp_line}")
        self.runCmd("r")

        self.continue_cpu_and_wait_for_gpu_to_stop()

        # Should be stopped on GPU breakpoint
        self.select_gpu()
        self.expect(
            "thread list",
            substrs=[f"at {source}:{gpu_bp_line}"],
        )

        # New breakpoints set after GPU target loading should also be disabled.
        cpu_target_kernel_bp_by_name_after_gpu_target = (
            self.cpu_target.BreakpointCreateByName("my_kernel")
        )

        # All locations of the kernel-name breakpoint should be disabled —
        # they all fell within the __device_stub_ shadow wrapper range.
        self.assertBreakpointLocationsDisabled(cpu_target_kernel_bp_by_name)
        self.assertBreakpointLocationsDisabled(cpu_target_kernel_bp_by_file_line)
        self.assertBreakpointLocationsDisabled(
            cpu_target_kernel_bp_by_name_after_gpu_target
        )

        stats_stream = lldb.SBStream()
        self.assertSuccess(self.cpu_target.GetStatistics().GetAsJSON(stats_stream))
        stats = json.loads(stats_stream.GetData())
        target_stats = stats["targets"][0]
        self.assertIn("shadowFunctionIdentificationTime", target_stats)
        self.assertGreater(target_stats["shadowFunctionIdentificationTime"], 0)
