import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUShadowFunctions(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_shadow_function_breakpoint_disabled(self):
        """Test that breakpoints set on kernel function names have their CPU-side
        shadow wrapper locations disabled after module load."""
        self.killCPUOnTeardown()

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd(f"file {exe}")

        source = "shadow_functions.cu"
        cpu_bp_line: int = line_number(source, "// cpu breakpoint")
        gpu_bp_line: int = line_number(source, "// gpu breakpoint")

        # Set a breakpoint on the GPU kernel by name. On the CPU target, this
        # resolves to the host-side shadow wrapper (the __device_stub_ wrapper).
        kernel_bp = self.cpu_target.BreakpointCreateByName("my_kernel")
        self.assertTrue(kernel_bp.IsValid())

        # Set a GPU source line breakpoint and a CPU breakpoint before launch.
        self.runCmd(f"b {gpu_bp_line}")
        self.runCmd(f"b {cpu_bp_line}")
        self.runCmd("r")

        # Continue the CPU past the kernel launch and wait for the GPU to stop.
        # This triggers RecordLoadedModule → IdentifyShadowFunctions →
        # DisableShadowFunctionBreakpoints on the CPU target.
        self.continue_cpu_and_wait_for_gpu_to_stop()

        # All locations of the kernel-name breakpoint should be disabled —
        # they all fell within the __device_stub_ shadow wrapper range.
        num_locations = kernel_bp.GetNumLocations()
        self.assertGreater(num_locations, 0,
            "kernel breakpoint should have resolved to at least one location")
        for i in range(num_locations):
            loc = kernel_bp.GetLocationAtIndex(i)
            self.assertFalse(
                loc.IsEnabled(),
                f"shadow function breakpoint location {i} should be disabled"
            )

        # The GPU should have stopped at the actual kernel body.
        self.select_gpu()
        self.expect(
            "thread list",
            substrs=[f"at {source}:{gpu_bp_line}"],
        )
