"""
Test that the CPU does not stop in the host code that nvcc generates to launch
a kernel, at a breakpoint that the GPU target resolves in the kernel itself.
"""

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "shadow_breakpoints.cu"
TWINS = "twins.cu"


class TestNVGPUShadowBreakpoints(NVGPUTestCaseBase):
    def location_states(self, breakpoint):
        """Map the function of each location of `breakpoint` in the program,
        rather than in a library, to whether the location is enabled."""
        states = {}
        for location in breakpoint.locations:
            address = location.GetAddress()
            if address.GetModule().GetFileSpec().GetFilename() != "a.out":
                continue
            states[address.GetFunction().GetName()] = location.IsEnabled()
        return states

    def assert_location_states(self, breakpoint, states):
        self.assertEqual(self.location_states(breakpoint), states, str(breakpoint))

    def assert_symbol_states(self, breakpoint, states):
        """Assert that `states` maps the mangled name of the symbol of each
        location of `breakpoint` in nodebug.out, which has no host debug
        information, to whether the location is enabled."""
        actual = {}
        for location in breakpoint.locations:
            address = location.GetAddress()
            if address.GetModule().GetFileSpec().GetFilename() != "nodebug.out":
                continue
            self.assertFalse(address.GetLineEntry().IsValid(), str(location))
            actual[address.GetSymbol().GetMangledName()] = location.IsEnabled()
        self.assertEqual(actual, states, str(breakpoint))

    def test_breakpoints_set_before_run(self):
        """Test that lldb disables the CPU locations in a kernel's launch code
        of breakpoints set before the run, once it creates the GPU target, and
        that the GPU stops in the kernel."""
        self.build()
        target = lldbutil.run_to_breakpoint_make_target(self)
        by_name = target.BreakpointCreateByName("kernel")
        by_line = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// kernel body")
        )
        # Without a GPU target, lldb leaves the CPU target's breakpoints alone.
        self.assert_location_states(
            by_name, {"kernel(int)": True, "kernel(float)": True}
        )
        self.assert_location_states(by_line, {"kernel(int)": True})

        # The first launch starts CUDA inside the launch stub. Both
        # breakpoints' locations in the launch code come after the call to the
        # stub, so the CPU must not stop at them.
        self.run_to_cpu_line(SOURCE, "// break after the first launch", target=target)
        self.assert_location_states(
            by_name, {"kernel(int)": False, "kernel(float)": True}
        )
        self.assert_location_states(by_line, {"kernel(int)": False})

        # lldb copied both breakpoints into the GPU target, where they resolve
        # in the kernel.
        self.wait_for_gpu_to_stop()
        thread = self.assert_gpu_stop_reason(lldb.eStopReasonBreakpoint)
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "kernel(int)")

    def test_breakpoints_set_after_cuda_starts(self):
        """Test that lldb disables the CPU locations in a kernel's launch code
        of breakpoints set once the GPU target exists, whichever way nvcc names
        that code, and leaves the locations in other host code alone."""
        self.build()
        target, process, _, _ = self.run_to_cpu_line(
            SOURCE, "// break after the first launch"
        )
        self.assertIsNotNone(self.gpu_target, "GPU target after the first launch")
        by_kernel_name = target.BreakpointCreateByName("kernel")
        in_both = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// both body")
        )
        self.assert_location_states(
            by_kernel_name, {"kernel(int)": False, "kernel(float)": True}
        )
        self.assert_location_states(
            target.BreakpointCreateByName("c_kernel"), {"::c_kernel(int)": False}
        )
        self.assert_location_states(
            target.BreakpointCreateByLocation(
                SOURCE, line_number(SOURCE, "// c_kernel body")
            ),
            {"::c_kernel(int)": False},
        )
        self.assert_location_states(
            target.BreakpointCreateByName("static_kernel"),
            {"::static_kernel(int)": False},
        )
        self.assert_location_states(
            target.BreakpointCreateByName("template_kernel"),
            {"ns::template_kernel<int>(int)": False},
        )
        self.assert_location_states(in_both, {"both(int)": True})

        # The CPU launches the other kernels without stopping, and then stops
        # in the host functions.
        for breakpoint, function in [
            (by_kernel_name, "kernel(float)"),
            (in_both, "both(int)"),
        ]:
            threads = lldbutil.continue_to_breakpoint(process, breakpoint)
            self.assertEqual(len(threads), 1, f"CPU stop at {breakpoint}")
            self.assertEqual(threads[0].GetFrameAtIndex(0).GetFunctionName(), function)

    def test_host_functions_with_the_names_of_kernels(self):
        """Test that lldb leaves alone the CPU locations in host functions with
        the names and parameters of kernels in another file, and that the CPU
        stops in them. When one of the two functions is static, the kernel's
        launch stub has the name that lldb looks up for the host function."""
        self.build()
        target, process, _, _ = self.run_to_cpu_line(
            SOURCE, "// break after the first launch"
        )
        static_host_twin = target.BreakpointCreateByName("static_host_twin")
        static_kernel_twin = target.BreakpointCreateByName("static_kernel_twin")
        self.assert_location_states(
            static_host_twin,
            {"static_host_twin(int)": False, "::static_host_twin(int)": True},
        )
        self.assert_location_states(
            static_kernel_twin,
            {"::static_kernel_twin(int)": False, "static_kernel_twin(int)": True},
        )

        # The CPU launches the kernels without stopping, and then stops in the
        # host functions.
        for breakpoint, function in [
            (static_host_twin, "::static_host_twin(int)"),
            (static_kernel_twin, "static_kernel_twin(int)"),
        ]:
            threads = lldbutil.continue_to_breakpoint(process, breakpoint)
            self.assertEqual(len(threads), 1, f"CPU stop at {breakpoint}")
            self.assertEqual(
                threads[0].GetFrameAtIndex(0).GetFunction().GetName(), function
            )

        # A breakpoint on a line of each host function stays enabled too.
        self.assert_location_states(
            target.BreakpointCreateByLocation(
                TWINS, line_number(TWINS, "// static_host_twin body")
            ),
            {"::static_host_twin(int)": True},
        )
        self.assert_location_states(
            target.BreakpointCreateByLocation(
                TWINS, line_number(TWINS, "// static_kernel_twin body")
            ),
            {"static_kernel_twin(int)": True},
        )

    def test_launch_code_without_debug_info(self):
        """Test that lldb finds the launch code of kernels in host code without
        debug information, and tells a static host function from the launch
        code of an external kernel with its name and parameters by the linkage
        that their names show. The CPU stops in the host function."""
        self.build(make_targets=["nodebug.out"])
        target = lldbutil.run_to_breakpoint_make_target(self, "nodebug.out")
        # Without line tables, stop in a function that the program calls once
        # it has launched the kernels.
        _, process, _, _ = lldbutil.run_to_breakpoint_do_run(
            self, target, target.BreakpointCreateByName("call_twins")
        )
        self.assertIsNotNone(self.gpu_target, "GPU target after the launches")

        static_host_twin = target.BreakpointCreateByName("static_host_twin")
        self.assert_symbol_states(
            static_host_twin,
            {"_Z16static_host_twini": False, "_ZL16static_host_twini": True},
        )
        self.assert_symbol_states(
            target.BreakpointCreateByName("kernel"),
            {"_Z6kerneli": False, "_Z6kernelf": True},
        )
        self.assert_symbol_states(
            target.BreakpointCreateByName("static_kernel"),
            {"_ZL13static_kerneli": False},
        )
        self.assert_symbol_states(
            target.BreakpointCreateByName("template_kernel"),
            {"_ZN2nsL15template_kernelIiEEvT_": False},
        )
        # Without debug information, nothing tells an external host function
        # from the launch code of a static kernel with its name and parameters:
        # nvcc gives the stubs of template kernels internal linkage even when
        # the names of their host functions do not show it. nvgpu-support.rst
        # documents this.
        self.assert_symbol_states(
            target.BreakpointCreateByName("static_kernel_twin"),
            {"_ZL18static_kernel_twini": False, "_Z18static_kernel_twini": False},
        )

        threads = lldbutil.continue_to_breakpoint(process, static_host_twin)
        self.assertEqual(len(threads), 1, f"CPU stop at {static_host_twin}")
        self.assertEqual(
            threads[0].GetFrameAtIndex(0).GetSymbol().GetMangledName(),
            "_ZL16static_host_twini",
        )

    def test_breakpoints_on_launch_code_stay_enabled(self):
        """Test that lldb leaves enabled the CPU locations of breakpoints that
        ask for a kernel's launch code: on an address, on a launch stub's name,
        or limited to the program's module, and a location the user enables
        again. The CPU stops at each."""
        self.build()
        target, process, _, _ = self.run_to_cpu_line(
            SOURCE, "// break after the first launch"
        )
        functions = target.FindFunctions("c_kernel")
        self.assertEqual(functions.GetSize(), 1, "CPU functions named c_kernel")
        by_address = target.BreakpointCreateBySBAddress(
            functions.GetContextAtIndex(0).GetFunction().GetStartAddress()
        )
        by_stub_name = target.BreakpointCreateByName("__device_stub__Z8c_kerneli")
        program = lldb.SBFileSpecList()
        program.Append(lldb.SBFileSpec("a.out"))
        in_program = target.BreakpointCreateByName(
            "static_kernel",
            lldb.eFunctionNameTypeAuto,
            program,
            lldb.SBFileSpecList(),
        )
        enabled_again = target.BreakpointCreateByName("template_kernel")
        self.assert_location_states(
            enabled_again, {"ns::template_kernel<int>(int)": False}
        )
        enabled_again.GetLocationAtIndex(0).SetEnabled(True)

        # The program launches c_kernel, static_kernel and template_kernel in
        # this order.
        for breakpoint in [by_address, by_stub_name, in_program, enabled_again]:
            self.assertTrue(
                breakpoint.GetLocationAtIndex(0).IsEnabled(), str(breakpoint)
            )
            threads = lldbutil.continue_to_breakpoint(process, breakpoint)
            self.assertEqual(len(threads), 1, f"CPU stop at {breakpoint}")

    def test_run_again_after_deleting_the_gpu_target(self):
        """Test that lldb handles each CPU location in a kernel's launch code
        once, even when the user deletes the GPU target before running the
        program again: a location the user enabled again stays enabled, and
        the location of a breakpoint set while there is no GPU target is
        disabled."""
        self.build()
        target, process, _, marker = self.run_to_cpu_line(
            SOURCE, "// break after the first launch"
        )
        enabled_again = target.BreakpointCreateByName("template_kernel")
        enabled_again.GetLocationAtIndex(0).SetEnabled(True)

        # Nothing else holds the GPU target, so deleting it destroys it.
        self.assertSuccess(process.Kill())
        self.assertTrue(self.dbg.DeleteTarget(self.gpu_target), "delete GPU target")
        set_without_gpu_target = target.BreakpointCreateByName("c_kernel")

        lldbutil.run_to_breakpoint_do_run(self, target, marker)
        self.assertIsNotNone(self.gpu_target, "GPU target of the second run")
        self.assert_location_states(
            enabled_again, {"ns::template_kernel<int>(int)": True}
        )
        self.assert_location_states(set_without_gpu_target, {"::c_kernel(int)": False})

    def test_optimized_launch_code(self):
        """Test that lldb finds a kernel's launch code when the host compiler
        optimizes it and inlines the launch stubs, and with them the stub
        names of static and template kernels."""
        self.build(dictionary={"CFLAGS_EXTRAS": "-Xcompiler -O3"})
        target, _, _, _ = self.run_to_cpu_line(
            SOURCE, "// break after the first launch"
        )
        self.assert_location_states(
            target.BreakpointCreateByName("c_kernel"), {"::c_kernel(int)": False}
        )
        self.assert_location_states(
            target.BreakpointCreateByName("static_kernel"),
            {"::static_kernel(int)": False},
        )
        self.assert_location_states(
            target.BreakpointCreateByName("template_kernel"),
            {"ns::template_kernel<int>(int)": False},
        )
        self.assert_location_states(
            target.BreakpointCreateByLocation(
                SOURCE, line_number(SOURCE, "// kernel body")
            ),
            {"kernel(int)": False},
        )
