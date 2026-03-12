"""
Basic tests for the AMDGPU plugin.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SHADOW_THREAD_NAME = "AMD Native Shadow Thread"


class BasicAmdGpuTestCase(AmdGpuTestCaseBase):
    def test_gpu_target_created_on_demand(self):
        """Test that we create the gpu target automatically."""
        self.build()

        # There should be no targets before we run the program.
        self.assertEqual(self.dbg.GetNumTargets(), 0, "There are no targets")

        # Set a breakpoint in the CPU source and run to it.
        source_spec = lldb.SBFileSpec("hello_world.hip", False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// CPU BREAKPOINT - BEFORE LAUNCH", source_spec
        )
        self.assertEqual(self.cpu_target, cpu_target)

        # Make sure the GPU target was created and has the default thread.
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        self.assertEqual(
            gpu_thread.GetName(), SHADOW_THREAD_NAME, "GPU thread has the right name"
        )

        # The target should have the triple set correctly.
        self.assertIn("amdgcn-amd-amdhsa", self.gpu_target.GetTriple())



    def test_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

    def test_num_threads(self):
        """Test that we get the expected number of threads."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        # We launch one thread for each character in the output string.
        gpu_threads = self.gpu_process.threads
        num_expected_threads = len("Hello, world!")
        self.assertEqual(len(gpu_threads), num_expected_threads)

        # The shadow thread should not be listed once we have real threads
        for thread in gpu_threads:
            self.assertNotEqual(SHADOW_THREAD_NAME, thread.GetName())

        # All threads should be stopped at the breakpoint.
        self.assertEqual(len(gpu_threads_at_bp), num_expected_threads)

    def test_num_threads_divergent_breakpoint(self):
        """Test that we get the expected number of threads in a divergent breakpoint."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(
            source, "// DIVERGENT BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        # We launch one thread for each character in the output string.
        # So all threads should be present in the process.
        gpu_threads = self.gpu_process.threads
        total_num_threads = len("Hello, world!")
        self.assertEqual(len(gpu_threads), total_num_threads)

        # Since all the threads are in the same wave, they all share the same pc
        # and should be stopped at the same breakpoint. At some point, we need to
        # represent active/inactive threads in lldb, but that support does not yet
        # exist.
        self.assertEqual(len(gpu_threads_at_bp), total_num_threads)

    def test_no_unexpected_stop(self):
        """Test that we do not unexpectedly hit a stop in the debugger when
        No breakpoints are set."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

    def test_image_list(self):
        """Test that we can load modules on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        # There should two modules loaded for the gpu.
        # There should be one module loaded from the executable (the kernel) and one
        # loaded from memory (driver/debugger lib code).
        # File-backed modules keep their original file path (e.g. /path/to/a.out).
        # Memory-backed modules are named: amd_memory_kernel[start, end)
        gpu_modules = self.gpu_target.modules
        self.assertEqual(2, len(gpu_modules), "GPU should have two modules")

        # Check that one module contains "a.out" (file-backed, keeps original path)
        # and one starts with "amd_memory_kernel[" (memory-backed).
        module_names = [str(module.file) for module in gpu_modules]
        has_file_module = any("a.out" in name for name in module_names)
        has_memory_module = any("amd_memory_kernel[" in name for name in module_names)
        self.assertTrue(has_file_module,
                        f"Expected a file-backed module with 'a.out' in path, got: {module_names}")
        self.assertTrue(has_memory_module,
                        f"Expected a memory-backed module with 'amd_memory_kernel[' prefix, got: {module_names}")

        # Verify the "image list" command output shows the [offset-end) bracket
        # for the file-backed embedded GPU module (no space before the bracket).
        # Select GPU target so the command runs against it.
        self.dbg.SetSelectedTarget(self.gpu_target)
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("image list", result)
        output = result.GetOutput()
        self.assertTrue(result.Succeeded(), f"image list failed: {result.GetError()}")
        # File-backed module should show path[0x...-0x...) with no space before bracket.
        import re
        has_bracket_format = re.search(r'a\.out\[0x[0-9a-f]+-0x[0-9a-f]+\)', output)
        self.assertTrue(has_bracket_format,
                        f"Expected 'a.out[offset-end)' format in image list output, got:\n{output}")
        # Memory-backed module should show amd_memory_kernel[start, end)
        self.assertIn("amd_memory_kernel[", output,
                      f"Expected 'amd_memory_kernel[' in image list output, got:\n{output}")
