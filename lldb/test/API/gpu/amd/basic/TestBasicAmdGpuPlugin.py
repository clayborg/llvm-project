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

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch. The GPU plugin
        # stops the CPU when GPU modules are loaded (auto_resume_native=false).
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")

        # Make sure the GPU target has the default thread.
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
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

    def test_gpu_basic_step_over(self):
        """Test that a GPU thread can step over a source line."""
        self.build()

        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")
        self.step_over_gpu_thread(
            gpu_threads[0], line_number(source, "// GPU STEP OVER")
        )

    def test_gpu_resume_to_next_breakpoint(self):
        """Test that resuming GPU threads can hit a later breakpoint."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertFalse(error.Fail(), "Process launch failed: %s" % error.GetCString())
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        first_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT")
        next_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT AFTER")

        gpu_threads = self.continue_to_gpu_breakpoint(first_bkpt_id)
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        gpu_threads = self.continue_gpu_to_breakpoint(next_bkpt_id)
        self.assertNotEqual(None, gpu_threads, "GPU should hit next breakpoint")

    def test_gpu_breakpoint_delete(self):
        """Test that deleting a GPU breakpoint removes the trap opcode."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        source = "hello_world.hip"
        gpu_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT")
        self.assertTrue(
            self.gpu_target.BreakpointDelete(gpu_bkpt_id),
            "GPU breakpoint should be deleted",
        )

        self.setAsync(True)
        listener = self.dbg.GetListener()

        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateExited]
        )
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateExited]
        )

    def test_num_threads(self):
        """Test that we get the expected number of threads."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
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
            source, "// DIVERGENT BREAKPOINT"
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
        """Test that when no user breakpoints are set, the process stops at the
        GPU internal breakpoint for GPU target creation, and exits normally
        when continued."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch.
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should exist")

        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue GPU (non-blocking in async mode)
        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        # Continue CPU (non-blocking in async mode)
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        # Now wait for both to exit
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateExited]
        )
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateExited]
        )

    def test_image_list(self):
        """Test that we can load modules on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
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

    def test_wave_and_group_ids(self):
        """Test that a kernel smaller than a wave maps onto a single SIMD group."""
        self.build()

        source = "hello_world.hip"
        self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")

        # A single block smaller than a wave lands entirely in one wave, so
        # every thread shares one SIMD id. Lane ids happen to be unique across
        # the process here only because of that; TestMultiWaveAmdGpuPlugin
        # covers a kernel spread over several waves, where they are not.
        gpu_threads = self.gpu_process.threads
        simd_ids = {thread.GetSIMD() for thread in gpu_threads}
        self.assertEqual(
            len(simd_ids), 1, f"all lanes should share one wave, got {simd_ids}"
        )
        self.assertNotIn(
            lldb.LLDB_INVALID_SIMD_ID, simd_ids, "GPU threads should have a SIMD id"
        )

        # A CPU thread is neither a lane nor part of a SIMD group.
        self.stop_cpu_if_running(self.dbg.GetListener())
        cpu_thread = self.cpu_process.GetThreadAtIndex(0)
        self.assertTrue(cpu_thread.IsValid(), "CPU thread should be valid")
        self.assertEqual(cpu_thread.GetLaneID(), lldb.LLDB_INVALID_LANE_ID)
        self.assertEqual(cpu_thread.GetSIMD(), lldb.LLDB_INVALID_SIMD_ID)
