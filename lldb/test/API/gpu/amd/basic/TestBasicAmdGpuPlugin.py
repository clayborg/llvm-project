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

    def test_frame_info_shows_source(self):
        """Test that 'frame info' shows source context on GPU target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        # Switch to the GPU target and verify 'frame info' shows source context.
        self.select_gpu()
        self.expect(
            "frame info",
            substrs=["helloworld_kernel", "hello_world.hip", "->"],
        )

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
        # loaded from memory (driver/debugger lib code). The in-memory module name is the
        # same as the cpu process id.
        gpu_modules = self.gpu_target.modules
        self.assertEqual(2, len(gpu_modules), "GPU should have two modules")

        # Check for the expected modules but allow them to be in any order.
        exe = "a.out"
        pid = str(self.cpu_process.GetProcessID())
        module_files = [module.file.basename for module in gpu_modules]
        self.assertIn(exe, module_files)
        self.assertIn(pid, module_files)
