"""
Basic tests for the AMDGPU plugin.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


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
        self.assertEqual(gpu_thread.GetName(), "AMD Native Shadow Thread", "GPU thread has the right name")

    def test_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(
            source, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

    def test_no_unexpected_stop(self):
        """Test that we do not unexpectedly hit a stop in the debugger when
        No breakpoints are set."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
