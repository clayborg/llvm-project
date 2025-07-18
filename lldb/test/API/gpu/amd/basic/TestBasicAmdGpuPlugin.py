"""
Basic tests for the AMDGPU plugin.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class BasicAmdGpuTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

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
        self.assertEqual(self.dbg.GetTargetAtIndex(0), cpu_target)

        # Get the GPU target.
        # This target is created at the first CPU stop.
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")
        gpu_target = self.dbg.GetTargetAtIndex(1)
        gpu_thread = gpu_target.GetProcess().GetThreadAtIndex(0)
        self.assertEqual(gpu_thread.GetName(), "AMD Native Shadow Thread", "GPU thread has the right name")

    def test_gpu_breakpoint_hit(self):
        """Test that we create the gpu target automatically."""
        self.build()

        # Set a breakpoint in the CPU source and run to it.
        source = "hello_world.hip"
        source_spec = lldb.SBFileSpec(source, False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// CPU BREAKPOINT - BEFORE LAUNCH", source_spec
        )

        # Switch to the GPU target so we can set a breakpoint.
        gpu_target = self.dbg.GetTargetAtIndex(1)
        gpu_process = gpu_target.GetProcess()
        self.dbg.SetSelectedTarget(gpu_target)


        # Set a breakpoint in the GPU source.
        # This will not yet resolve to a location.
        line = line_number(source, "// GPU BREAKPOINT")
        gpu_bkpt = lldbutil.run_break_set_by_file_and_line(
            self, source, line, num_expected_locations=0, loc_exact=False
        )

        # Need to run these commands asynchronously to be able to switch targets.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue the GPU process.
        self.runCmd("c")
        lldbutil.expect_state_changes(self, listener, gpu_process, [lldb.eStateRunning])

        # Continue the CPU process.
        self.dbg.SetSelectedTarget(cpu_target)
        self.runCmd("c")
        lldbutil.expect_state_changes(self, listener, cpu_process, [lldb.eStateRunning])

        # GPU breakpoint should get hit.
        lldbutil.expect_state_changes(self, listener, gpu_process, [lldb.eStateStopped])
        threads = lldbutil.get_threads_stopped_at_breakpoint_id(gpu_process, gpu_bkpt)
        self.assertNotEqual(None, threads, "GPU thread should be stopped at breakpoint")
