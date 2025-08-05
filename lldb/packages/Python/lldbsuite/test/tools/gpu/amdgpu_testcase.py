from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number


class AmdGpuTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python AMDGPU tests.
    """
    NO_DEBUG_INFO_TESTCASE = True

    def run_to_gpu_breakpoint(
        self, source, gpu_bkpt_pattern, cpu_bkpt_pattern
    ):
        """Run the test executable unit it hits the provided GPU breakpoint.
        The CPU breakpoint is used as a stopping point to switch to the GPU target
        and set the GPU breakpoint on that target.
        """
        # Set a breakpoint in the CPU source and run to it.
        source_spec = lldb.SBFileSpec(source, False)
        (cpu_target, cpu_process, cpu_thread, cpu_bkpt) = (
            lldbutil.run_to_source_breakpoint(self, cpu_bkpt_pattern, source_spec)
        )
        self.assertEqual(cpu_target, self.cpu_target)

        # Switch to the GPU target so we can set a breakpoint.
        self.select_gpu()

        # Set a breakpoint in the GPU source.
        # This might not yet resolve to a location so use -2 to not check
        # for the number of locations.
        line = line_number(source, gpu_bkpt_pattern)
        gpu_bkpt = lldbutil.run_break_set_by_file_and_line(
            self, source, line, num_expected_locations=-2, loc_exact=False
        )

        # Need to run these commands asynchronously to be able to switch targets.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue the GPU process.
        self.runCmd("c")
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateRunning])

        # Continue the CPU process.
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(self, listener, self.cpu_process, [lldb.eStateRunning])

        # GPU breakpoint should get hit.
        lldbutil.expect_state_changes(self, listener, self.gpu_process, [lldb.eStateStopped])
        return lldbutil.get_threads_stopped_at_breakpoint_id(self.gpu_process, gpu_bkpt)
