from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from typing import List


class AmdGpuTestCaseBase(GpuTestCaseBase):
    """
    Class that should be used by all python AMDGPU tests.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def run_to_gpu_breakpoint(
        self, source: str, gpu_bkpt_pattern: str, cpu_bkpt_pattern: str
    ) -> List[lldb.SBThread]:
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

        gpu_bkpt_id = self.set_gpu_source_breakpoint(source, gpu_bkpt_pattern)

        return self.continue_to_gpu_breakpoint(gpu_bkpt_id)

    def set_gpu_source_breakpoint(self, source: str, gpu_bkpt_pattern: str) -> int:
        """Set a breakpoint on the gpu target. Returns the breakpoint id."""
        # Switch to the GPU target so we can set a breakpoint.
        self.assertTrue(self.gpu_target.IsValid())
        self.select_gpu()

        # Set a breakpoint in the GPU source.
        # This might not yet resolve to a location so use -2 to not check
        # for the number of locations.
        line = line_number(source, gpu_bkpt_pattern)
        return lldbutil.run_break_set_by_file_and_line(
            self, source, line, num_expected_locations=-2, loc_exact=False
        )

    def continue_to_gpu_breakpoint(self, gpu_bkpt_id: int) -> List[lldb.SBThread]:
        """Continues execution on the cpu and gpu until we hit the gpu breakpoint"""
        # Need to run these commands asynchronously to be able to switch targets.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue the GPU process.
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        # Continue the CPU process.
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        # GPU breakpoint should get hit.
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateStopped]
        )
        return lldbutil.get_threads_stopped_at_breakpoint_id(
            self.gpu_process, gpu_bkpt_id
        )

    def continue_to_gpu_source_breakpoint(
        self, source: str, gpu_bkpt_pattern: str
    ) -> List[lldb.SBThread]:
        """
        Sets a gpu breakpoint set by source regex gpu_bkpt_pattern, continues the process, and deletes the breakpoint again.
        Otherwise the same as `continue_to_gpu_breakpoint`.
        Inspired by lldbutil.continue_to_source_breakpoint.
        """
        gpu_bkpt_id = self.set_gpu_source_breakpoint(source, gpu_bkpt_pattern)
        gpu_threads = self.continue_to_gpu_breakpoint(gpu_bkpt_id)
        self.gpu_target.BreakpointDelete(gpu_bkpt_id)

        return gpu_threads
