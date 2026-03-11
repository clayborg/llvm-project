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

    def continue_to_gpu_target_creation(self):
        """Continues the process until the GPU target is created."""
        # Need to run these commands asynchronously to be able to switch targets.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue the CPU process until stop.
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There should be two targets")

    def run_to_gpu_breakpoint(
        self, source: str, gpu_bkpt_pattern: str
    ) -> List[lldb.SBThread]:
        """Launch the process, wait for the GPU target to be created, set a
        GPU breakpoint, and continue until it is hit. No CPU breakpoint is
        needed because the GPU plugin automatically stops the CPU process
        when GPU modules are loaded (auto_resume_native=false)."""
        target = lldbutil.run_to_breakpoint_make_target(self)

        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertFalse(error.Fail(), "Process launch failed: %s" % error.GetCString())

        # The GPU target should be created after launch. The GPU plugin
        # stops the CPU when GPU modules are loaded (auto_resume_native=false).
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

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
