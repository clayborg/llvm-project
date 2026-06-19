from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldb
import os
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from typing import List


class IntelGtTestCaseBase(GpuTestCaseBase):
    """
    Base class for all IntelGT GPU tests.

    Prerequisites:
        - oneAPI environment must be sourced before running tests:
          source ~/intel/oneapi/setvars.sh
        - ZET_ENABLE_PROGRAM_DEBUGGING=1 must be set
    """
    NO_DEBUG_INFO_TESTCASE = True

    def run_to_gpu_breakpoint(
        self, source: str, gpu_bkpt_pattern: str, cpu_bkpt_pattern: str, exe_name: str = "simple_kernel"
    ) -> List[lldb.SBThread]:
        """Run until GPU breakpoint hits.

        For IntelGT: Launch process first to trigger GPU target creation, then set GPU
        breakpoint and continue. The GPU target is created during zeModuleCreate but the
        kernel does not execute until we continue from the initial stop.

        Args:
            source: Source file name
            gpu_bkpt_pattern: Pattern to find GPU breakpoint location
            cpu_bkpt_pattern: Pattern to find CPU breakpoint location (unused for IntelGT)
            exe_name: Executable name (default: "simple_kernel")

        Returns:
            List of GPU threads stopped at the breakpoint
        """
        # Build and create target
        exe = self.getBuildArtifact(exe_name)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        # Pass environment to child process (inherits oneAPI from parent shell)
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{key}={value}" for key, value in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        # Launch the process - this will create both CPU and GPU targets
        process = target.Launch(launch_info, lldb.SBError())
        self.assertTrue(process.IsValid(), "Process should launch")

        # Wait for process to stop (IntelGT plugin stops process when GPU target is created)
        state = process.GetState()
        if state == lldb.eStateRunning:
            # Wait for it to stop
            import time
            max_wait = 30  # seconds
            start_time = time.time()
            while state == lldb.eStateRunning and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
                state = process.GetState()

        # Now GPU target should exist
        self.assertEqual(self.dbg.GetNumTargets(), 2, "Should have CPU and GPU targets")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be valid")

        # Set GPU breakpoint on the GPU target
        self.select_gpu()
        source_spec = lldb.SBFileSpec(source, False)
        gpu_bkpt = self.gpu_target.BreakpointCreateBySourceRegex(gpu_bkpt_pattern, source_spec)
        self.assertTrue(gpu_bkpt.IsValid(), "GPU breakpoint should be valid")

        # Continue the CPU process to let kernel execute
        self.select_cpu()
        error = self.cpu_process.Continue()
        if not error.Success():
            self.fail(f"Failed to continue: {error}")

        # Check if we hit the GPU breakpoint
        gpu_threads = lldbutil.get_threads_stopped_at_breakpoint(self.gpu_process, gpu_bkpt)
        return gpu_threads

    def set_gpu_source_breakpoint(self, source: str, gpu_bkpt_pattern: str) -> int:
        """Set a GPU breakpoint, return ID.

        Args:
            source: Source file name
            gpu_bkpt_pattern: Pattern to find breakpoint location

        Returns:
            Breakpoint ID
        """
        self.assertTrue(self.gpu_target.IsValid())
        self.select_gpu()

        line = line_number(source, gpu_bkpt_pattern)
        return lldbutil.run_break_set_by_file_and_line(
            self, source, line, num_expected_locations=-2, loc_exact=False
        )

    def continue_to_gpu_breakpoint(self, gpu_bkpt_id: int) -> List[lldb.SBThread]:
        """Continue execution until GPU breakpoint hits.

        Args:
            gpu_bkpt_id: Breakpoint ID to wait for

        Returns:
            List of GPU threads stopped at the breakpoint
        """
        # IntelGT plugin: Use synchronous continue with both processes.
        # The GPU process automatically resumes when the CPU process continues.
        self.select_cpu()
        self.setAsync(False)

        # Continue the CPU process - this implicitly continues GPU as well
        error = self.cpu_process.Continue()
        if not error.Success():
            self.fail(f"Failed to continue CPU process: {error}")

        # Check if GPU process hit the breakpoint
        # The GPU process should now be stopped at the breakpoint
        gpu_threads = lldbutil.get_threads_stopped_at_breakpoint_id(
            self.gpu_process, gpu_bkpt_id
        )
        if not gpu_threads:
            # GPU might not have hit the breakpoint - kernel may have completed
            # Return empty list to indicate no threads at breakpoint
            return []

        return gpu_threads

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
