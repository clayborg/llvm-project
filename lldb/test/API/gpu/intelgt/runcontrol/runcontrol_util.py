"""
Shared helpers for IntelGT run-control API tests.

These tests launch a 1-D parallel_for kernel (parallel_1d.cpp) over 32
work-items, which the hardware splits across multiple EU hardware threads.
The helpers below launch the kernel, stop at a source breakpoint, and expose
the resulting GPU lane-threads grouped by EU thread.
"""

import lldb
import os
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase

NUM_WORK_ITEMS = 32
KERNEL_SOURCE = "parallel_1d.cpp"
KERNEL_EXE = "parallel_1d"


class RunControlTestCaseBase(IntelGtTestCaseBase):
    """Common launch / inspection helpers for run-control tests."""

    NUM_WORK_ITEMS = NUM_WORK_ITEMS

    def _create_target_with_breakpoint(self, pattern):
        """Build, create the target, and set a source-regex GPU breakpoint."""
        self.build()
        exe = self.getBuildArtifact(KERNEL_EXE)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex(
            pattern, lldb.SBFileSpec(KERNEL_SOURCE))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0,
                           "Breakpoint should have a location")
        return target, bp

    def _launch(self, target):
        """Launch the target and wait for the host process to stop."""
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{k}={v}" for k, v in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(),
                        f"Launch should succeed: {error.GetCString()}")
        self._wait_stopped(process)
        return process

    def _wait_stopped(self, process):
        """Block until *process* leaves the running state."""
        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(30, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()
        return state

    def _gpu_process(self):
        """Return the (valid) GPU process, asserting it exists."""
        self.assertEqual(self.dbg.GetNumTargets(), 2,
                         "GPU target should be created")
        gpu_process = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")
        return gpu_process

    def _lane_threads(self, gpu_process):
        """All GPU lane-threads (excluding the shadow thread, TID == 1)."""
        return [t for t in gpu_process.threads if t.GetThreadID() != 1]

    def launch_and_stop(self, pattern="inside-kernel"):
        """Build, launch, stop at *pattern*, and return (gpu_process, lanes)."""
        target, _ = self._create_target_with_breakpoint(pattern)
        self._launch(target)
        gpu_process = self._gpu_process()
        lanes = self._lane_threads(gpu_process)
        self.assertGreater(len(lanes), 0, "Should have at least one GPU lane")
        return gpu_process, lanes

    @staticmethod
    def lane_of(thread):
        """SIMD lane index = low 8 bits of the TID."""
        return thread.GetThreadID() & 0xFF

    @staticmethod
    def eu_of(thread):
        """EU-thread base = TID with the low 8 lane bits cleared."""
        return thread.GetThreadID() & ~0xFF

    @staticmethod
    def read_int(thread, name):
        var = thread.GetFrameAtIndex(0).FindVariable(name)
        if not var.IsValid():
            return None
        return var.GetValueAsSigned()
