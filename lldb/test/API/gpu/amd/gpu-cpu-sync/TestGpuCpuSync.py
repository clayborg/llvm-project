"""
Test GPU-CPU synchronization when no CPU breakpoint is set.

This test verifies that when a GPU target is created, the CPU process
automatically stops (via auto_resume_native=false) so the user can interact
with the GPU target. The user must manually resume both the GPU and CPU
processes.

Without the GPU-CPU sync changes, the CPU would auto-continue after the
internal GPU breakpoint, the kernel would execute and complete, and we
would never be able to set/hit a GPU breakpoint.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


class GpuCpuSyncTestCase(AmdGpuTestCaseBase):
    def test_gpu_breakpoint_without_cpu_breakpoint(self):
        """Test that we can hit a GPU breakpoint without setting any CPU
        breakpoint. The CPU process should automatically stop when the GPU
        target is created, allowing us to set a GPU breakpoint."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch. The GPU plugin
        # stops the CPU when GPU modules are loaded (auto_resume_native=false).
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be valid")
        self.assertTrue(self.gpu_process.IsValid(), "GPU process should be valid")

        # Switch to the GPU target and set a breakpoint in the kernel.
        source = "gpu_cpu_sync.hip"
        gpu_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT")

        # Resume the GPU and CPU processes asynchronously.
        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Resume the GPU process first.
        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        # Resume the CPU process so the kernel can execute.
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        # GPU breakpoint should get hit.
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateStopped]
        )

        # Verify the GPU breakpoint was hit.
        gpu_threads = lldbutil.get_threads_stopped_at_breakpoint_id(
            self.gpu_process, gpu_bkpt_id
        )
        self.assertGreater(
            len(gpu_threads), 0, "At least one GPU thread stopped at breakpoint"
        )
