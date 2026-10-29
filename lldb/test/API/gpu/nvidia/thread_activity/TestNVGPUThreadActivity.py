import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUThreadActivity(NVGPUTestCaseBase):
    def run_to_divergent_store(self):
        """Stop the GPU at the store that only the odd threads of the warp
        execute."""
        source = "activity.cu"
        self.build()
        # The program loads this from its working directory.
        lldbutil.install_to_target(self, self.getBuildArtifact("loaded.fatbin"))
        self.run_to_cpu_line(source, "// before kernel launch")
        self.select_gpu()
        breakpoint = self.gpu_target.BreakpointCreateByLocation(
            lldb.SBFileSpec(source), line_number(source, "// divergent store")
        )
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(breakpoint)

    def test_divergent_branch(self):
        """Report the threads that skip a branch as inactive."""
        self.run_to_divergent_store()
        activity = {
            self.thread_coords(thread)[1][0]: thread.IsActive()
            for thread in self.gpu_process.threads
        }
        self.assertEqual(activity, {x: x % 2 == 1 for x in range(32)})

    def test_module_load(self):
        """Report every thread as active at a module load, which does not read
        the GPU's thread state, instead of its activity at the previous stop."""
        self.run_to_divergent_store()
        self.assertIn(
            False,
            [thread.IsActive() for thread in self.gpu_process.threads],
            "no thread is inactive at the divergent store, so none can go stale",
        )
        # Set only now: the setting also applies to the CPU, which would stop
        # at each library it loads before this point.
        self.runCmd("settings set target.process.stop-on-sharedlibrary-events 1")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings set target.process.stop-on-sharedlibrary-events 0"
            )
        )
        # The module load after the kernel lists the same threads again.
        self.continue_gpu_until_stop_reason(lldb.eStopReasonDynamicLoader)
        self.assertEqual(
            [thread.IsActive() for thread in self.gpu_process.threads], [True] * 32
        )
