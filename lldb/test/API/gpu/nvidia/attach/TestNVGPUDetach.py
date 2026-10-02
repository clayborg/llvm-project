import lldb
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDetach(NVGPUTestCaseBase):
    """Tests for cleanly detaching from an attached CUDA application.

    Builds on the late-attach flow: attach to an already-running kernel, set a
    GPU breakpoint, then detach. lldb removes the breakpoint before it detaches,
    and the detach must let the driver clean up, reset the driver's handshake
    flags and leave the CPU application running. A second attach to
    the same pid must then succeed, which is what proves the cleanup was
    complete: a driver still believing a debugger is attached makes the
    re-attach fail or wedge.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def _detach_everything(self):
        """Detach the GPU target, then the CPU target."""
        self.select_gpu()
        self.runCmd("detach")

        # The GPU detach runs the CPU for the driver's cleanup and stops it again
        # before it returns, so lldb already sees it stopped.
        self.assertEqual(
            self.cpu_process.GetState(),
            lldb.eStateStopped,
            "the GPU detach left the CPU running",
        )
        self.select_cpu()
        self.runCmd("detach")

        # Drop both detached targets. They linger in the debugger otherwise, and
        # because cpu_target/gpu_target match on triple, a later re-attach would
        # be handed this session's dead GPU target.
        targets = [
            self.dbg.GetTargetAtIndex(i) for i in range(self.dbg.GetNumTargets())
        ]
        for target in targets:
            self.dbg.DeleteTarget(target)

    def _attach_and_set_gpu_breakpoint(self):
        """Attach, then set a GPU breakpoint, so the detach starts with one
        inserted on the device."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        self._popen = self.start_resident_kernel(exe, ready_marker)

        self.attach_to_running_cuda_app(self._popen.pid)
        self.select_gpu()
        self.runCmd("breakpoint set -n spinKernel")

    def test_detach_keeps_cpu_running(self):
        """Attach, set a GPU breakpoint, detach, and verify the CPU app keeps
        running."""
        self._attach_and_set_gpu_breakpoint()
        self._detach_everything()

        self.assertIsNone(
            self._popen.poll(),
            "the CPU application must keep running after detach",
        )

    def test_reattach_after_detach(self):
        """Attach, detach, then verify a re-attach to the same pid brings the
        GPU target back up."""
        self._attach_and_set_gpu_breakpoint()
        self._detach_everything()

        self.assertIsNone(
            self._popen.poll(), "the CPU application exited before re-attach"
        )
        self.assertTrue(
            self.wait_for_no_tracer(self._popen.pid),
            "lldb-server was still attached to the inferior after detach",
        )

        self.attach_to_running_cuda_app(self._popen.pid)
        self.find_thread_by_function("spinKernel")
