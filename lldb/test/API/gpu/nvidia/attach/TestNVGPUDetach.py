from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDetach(NVGPUTestCaseBase):
    """Tests for cleanly detaching from an attached CUDA application.

    Builds on the late-attach flow: attach to an already-running kernel, set a
    GPU breakpoint, then detach. The plugin-owned detach cleanup must tear down
    the device breakpoints, let the driver clean up, reset its handshake flags
    and leave the CPU application running. A second attach to the same pid must
    then succeed, which is what proves the cleanup was complete: a leftover
    device breakpoint or a driver still believing a debugger is attached makes
    the re-attach fail or wedge.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def _detach_everything(self):
        """Detach the GPU target, then the CPU target."""
        self.select_gpu()
        self.runCmd("detach")

        # The CPU was resumed for the attach handshake and the server has
        # since halted it, but its public state only advances as that stop
        # event is drained. Detaching while LLDB still believes it is
        # running makes Process::Detach interrupt and then wait out
        # target.process.interrupt-timeout for a stop that already happened.
        self.assertTrue(
            self.wait_for_cpu_process_stopped(),
            "CPU process never reported stopped before detach",
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
        """Attach, then set a GPU breakpoint so the detach path has device
        breakpoints to tear down."""
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
        self.select_gpu()
        self.assertGreater(
            len(self.gpu_process.threads),
            0,
            "expected the kernel's threads to appear after re-attach",
        )
