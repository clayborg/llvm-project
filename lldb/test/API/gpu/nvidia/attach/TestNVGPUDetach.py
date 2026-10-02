import lldb
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDetach(NVGPUTestCaseBase):
    """Tests for cleanly detaching from an attached CUDA application.

    Builds on the late-attach flow: attach to an already-running kernel, set a
    GPU breakpoint, then detach. lldb removes the breakpoint before it detaches,
    and the GPU detach must let the driver clean up and reset its handshake
    flags. The GPU target can detach on its own, while detaching the CPU target
    detaches the GPU target first. Either way the application must keep
    running, and a second attach to the same pid must then succeed, which is
    what proves the cleanup was complete: a driver still believing a debugger
    is attached makes the re-attach fail or wedge.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def _wait_for_detached(self, process):
        """Pull the process's state changes until it reports that it detached."""
        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        while listener.WaitForEventForBroadcasterWithType(
            30,
            process.GetBroadcaster(),
            lldb.SBProcess.eBroadcastBitStateChanged,
            event,
        ):
            if lldb.SBProcess.GetStateFromEvent(event) == lldb.eStateDetached:
                return
        self.fail("process %d did not detach" % process.GetProcessID())

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

    def _check_detached_and_reattach(self):
        """Check that the application kept running with nothing attached, then
        attach to it again, which only brings the GPU up if the driver cleaned
        up."""
        self.assertIsNone(
            self._popen.poll(),
            "the CPU application must keep running after detach",
        )
        self.assertTrue(
            self.wait_for_no_tracer(self._popen.pid),
            "lldb-server was still attached to the inferior after detach",
        )

        # Drop both detached targets. They linger in the debugger otherwise, and
        # because cpu_target/gpu_target match on triple, a later re-attach would
        # be handed this session's dead GPU target.
        targets = [
            self.dbg.GetTargetAtIndex(i) for i in range(self.dbg.GetNumTargets())
        ]
        for target in targets:
            self.dbg.DeleteTarget(target)

        self.attach_to_running_cuda_app(self._popen.pid)
        self.find_thread_by_function("spinKernel")

    def test_detach_from_the_gpu_target(self):
        """Detaching the GPU target leaves the CPU target attached, and the
        application keeps running once that detaches too."""
        self._attach_and_set_gpu_breakpoint()
        cpu_process = self.cpu_process
        gpu_process = self.gpu_process

        self.select_gpu()
        self.runCmd("detach")
        self._wait_for_detached(gpu_process)
        # The GPU detach runs the CPU for the driver's cleanup and stops it again
        # before it returns, so lldb already sees it stopped.
        self.assertEqual(
            cpu_process.GetState(),
            lldb.eStateStopped,
            "the GPU detach did not leave the CPU attached and stopped",
        )

        self.select_cpu()
        self.runCmd("detach")
        self._wait_for_detached(cpu_process)
        self._check_detached_and_reattach()

    def test_detach_from_the_cpu_target(self):
        """Detaching the CPU target detaches the GPU target first, so the driver
        still cleans up."""
        self._attach_and_set_gpu_breakpoint()
        cpu_process = self.cpu_process
        gpu_process = self.gpu_process

        self.select_cpu()
        self.runCmd("detach")
        self._wait_for_detached(gpu_process)
        self._wait_for_detached(cpu_process)
        self._check_detached_and_reattach()
