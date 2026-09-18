import os
import time

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUDetach(NVGPUTestCaseBase):
    """Tests for cleanly detaching from an attached CUDA application.

    Builds on the late-attach flow: attach to an already-running kernel, set a
    GPU breakpoint, then `process detach`. The plugin-owned detach cleanup must
    tear down the device breakpoints, reset the driver handshake flags, and let
    the CPU application keep running. A second attach to the same pid must then
    succeed, which exercises the breakpoint teardown and flag reset (a stale
    "debugger initialized"/IPC flag or leftover device breakpoint would make the
    re-attach fail or wedge).
    """

    NO_DEBUG_INFO_TESTCASE = True

    def _wait_for(self, predicate, timeout_seconds=60):
        """Poll predicate() until it is truthy or the timeout elapses."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.5)
        return predicate()

    def _wait_for_marker_or_exit(self, popen, marker_path, timeout_seconds=60):
        """Wait for the readiness marker, but bail out fast if the inferior
        exits first (e.g. on a host without a usable CUDA device)."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if os.path.exists(marker_path):
                return
            exit_code = popen.poll()
            if exit_code is not None:
                self.skipTest(
                    "CUDA inferior exited (code %s) before signalling a "
                    "resident kernel; the host likely lacks a usable CUDA "
                    "device or driver (see the inferior's stderr)" % exit_code
                )
            time.sleep(0.5)
        self.fail("CUDA inferior did not report a resident kernel before attach")

    def _attach_and_bring_up_gpu(self, exe, ready_marker):
        """Attach to the running inferior, drive the late-attach handshake, and
        return once the GPU target is stopped with the kernel's threads."""
        # `process attach` creates its target using the debugger's selected
        # platform, which is independent of the selected target. Once a GPU
        # target has been created the selected platform is PlatformNVGPU, whose
        # Attach() is intentionally unimplemented, so pin the host platform
        # before attaching. This matters on the re-attach below.
        self.runCmd("platform select host")
        self.runCmd("process attach -p %d" % self._popen.pid)

        cpu_target = self.cpu_target
        self.assertTrue(
            cpu_target and cpu_target.IsValid(), "no CPU target after attach"
        )

        # The safe attach procedure needs the application to keep running so the
        # driver can inject the debug engine at a safe point. Resume the CPU
        # asynchronously (manual-continue UX; the plugin does not auto-resume).
        self.setAsync(True)
        cpu_process = cpu_target.GetProcess()
        self.assertTrue(
            cpu_process and cpu_process.IsValid(), "no CPU process after attach"
        )
        cpu_process.Continue()

        self.assertTrue(
            self._wait_for(lambda: self.gpu_target is not None),
            "GPU target was not created after attaching to the running CUDA app",
        )
        self.assertTrue(
            self.wait_for_gpu_process_stopped(),
            "GPU process did not stop after attach",
        )
        return cpu_process

    def _start_resident_kernel(self, exe, ready_marker):
        """Start the CUDA inferior independently and wait until a kernel is
        resident, so the attach below is a true late attach."""
        if os.path.exists(ready_marker):
            os.remove(ready_marker)

        # Start the CUDA process independently so a kernel is already resident by
        # the time we attach. spawnSubprocess registers a teardown hook that
        # kills the process.
        #
        # LATE_ATTACH_INFERIOR_ENV disables the driver's lazy loading features,
        # without which the driver would refuse the attach and the GPU target
        # would never come up.
        self._popen = self.spawnSubprocess(
            exe, args=[ready_marker], extra_env=self.LATE_ATTACH_INFERIOR_ENV
        )
        self._wait_for_marker_or_exit(self._popen, ready_marker)

    def _detach_everything(self):
        """Detach the GPU target, then the CPU target."""
        # Detaching the GPU drives ProcessNVGPU::Detach ->
        # LLDBServerPluginNVGPU::DetachCleanup, which tears down the device
        # breakpoints, resets the driver handshake flags, and clears the attach
        # state without killing the CPU application.
        self.select_gpu()
        self.runCmd("detach")

        # Then the CPU target, so the inferior is left running with no debugger
        # attached at all.
        self.select_cpu()
        self.runCmd("detach")

        # Drop both detached targets. They linger in the debugger otherwise, and
        # the cpu_target/gpu_target helpers match on triple and would hand back
        # this session's dead GPU target after a later re-attach.
        targets = [
            self.dbg.GetTargetAtIndex(i) for i in range(self.dbg.GetNumTargets())
        ]
        for target in targets:
            self.dbg.DeleteTarget(target)

    def test_detach_keeps_cpu_running(self):
        """Attach, set a GPU breakpoint, detach, and verify the CPU app keeps
        running."""
        self.skip_if_no_cuda_device()

        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        self._start_resident_kernel(exe, ready_marker)

        self._attach_and_bring_up_gpu(exe, ready_marker)

        # Set a GPU breakpoint so the detach path has device breakpoints to tear
        # down. Any resident-kernel address works; use the kernel symbol.
        self.select_gpu()
        self.runCmd("breakpoint set -n spinKernel")

        self._detach_everything()

        # The CPU application must still be alive after detach.
        self.assertIsNone(
            self._popen.poll(),
            "the CPU application must keep running after detach",
        )

    def test_reattach_after_detach(self):
        """Attach, detach, then verify a re-attach to the same pid brings the
        GPU target back up."""
        self.skip_if_no_cuda_device()

        self.build()
        exe = self.getBuildArtifact("a.out")

        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        self._start_resident_kernel(exe, ready_marker)

        # First attach: bring up the GPU target.
        self._attach_and_bring_up_gpu(exe, ready_marker)

        # Set a GPU breakpoint so the detach path has device breakpoints to tear
        # down. Any resident-kernel address works; use the kernel symbol.
        self.select_gpu()
        self.runCmd("breakpoint set -n spinKernel")

        self._detach_everything()

        self.assertTrue(
            self._wait_for(lambda: self._popen.poll() is None, timeout_seconds=5),
            "the CPU application exited before re-attach",
        )
        self.assertTrue(
            self.wait_for_no_tracer(self._popen.pid),
            "lldb-server was still attached to the inferior after detach",
        )
        self._attach_and_bring_up_gpu(exe, ready_marker)
        self.select_gpu()
        self.assertGreater(
            len(self.gpu_process.threads),
            0,
            "expected the kernel's threads to appear after re-attach",
        )
