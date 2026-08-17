import os
import time

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUAttach(NVGPUTestCaseBase):
    """Tests for attaching to an already-running CUDA application.

    Unlike the other NVGPU tests, which launch the inferior under the debugger
    and rely on the cuInit-style initialization breakpoint, this test starts the
    CUDA process independently (with a resident kernel already executing) and
    then attaches to it. The lldb-server NVGPU plugin must transparently detect
    the running CUDA driver, initiate the safe attach procedure, bring up the
    GPU target, and surface the in-flight kernel's threads in the thread list.
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
        exits first.

        On a host without a usable CUDA device the inferior's first CUDA call
        fails and the process exits almost immediately. Without this check the
        test would block for the full timeout waiting for a marker that can
        never appear. Detecting the early exit lets us skip promptly; the
        inferior's CUDA error is on its stderr (inherited by the test runner).
        """
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

    def test_attach_to_running_cuda_app(self):
        """Attach to a process whose CUDA kernel is already executing and verify
        that a GPU target comes up with the kernel's threads."""
        # Late attach needs real CUDA hardware; skip up front on a GPU-less host
        # instead of waiting out the readiness/attach timeouts.
        self.skip_if_no_cuda_device()

        self.build()
        exe = self.getBuildArtifact("a.out")

        # The inferior writes this marker file once a kernel is confirmed
        # resident on the GPU. Waiting for it (instead of a fixed sleep) makes
        # the test deterministic and guarantees we exercise true late attach to
        # a running kernel rather than racing the cuInit initialization path.
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        if os.path.exists(ready_marker):
            os.remove(ready_marker)

        # Start the CUDA process independently so a kernel is already resident
        # by the time we attach. spawnSubprocess registers a teardown hook that
        # kills the process.
        #
        # Force eager CUDA module loading in the inferior's environment. The
        # driver rejects late attach to a process using lazy module loading
        # ("Late attaching of a debugger that does not support CUDA lazy loading
        # is not supported"), which would leave the GPU target never coming up.
        popen = self.spawnSubprocess(
            exe, args=[ready_marker], extra_env=["CUDA_MODULE_LOADING=EAGER"]
        )

        # Wait for the inferior to signal that its kernel is resident before
        # attaching, failing fast if it exits early instead of blocking for the
        # full timeout.
        self._wait_for_marker_or_exit(popen, ready_marker)

        # Attach to the running process. This drives the late attach handshake
        # on the lldb-server NVGPU plugin.
        self.runCmd("process attach -p %d" % popen.pid)

        cpu_target = self.cpu_target
        self.assertTrue(cpu_target and cpu_target.IsValid(), "no CPU target after attach")

        # The safe attach procedure needs the application to keep running so the
        # driver can inject the debug engine at a safe point and call
        # CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED. Resume the CPU asynchronously;
        # a synchronous Continue() would block forever because the CPU host of
        # the resident kernel never stops on its own.
        self.setAsync(True)
        listener = self.dbg.GetListener()
        cpu_process = cpu_target.GetProcess()
        self.assertTrue(cpu_process and cpu_process.IsValid(), "no CPU process after attach")
        cpu_process.Continue()

        # Once the safe-attach handshake completes, the plugin reverse-connects
        # a GPU target. Target creation is observable just by enumerating the
        # debugger's targets, so poll for it.
        self.assertTrue(
            self._wait_for(lambda: self.gpu_target is not None),
            "GPU target was not created after attaching to the running CUDA app",
        )

        # The GPU process must now reach a stopped state with the in-flight
        # kernel's threads enumerated.
        #
        # In async mode a process's *public* state (what SBProcess.GetState()
        # returns) is only advanced when its state-changed event is pulled off
        # a listener: Process::SetPublicState runs from
        # ProcessEventData::DoOnRemoval as the event leaves the queue. The
        # earlier version of this test polled gpu_process.GetState() without
        # ever draining the debugger's listener, so the public state stayed
        # "running" and the wait timed out even though the GPU had already
        # stopped server-side. The interactive `continue` repro works precisely
        # because its event loop drains these events; mirror that here by
        # pumping the debugger listener until the GPU process reports stopped.
        def gpu_stopped():
            proc = self.gpu_process
            return (
                proc is not None
                and proc.IsValid()
                and proc.GetState() == lldb.eStateStopped
            )

        event = lldb.SBEvent()
        deadline = time.time() + 60
        while time.time() < deadline and not gpu_stopped():
            listener.WaitForEvent(1, event)

        self.assertTrue(gpu_stopped(), "GPU process did not stop after attach")

        self.select_gpu()
        self.assertGreater(
            len(self.gpu_process.threads),
            0,
            "expected the attached kernel's threads to appear in the thread list",
        )
