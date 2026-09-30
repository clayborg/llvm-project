import os

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUAttachUnsupportedDriver(NVGPUTestCaseBase):
    """Attach to a process whose libcuda lacks the symbols of the driver's safe
    attach procedure. CUDA may already be running in such a process, so the
    user has to be told that its GPU cannot be attached to. No GPU is needed:
    the inferior loads a stand-in libcuda."""

    NO_DEBUG_INFO_TESTCASE = True

    def _attach_and_collect_warnings(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("ready.marker")
        if os.path.exists(ready_marker):
            os.remove(ready_marker)
        popen = self.spawnSubprocess(
            exe,
            args=[ready_marker],
            extra_env=["LD_LIBRARY_PATH=" + self.getBuildDir()],
        )
        self.assertTrue(
            self.wait_for(lambda: os.path.exists(ready_marker)),
            "the inferior never reported that it was ready",
        )

        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitWarning
        )
        self.runCmd("process attach -p %d" % popen.pid)
        self.assertIsNone(self.gpu_target, "a GPU target came up")

        warnings = []
        event = lldb.SBEvent()
        while listener.WaitForEventForBroadcaster(1, broadcaster, event):
            diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
            warnings.append(diagnostic.GetValueForKey("message").GetStringValue(1024))
        return warnings

    def _assert_warned(self, warnings):
        self.assertTrue(
            any(
                "cannot attach to the GPU" in warning
                and "cudbgInitiateDebuggerAttachProcedureFd" in warning
                for warning in warnings
            ),
            "no warning that the GPU cannot be attached to, got %s" % warnings,
        )

    def test_attach_warns_that_the_gpu_cannot_be_attached_to(self):
        self._assert_warned(self._attach_and_collect_warnings())

    def test_attach_warns_without_waiting_for_the_gpu(self):
        """The warning does not depend on lldb waiting for the GPU attach."""
        self.runCmd("settings set plugin.process.gdb-remote.wait-for-gpu-attach false")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings clear plugin.process.gdb-remote.wait-for-gpu-attach"
            )
        )
        self._assert_warned(self._attach_and_collect_warnings())
