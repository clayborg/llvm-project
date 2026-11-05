"""
Test attaching to a program whose libcuda predates the CUDA driver's attach
procedure.
"""

import os
from unittest import mock

from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUAttachUnsupportedDriver(NVGPUTestCaseBase):
    """lldb-server cannot ask such a driver to attach, so it logs why, and lldb
    attaches to the CPU process only. The program loads a stand-in for libcuda,
    so no GPU is needed."""

    NEEDS_GPU = False

    def test_attach_logs_that_the_gpu_cannot_be_attached_to(self):
        self.build()
        log_file = self.getBuildArtifact("lldb-server.log")
        # lldb passes these to the lldb-server that it starts.
        patcher = mock.patch.dict(
            os.environ,
            {
                "LLDB_DEBUGSERVER_LOG_FILE": log_file,
                "LLDB_SERVER_LOG_CHANNELS": "gdb-remote plugin",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        ready = self.getBuildArtifact("ready")
        program = self.spawnSubprocess(
            self.getBuildArtifact("a.out"),
            args=[ready],
            extra_env=["LD_LIBRARY_PATH=" + self.getBuildDir()],
        )
        self.assertTrue(
            self.wait_until(lambda: os.path.exists(ready)),
            "the program did not get ready",
        )

        # process attach makes its target with the selected platform, which an
        # earlier test in this process can leave as a GPU target's.
        self.runCmd("platform select host")
        self.runCmd(f"process attach -p {program.pid}")
        self.assertIsNone(self.gpu_target, "lldb added a GPU target")

        def logged_why():
            if not os.path.exists(log_file):
                return False
            with open(log_file) as log:
                return "cudbgInitiateDebuggerAttachProcedureFd was not found" in (
                    log.read()
                )

        self.assertTrue(
            self.wait_until(logged_why, 10),
            "lldb-server did not log why the GPU cannot be attached to",
        )
