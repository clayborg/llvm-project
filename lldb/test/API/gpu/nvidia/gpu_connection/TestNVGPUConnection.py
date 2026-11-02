import os
import signal
from unittest import mock

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import PROCESS_IS_VALID
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUConnection(NVGPUTestCaseBase):
    """lldb connects to the GPU server that lldb-server starts when the program
    initializes CUDA."""

    def test_lldb_cannot_reach_the_gpu_server(self):
        """When lldb cannot connect to the GPU server, lldb-server goes on
        debugging the CPU process, which lldb can still interrupt and detach
        from."""
        self.build()

        # lldb-server gives lldb this address for the GPU server, which listens
        # on 127.0.0.1 only, so lldb's connection is refused. lldb-server
        # inherits this process's environment.
        patcher = mock.patch.dict(
            os.environ, {"NVGPU_DEBUGGER_REMOTE_HOST": "127.0.0.2"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        target = self.createTestTarget()
        broadcaster = self.dbg.GetBroadcaster()
        error_listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitError
        )

        self.setAsync(True)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        pid = process.GetProcessID()

        def kill_program():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # The test detaches from the program, which then keeps running.
        self.addTearDownHook(kill_program)

        event = lldbutil.fetch_next_event(self, error_listener, broadcaster, timeout=60)
        diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
        self.assertIn(
            "Failed to connect to 127.0.0.2",
            diagnostic.GetValueForKey("message").GetStringValue(1024),
        )
        self.assertEqual(
            self.gpu_process.GetProcessID(),
            lldb.LLDB_INVALID_PROCESS_ID,
            "lldb connected to the GPU server",
        )

        # lldb-server still debugs the CPU process, so it stops when
        # interrupted, and lldb can detach from it.
        self.assertEqual(
            self.pull_state(process),
            lldb.eStateRunning,
            "the CPU process should be running",
        )
        self.assertSuccess(process.Stop())
        self.wait_for_stop(process)
        self.assertSuccess(process.Detach())
        # TODO: Check that the program gets past CUDA initialization once the
        # session ends. The driver keeps it waiting there until lldb-server is
        # gone, and then usually lets it go on, but crashes it about one time
        # in four.
