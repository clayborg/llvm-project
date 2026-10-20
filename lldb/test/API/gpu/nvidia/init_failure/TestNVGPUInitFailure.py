import os
from unittest import mock

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import PROCESS_IS_VALID
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUInitFailure(NVGPUTestCaseBase):
    def test_no_visible_device(self):
        """With no CUDA device visible, the CUDA debugger API fails to
        initialize. lldb reports the error and adds no GPU target, and the
        program runs to its exit instead of waiting for a debugger that does
        not answer."""
        self.build()

        # lldb-server inherits this process's environment, and the debugger
        # API fails to initialize when it sees no device. The program gets the
        # same value through the target's environment.
        patcher = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

        target = self.createTestTarget()
        broadcaster = self.dbg.GetBroadcaster()
        error_listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitError
        )

        # Launch asynchronously, so that a program that waits for the debugger
        # fails the test at the timeout instead of hanging it.
        self.setAsync(True)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        lldbutil.expect_state_changes(
            self,
            self.dbg.GetListener(),
            process,
            [lldb.eStateRunning, lldb.eStateExited],
            timeout=60,
        )

        self.assertEqual(
            process.GetExitStatus(), 0, "the program did not report cudaErrorNoDevice"
        )
        self.assertEqual(self.dbg.GetNumTargets(), 1, "a GPU target was added")

        event = lldbutil.fetch_next_event(self, error_listener, broadcaster)
        diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
        self.assertIn(
            "CUDBG_ERROR_NO_DEVICE_AVAILABLE",
            diagnostic.GetValueForKey("message").GetStringValue(1024),
        )
