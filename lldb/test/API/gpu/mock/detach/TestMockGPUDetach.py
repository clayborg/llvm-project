"""
Test detaching from GPU targets.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase


class TestMockGPUDetach(GpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        self.source_spec = lldb.SBFileSpec("main.c")

    def run_to_the_gpu_target(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", self.source_spec
        )
        gpu_process = self.gpu_process
        self.assertTrue(gpu_process.IsValid(), "the plug-in connected its GPU target")
        return process, gpu_process

    def test_detach_the_gpu_target(self):
        """Detaching the GPU target leaves lldb-server running for the process,
        although the plug-in's GDB server shares lldb-server's main loop. The
        process then hits the plug-in's breakpoint on gpu_shlib_load, whose
        GPUActions lldb skips, since the GPU process is gone."""
        process, gpu_process = self.run_to_the_gpu_target()
        self.assertSuccess(gpu_process.Detach())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitError
        )
        threads = lldbutil.continue_to_source_breakpoint(
            self, process, "// break again", self.source_spec
        )
        self.assertTrue(threads, "the process stopped at the breakpoint")
        errors = []
        event = lldb.SBEvent()
        while listener.GetNextEventForBroadcaster(broadcaster, event):
            diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
            errors.append(diagnostic.GetValueForKey("message").GetStringValue(4096))
        self.assertEqual(errors, [], "lldb reported errors")
