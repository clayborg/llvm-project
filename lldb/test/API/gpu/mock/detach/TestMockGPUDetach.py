"""
Test detaching from GPU targets.
"""

import os

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

    def test_detach_detaches_the_gpu_target(self):
        """Detaching the process detaches its GPU target first, which lldb
        would otherwise leave stopped once the process has gone."""
        process, gpu_process = self.run_to_the_gpu_target()
        self.assertSuccess(process.Detach())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(process.GetState(), lldb.eStateDetached)

    def test_detach_that_cannot_detach_the_gpu_target(self):
        """A detach fails before the process detaches when the GPU target does
        not detach, since the GPU target would be left stopped without a
        debugger. lldb still debugs both, the process keeps its breakpoints,
        and the detach can be tried again."""
        # lldb-server inherits this, and the mock GPU plug-in then refuses to
        # detach the first time.
        os.environ["LLDB_MOCK_GPU_FAIL_DETACH"] = "1"
        self.addTearDownHook(lambda: os.environ.pop("LLDB_MOCK_GPU_FAIL_DETACH"))
        process, gpu_process = self.run_to_the_gpu_target()
        breakpoint = process.GetTarget().BreakpointCreateBySourceRegex(
            "// break again", self.source_spec
        )
        error = process.Detach()
        self.assertTrue(error.Fail(), "the process detached")
        self.assertIn("could not detach from GPU process", error.GetCString())
        self.assertNotEqual(
            gpu_process.GetState(), lldb.eStateDetached, "the GPU target detached"
        )
        self.assertState(process.GetState(), lldb.eStateStopped)
        threads = lldbutil.continue_to_breakpoint(process, breakpoint)
        self.assertTrue(threads, "the process lost its breakpoint")
        self.assertSuccess(process.Detach())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(process.GetState(), lldb.eStateDetached)

    def test_detach_that_cannot_keep_the_process_stopped(self):
        """A detach that keeps the process stopped fails before it changes
        anything when lldb-server cannot do that, so the GPU target stays
        attached too."""
        process, gpu_process = self.run_to_the_gpu_target()
        self.assertFalse(process.Detach(True).Success(), "lldb-server kept it")
        self.assertSuccess(gpu_process.Detach(), "the GPU target was attached")
        self.assertSuccess(process.Detach())
