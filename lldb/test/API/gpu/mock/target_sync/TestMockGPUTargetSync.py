"""
Test that a GPU plug-in halting the native process stops it for
eStopReasonTargetSync.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase


class TestMockGPUTargetSync(GpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_halt_during_a_step_over(self):
        """When the plug-in halts the process while a line is stepped over,
        the process stops there for eStopReasonTargetSync with the plug-in's
        target, and continuing finishes the step."""
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// step over this line", lldb.SBFileSpec("main.c")
        )
        gpu_target = self.gpu_target
        self.assertTrue(gpu_target, "the plug-in connected its GPU target")

        thread.StepOver()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonTargetSync)
        self.assertEqual(
            thread.GetStopDescription(256),
            "target sync with target #%d" % self.dbg.GetIndexOfTarget(gpu_target),
        )
        functions = [frame.GetFunctionName() for frame in thread.frames]
        self.assertIn("wait_for_the_halt", functions, "the step finished")

        # The stop interrupts the step without ending it, as a breakpoint does.
        error = lldb.SBError()
        wait = target.FindFirstGlobalVariable("g_wait_for_the_halt")
        wait.SetValueFromCString("0", error)
        self.assertSuccess(error)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")
