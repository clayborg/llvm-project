import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestTargetSyncStopReason(GDBRemoteTestBase):
    def test(self):
        """A thread that a stop reply reports with reason "target sync" stops
        for eStopReasonTargetSync, and is not selected over a thread that
        stopped for a reason of its own. Its description names no target when
        lldb has none for the plug-in that the reply names."""

        class MyResponder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "m1,2"

            def haltReason(self):
                return "T02thread:1;"

            def cont(self):
                # The server stopped thread 2 with SIGSTOP to keep the process
                # in sync with the target of the "mock-gpu" plug-in.
                return "T13thread:2;reason:target sync;sync-plugin:mock-gpu;"

            def threadStopInfo(self, threadnum):
                # Meanwhile thread 1 stopped for SIGINT. An empty reply would
                # tell lldb that the packet is not supported.
                if threadnum == 1:
                    return "T02thread:1;reason:signal;"
                return "T00thread:%x;" % threadnum

        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.assertSuccess(process.Continue())
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetThreadByID(2)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonTargetSync)
        self.assertEqual(thread.GetStopDescription(256), "target sync")
        self.assertEqual(process.GetSelectedThread().GetThreadID(), 1)
