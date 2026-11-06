import gdbremote_testcase
import os
import signal
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestStubSIGINTTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    @skipIfWindows
    @skipIfRemote  # The stub's pid is not a local process to signal.
    @add_test_categories(["llgs"])
    def test_sigint_shuts_down(self):
        """lldb-server shuts down on SIGINT as it does when its client
        disconnects, and exits normally instead of being killed."""
        server = self.connect_to_debug_monitor()
        os.kill(server.pid, signal.SIGINT)
        self.assertEqual(server.wait(timeout=self.DEFAULT_TIMEOUT), 0)
