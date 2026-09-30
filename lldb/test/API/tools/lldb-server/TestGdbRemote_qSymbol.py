import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestGdbRemote_qSymbol(gdbremote_testcase.GdbRemoteTestCaseBase):
    """lldb-server serves qSymbol for its GPU plug-ins. Without a plug-in that
    needs symbols, every round ends right away."""

    def start_server(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()

    @add_test_categories(["llgs"])
    def test_rounds_end_without_symbols_to_look_up(self):
        self.start_server()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qSymbol::#00",
                "send packet: $OK#00",
                # Answers for names nobody requested end the round too, whether
                # the symbol was found or not.
                "read packet: $qSymbol:1000:6578616d706c65#00",
                "send packet: $OK#00",
                "read packet: $qSymbol::6578616d706c65#00",
                "send packet: $OK#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_answer_without_a_name_is_rejected(self):
        """Only qSymbol:: opens a round, so an answer that is missing the
        symbol name is an error rather than the start of a new round."""
        self.start_server()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qSymbol:6578616d706c65#00",
                "send packet: $E03#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()
