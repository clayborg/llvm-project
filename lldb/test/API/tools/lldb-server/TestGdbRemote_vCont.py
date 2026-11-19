import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemote_vCont(gdbremote_testcase.GdbRemoteTestCaseBase):
    def vCont_supports_mode(self, mode, inferior_args=None):
        # Setup the stub and set the gdb remote command stream.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=inferior_args)
        self.add_vCont_query_packets()

        # Run the gdb remote command stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Pull out supported modes.
        supported_vCont_modes = self.parse_vCont_query_response(context)
        self.assertIsNotNone(supported_vCont_modes)

        # Verify we support the given mode.
        self.assertIn(mode, supported_vCont_modes)

    def test_vCont_supports_c(self):
        self.build()
        self.vCont_supports_mode("c")

    def test_vCont_supports_C(self):
        self.build()
        self.vCont_supports_mode("C")

    def test_vCont_supports_s(self):
        self.build()
        self.vCont_supports_mode("s")

    def test_vCont_supports_S(self):
        self.build()
        self.vCont_supports_mode("S")

    @add_test_categories(["llgs"])
    def test_vCont_supports_t(self):
        self.build()
        self.vCont_supports_mode("t")

    @add_test_categories(["llgs"])
    def test_vCont_step_until_active_not_supported(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_vCont_query_packets()
        self.add_threadinfo_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertNotIn("v", self.parse_vCont_query_response(context))
        (tid,) = self.parse_threadinfo_packets(context)

        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                f"read packet: $vCont;v:{tid:x}#00",
                "send packet: $E03#00",
                f"read packet: $vCont;c:{tid:x};v:{tid:x}#00",
                "send packet: $E03#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @skipIfWindows  # No pty support to test O* & I* notification packets.
    @skipIf(triple="^mips")
    def test_single_step_only_steps_one_instruction_with_Hc_vCont_s(self):
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="vCont;s"
        )

    @skipIfWindows  # No pty support to test O* & I* notification packets.
    @skipIf(triple="^mips")
    def test_single_step_only_steps_one_instruction_with_vCont_s_thread(self):
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=False, step_instruction="vCont;s:{thread}"
        )
