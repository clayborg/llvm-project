import re

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

    def vCont_overlapping_selectors(self, default_first, multiprocess=False):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior(inferior_args=["retval:23"])

        thread_pattern = r"[0-9a-f]+"
        if multiprocess:
            self.add_qSupported_packets(["multiprocess+"])
            context = self.expect_gdbremote_sequence()
            if "multiprocess+" not in context["qSupported_response"].split(";"):
                self.skipTest("stub does not support multiprocess")
            self.reset_test_sequence()
            thread_pattern = r"p[0-9a-f]+\.[0-9a-f]+"

        self.test_sequence.add_log_lines(
            [
                "read packet: $qC#00",
                {
                    "direction": "send",
                    "regex": rf"^\$QC({thread_pattern})#[0-9a-f]{{2}}$",
                    "capture": {1: "thread_id"},
                },
            ],
            True,
        )
        thread_id = self.expect_gdbremote_sequence()["thread_id"]
        default = "c"
        exit_reply = "$W17"
        if multiprocess:
            process_id = thread_id.split(".")[0][1:]
            default = f"c:p{process_id}.-1"
            exit_reply += f";process:{process_id}"

        step = f"s:{thread_id}"
        self.reset_test_sequence()
        if default_first:
            self.test_sequence.add_log_lines(
                [f"read packet: $vCont;{default};{step}#00"], True
            )
        else:
            self.test_sequence.add_log_lines(
                [
                    f"read packet: $vCont;{step};{default}#00",
                    {
                        "direction": "send",
                        "regex": rf"^\$T05thread:{re.escape(thread_id)};",
                    },
                    "read packet: $vCont;c#00",
                ],
                True,
            )
        self.test_sequence.add_log_lines([f"send packet: {exit_reply}#00"], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_vCont_global_continue_before_thread_step(self):
        self.vCont_overlapping_selectors(default_first=True)

    @add_test_categories(["llgs"])
    @skipIf(triple="^mips")
    def test_vCont_thread_step_before_global_continue(self):
        self.vCont_overlapping_selectors(default_first=False)

    @add_test_categories(["llgs"])
    def test_vCont_process_continue_before_thread_step(self):
        self.vCont_overlapping_selectors(default_first=True, multiprocess=True)

    @add_test_categories(["llgs"])
    @skipIf(triple="^mips")
    def test_vCont_thread_step_before_process_continue(self):
        self.vCont_overlapping_selectors(default_first=False, multiprocess=True)

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
