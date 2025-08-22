"""
Packets tests for the Mock GPU Plugin.
"""

import json

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class PacketsMockGpuTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    def setUp(self):
        super(PacketsMockGpuTestCase, self).setUp()
        self.build()

    def _make_breakpoint_packet(
        self, identifier, function_name, symbol_names, symbol_values
    ):
        data = {
            "breakpoint": {
                "addr_info": None,
                "identifier": identifier,
                "name_info": {"function_name": function_name, "shlib": "a.out"},
                "symbol_names": symbol_names,
            },
            "plugin_name": "mock-gpu",
            "symbol_values": symbol_values,
        }
        # GDB Remote Protocol escaping: } becomes }]
        return json.dumps(data, separators=(",", ":")).replace("}", "}]")

    def test_ordered_gpu_plugin_packet_sequence(self):
        """Test the full ordered packet exchange for the Mock GPU Plugin."""

        _procs = self.prep_debug_monitor_and_inferior()
        self.assertIsNotNone(_procs)

        # Define the expected sequence of packets and their content checks
        packet_sequence = [
            {
                "send": "jGPUPluginInitialize",
                "payload": None,
                "expect_regex": r"^\$(\[.*\])#[0-9a-fA-F]{2}$",
                "content_checks": [
                    '"breakpoints":',
                    '"plugin_name":"mock-gpu"',
                    '"identifier":"gpu_initialize"',
                    '"function_name":"gpu_initialize"',
                    '"shlib":"a.out"',
                    '"symbol_names":["gpu_shlib_load"]',
                ],
            },
            {
                "send": "jGPUPluginBreakpointHit",
                "payload": self._make_breakpoint_packet(
                    "gpu_initialize",
                    "gpu_initialize",
                    ["gpu_shlib_load"],
                    [{"name": "gpu_shlib_load", "value": 4198710}],
                ),
                "expect_regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                "content_checks": [
                    '"breakpoints":',
                    '"identifier":"gpu_shlib_load"',
                    '"load_address":',
                    '"connect_info":',
                    '"connect_url":',
                    "localhost:",
                    '"load_libraries":false',
                    '"plugin_name":"mock-gpu"',
                    '"disable_bp":true',
                ],
            },
            {
                "send": "jGPUPluginBreakpointHit",
                "payload": self._make_breakpoint_packet(
                    "gpu_shlib_load",
                    "gpu_shlib_load",
                    ["gpu_third_stop"],
                    [{"name": "gpu_third_stop", "value": 4210736}],
                ),
                "expect_regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                "content_checks": [
                    '"load_libraries":true',
                    '"plugin_name":"mock-gpu"',
                    '"disable_bp":false',
                ],
            },
            {
                "send": "jGPUPluginBreakpointHit",
                "payload": self._make_breakpoint_packet(
                    "gpu_third_stop", "gpu_third_stop", [], []
                ),
                "expect_regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                "content_checks": [
                    '"load_libraries":false',
                    '"plugin_name":"mock-gpu"',
                    '"disable_bp":false',
                ],
            },
            {
                "send": "jGPUPluginBreakpointHit",
                "payload": self._make_breakpoint_packet(
                    "gpu_shlib_load",
                    "gpu_shlib_load",
                    ["gpu_third_stop"],
                    [{"name": "g_shlib_list", "value": 4210736}],
                ),
                "expect_regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                "content_checks": [
                    '"load_libraries":true',
                    '"plugin_name":"mock-gpu"',
                    '"disable_bp":false',
                ],
            },
        ]

        for idx, pkt in enumerate(packet_sequence):
            raw = (
                f"${pkt['send']}"
                if pkt["payload"] is None
                else f"${pkt['send']}:{pkt['payload']}"
            )
            checksum = sum(ord(c) for c in raw[1:]) % 256
            packet_str = f"{raw}#{checksum:02x}"

            self.test_sequence.add_log_lines(
                [
                    f"read packet: {packet_str}",
                    {
                        "direction": "send",
                        "regex": pkt["expect_regex"],
                        "capture": {1: f"response_{idx}"},
                    },
                ],
                True,
            )

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)
            response = context.get(f"response_{idx}")
            self.assertIsNotNone(response)

            for check in pkt["content_checks"]:
                self.assertIn(
                    check,
                    response,
                    f"Packet {idx} missing '{check}' in response: {response}",
                )
