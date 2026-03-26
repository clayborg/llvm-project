"""
Test DAP reverse request functionality with the mock GPU plugin.
No real GPU hardware required.
"""

import dap_server
import lldbdap_testcase
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestDAPMockGPUReverseRequest(lldbdap_testcase.DAPTestCaseBase):
    def _find_mock_gpu_reverse_request(self):
        """Find the startDebugging reverse request from the mock GPU plugin."""
        for req in self.dap_server.reverse_requests:
            config = req.get("arguments", {}).get("configuration", {})
            if config.get("name") == "Mock GPU Session":
                return req
        return None

    def test_automatic_reverse_request_detection(self):
        """Test that the mock GPU plugin sends a correctly structured
        startDebugging reverse request."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "hello_world.cpp"
        breakpoint_line = line_number(source, "// CPU BREAKPOINT - AFTER")
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        req = self._find_mock_gpu_reverse_request()
        self.assertIsNotNone(req, "Missing Mock GPU Session reverse request")
        self.assertEqual(req["command"], "startDebugging")

        args = req["arguments"]
        self.assertEqual(args["request"], "attach")

        config = args["configuration"]
        self.assertEqual(config["name"], "Mock GPU Session")
        self.assertIsInstance(config["debuggerId"], int)
        self.assertIsInstance(config["targetId"], int)
        self.assertGreater(config["targetId"], 1)

    def test_dap_requests_to_cpu_and_gpu_targets(self):
        """Test DAP requests to both the CPU and mock GPU child session,
        including instruction breakpoints and continue."""
        program = self.getBuildArtifact("a.out")
        source = "hello_world.cpp"
        breakpoint_line_1 = line_number(source, "// CPU BREAKPOINT")
        breakpoint_line_2 = line_number(source, "// CPU BREAKPOINT - AFTER")

        # Server mode (TCP) is required for child session support.
        self.build()
        _, connection = self.start_server(connection="listen://localhost:0")
        self.dap_server = dap_server.DebugAdapterServer(
            connection=connection,
            spawn_helper=self.spawnSubprocess,
        )
        self.launch(program, disconnectAutomatically=False)

        breakpoint_ids = self.set_source_breakpoints(
            source, [breakpoint_line_1, breakpoint_line_2]
        )
        self.continue_to_breakpoints(breakpoint_ids)

        # --- CPU target: first breakpoint ---
        cpu_thread_id = self.dap_server.request_threads()["body"]["threads"][0]["id"]
        stack_resp = self.dap_server.request_stackTrace(threadId=cpu_thread_id)
        self.assertTrue(stack_resp["success"])
        frames = stack_resp["body"]["stackFrames"]
        self.assertEqual(frames[0].get("source", {}).get("name", ""), "hello_world.cpp")
        self.assertEqual(frames[0]["line"], breakpoint_line_1)

        eval_resp = self.dap_server.request_evaluate("argc", threadId=cpu_thread_id)
        self.assertTrue(eval_resp["success"])

        # --- CPU target: continue to second breakpoint ---
        self.continue_to_breakpoints(breakpoint_ids)

        cpu_thread_id = self.dap_server.request_threads()["body"]["threads"][0]["id"]
        stack_resp = self.dap_server.request_stackTrace(threadId=cpu_thread_id)
        self.assertTrue(stack_resp["success"])
        self.assertEqual(
            stack_resp["body"]["stackFrames"][0]["line"], breakpoint_line_2
        )

        eval_resp = self.dap_server.request_evaluate("argc", threadId=cpu_thread_id)
        self.assertTrue(eval_resp["success"])

        # --- Mock GPU child session ---
        child_sessions = self.dap_server.get_child_sessions()
        self.assertGreater(len(child_sessions), 0, "No GPU child session found")

        gpu_session = list(child_sessions.values())[0]

        # Threads
        gpu_threads_resp = gpu_session.request_threads()
        self.assertTrue(gpu_threads_resp["success"])
        gpu_threads = gpu_threads_resp["body"]["threads"]
        self.assertGreater(len(gpu_threads), 0)

        # Stack trace
        gpu_thread_id = gpu_threads[0]["id"]
        gpu_stack_resp = gpu_session.request_stackTrace(threadId=gpu_thread_id)
        self.assertTrue(gpu_stack_resp["success"])

        # Scopes
        gpu_frames = gpu_stack_resp["body"]["stackFrames"]
        self.assertGreater(len(gpu_frames), 0)
        scopes_resp = gpu_session.request_scopes(gpu_frames[0]["id"])
        self.assertTrue(scopes_resp["success"])
        self.assertGreater(len(scopes_resp["body"]["scopes"]), 0)

        # Instruction breakpoint — slow (~10s) due to resolving against
        # mock GPU modules whose files don't exist on disk.
        command_dict = {
            "command": "setInstructionBreakpoints",
            "type": "request",
            "arguments": {"breakpoints": [{"instructionReference": "0x80000"}]},
        }
        seq = gpu_session.send_packet(command_dict)
        bp_resp = gpu_session._recv_packet(
            predicate=lambda p: p.get("type") == "response"
            and p.get("request_seq") == seq,
            timeout=30,
        )
        self.assertIsNotNone(bp_resp, "setInstructionBreakpoints timed out")
        self.assertTrue(bp_resp["success"])
        self.assertEqual(len(bp_resp["body"]["breakpoints"]), 1)

        # Verify GPU session is still functional after setting breakpoints.
        gpu_threads_resp = gpu_session.request_threads()
        self.assertTrue(gpu_threads_resp["success"])
        gpu_stack_resp = gpu_session.request_stackTrace(
            threadId=gpu_threads_resp["body"]["threads"][0]["id"]
        )
        self.assertTrue(gpu_stack_resp["success"])
        self.assertGreater(len(gpu_stack_resp["body"]["stackFrames"]), 0)

        # Continue and verify GPU session still works.
        gpu_session.request_continue()
        gpu_session.wait_for_stopped()

        gpu_threads_resp = gpu_session.request_threads()
        self.assertTrue(gpu_threads_resp["success"])
        gpu_stack_resp = gpu_session.request_stackTrace(
            threadId=gpu_threads_resp["body"]["threads"][0]["id"]
        )
        self.assertTrue(gpu_stack_resp["success"])

        # Disconnect child before parent to avoid event processing conflicts.
        gpu_session.request_disconnect()
        try:
            self.dap_server.request_disconnect()
        except ValueError:
            pass
