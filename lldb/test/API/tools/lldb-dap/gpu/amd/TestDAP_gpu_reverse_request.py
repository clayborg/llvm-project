"""
Test DAP reverse request functionality for GPU debugging.
Tests the changes that allow creating new DAP targets through reverse requests,
specifically for GPU debugging scenarios using AMD HIP.
"""

import dap_server
import lldbdap_testcase
from subprocess import Popen, PIPE
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestDAPAMDReverseRequest(lldbdap_testcase.DAPTestCaseBase):
    """Test DAP session spawning - both basic and GPU scenarios"""

    def wait_for_cpu_breakpoint(self, cpu_breakpoint_ids):
        """Wait for the CPU session to stop at one of the breakpoints. It can
        stop at the internal GPU loader breakpoint first."""
        stopped_events = self.dap_server.wait_for_stopped()
        body = stopped_events[0].get("body", {}) if stopped_events else {}
        hit_bp_ids = body.get("hitBreakpointIds", [])
        if not any(str(bp_id) in cpu_breakpoint_ids for bp_id in hit_bp_ids):
            self.continue_to_breakpoints(cpu_breakpoint_ids)

    def run_to_gpu_child_session(self, source):
        """Launch the program from a DAP server and run it to the CPU
        breakpoint after GPU initialization, by which the GPU child session
        has started. Return the CPU breakpoint IDs and the GPU target's ID."""
        self.build()
        _, connection = self.start_server(connection="listen://localhost:0")
        self.dap_server = dap_server.DebugAdapterServer(
            connection=connection, log_file=self.getBuildArtifact("dap.txt")
        )
        self.launch(self.getBuildArtifact("a.out"), disconnectAutomatically=False)
        line = line_number(source, "// CPU BREAKPOINT - AFTER GPU INIT")
        cpu_breakpoint_ids = self.set_source_breakpoints(source, [line])
        self.do_continue()
        self.wait_for_cpu_breakpoint(cpu_breakpoint_ids)
        configuration = self.dap_server.reverse_requests[0]["arguments"][
            "configuration"
        ]
        return cpu_breakpoint_ids, configuration["targetId"]

    def continue_cpu_to_exit(self):
        """Continue the CPU session until the program exits, and return the
        exited event. On the way, the CPU can stop at the internal GPU loader
        breakpoint, as the program unloads its GPU code."""
        while True:
            self.do_continue()
            events = self.dap_server.wait_for_stopped()
            self.assertTrue(events, "The CPU session should stop or exit")
            if events[-1]["event"] == "exited":
                return events[-1]
            hit_bp_ids = events[-1]["body"].get("hitBreakpointIds", [])
            self.assertTrue(
                hit_bp_ids and all(bp_id < 0 for bp_id in hit_bp_ids),
                f"Unexpected stop: {events[-1]}",
            )

    def test_automatic_reverse_request_detection(self):
        """
        Test that we can detect when LLDB automatically sends reverse requests
        """
        program = self.getBuildArtifact("a.out")

        # Build and launch with settings that trigger reverse requests.
        # The breakpoint must be after GPU code objects are loaded (which
        # happens during hipMalloc) so the GPU target has been created.
        self.build_and_launch(program)
        source = "hello_world.hip"
        breakpoint_line = line_number(source, "// CPU BREAKPOINT - AFTER GPU INIT")
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        reverse_request_count = len(self.dap_server.reverse_requests)
        self.assertEqual(
            reverse_request_count, 1, "Should have received one reverse request"
        )

        # Validate the startDebugging reverse request structure
        req = self.dap_server.reverse_requests[0]

        # Check command
        self.assertIn("command", req, "Reverse request should have command")
        self.assertEqual(
            req["command"], "startDebugging", "Command should be startDebugging"
        )

        # Check arguments structure
        self.assertIn("arguments", req, "Reverse request should have arguments")
        args = req["arguments"]

        # Check request type
        self.assertIn("request", args, "Arguments should have request field")
        self.assertEqual(args["request"], "attach", "Request type should be 'attach'")

        # Check configuration
        self.assertIn("configuration", args, "Arguments should have configuration")
        config = args["configuration"]

        # Check configuration.name (session name from GPU plugin)
        self.assertIn("name", config, "Configuration should have name")
        self.assertEqual(
            config["name"],
            "AMD GPU Session",
            "Session name should be 'AMD GPU Session'",
        )

        # Check configuration.debuggerId (ID of existing debugger to reuse)
        self.assertIn("debuggerId", config, "Configuration should have debuggerId")
        self.assertIsInstance(
            config["debuggerId"], int, "debuggerId should be an integer"
        )

        # Check configuration.targetId (ID of GPU target to attach to)
        self.assertIn("targetId", config, "Configuration should have targetId")
        self.assertIsInstance(config["targetId"], int, "targetId should be an integer")
        self.assertGreater(config["targetId"], 1, "GPU target ID should be > 1")

    def test_gpu_breakpoint_hit(self):
        """
        Test that we can hit a breakpoint in GPU debugging session spawned through reverse requests.
        """
        self.build()
        log_file_path = self.getBuildArtifact("dap.txt")
        program = self.getBuildArtifact("a.out")
        source = "hello_world.hip"
        cpu_breakpoint_line = line_number(source, "// CPU BREAKPOINT - AFTER GPU INIT")
        gpu_breakpoint_line = line_number(source, "// GPU BREAKPOINT")
        # Launch DAP server with connection mode (required for child sessions)
        _, connection = self.start_server(connection="listen://localhost:0")

        self.dap_server = dap_server.DebugAdapterServer(
            connection=connection, log_file=log_file_path
        )
        self.launch(
            program,
            disconnectAutomatically=False,
        )

        # Set CPU breakpoint.
        breakpoint_ids = self.set_source_breakpoints(source, [cpu_breakpoint_line])

        # Continue - with lazy GPU target creation, the process may first stop
        # at the internal GPU loader breakpoint (_loader_debug_state) before
        # reaching the user breakpoint. The internal breakpoint stop triggers
        # the startDebugging reverse request which creates the GPU child session.
        stopped_events = self.continue_to_next_stop()
        body = stopped_events[0].get("body", {}) if stopped_events else {}
        hit_bp_ids = body.get("hitBreakpointIds", [])

        # If we stopped at an internal breakpoint (negative ID), continue
        # to the user breakpoint.
        if not any(str(bp_id) in breakpoint_ids for bp_id in hit_bp_ids):
            self.continue_to_breakpoints(breakpoint_ids)

        # We should have a GPU child session automatically spawned now
        self.assertEqual(
            len(self.dap_server.get_child_sessions()), 1, "Expected 1 child GPU session"
        )

        # Get the GPU target ID from the reverse request
        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "Expected 1 startDebugging reverse request",
        )
        reverse_req = self.dap_server.reverse_requests[0]
        gpu_target_id = reverse_req["arguments"]["configuration"]["targetId"]

        # Set breakpoint in GPU session
        gpu_breakpoint_ids = self.set_source_breakpoints_on(
            gpu_target_id, source, [gpu_breakpoint_line], wait_for_resolve=False
        )

        # Continue both GPU and CPU sessions
        self.do_continue_on(gpu_target_id)
        self.do_continue()
        # Verify that the GPU breakpoint is hit in the child session
        self.verify_breakpoint_hit_on(gpu_target_id, gpu_breakpoint_ids)

        # Manually disconnect sessions - must terminate debuggee to prevent
        # orphaned processes that hang indefinitely waiting on GPU synchronization.
        # Killing the main session will terminate the debuggee and the DAP server
        # will automatically disconnect all child GPU sessions.
        self.dap_server.request_disconnect(terminateDebuggee=True)

    def test_gpu_child_disconnect_keeps_cpu_session(self):
        """
        Test that ending the GPU child session, as when the user stops it,
        leaves the CPU session to run the program to its end.
        """
        _, gpu_target_id = self.run_to_gpu_child_session("hello_world.hip")
        child = self.dap_server.get_child_sessions()[gpu_target_id]
        self.assertTrue(child.request_disconnect()["success"])
        self.continue_to_exit()
        self.dap_server.request_disconnect()

    def test_program_exit_ends_gpu_child_session(self):
        """
        Test that when the program exits, the GPU child session reports it
        too, since the GPU process exits with the CPU process.
        """
        _, gpu_target_id = self.run_to_gpu_child_session("hello_world.hip")
        child = self.dap_server.get_child_sessions()[gpu_target_id]
        self.assertEqual(self.continue_cpu_to_exit()["body"]["exitCode"], 0)
        self.assertEqual(child.wait_for_exited()["body"]["exitCode"], 0)
        self.dap_server.request_disconnect()

    def test_gpu_restart_reuses_gpu_target(self):
        """
        Test that restarting starts a new GPU child session on the same GPU
        target, where the breakpoint set in the first child session is hit.
        """
        source = "hello_world.hip"
        cpu_breakpoint_ids, gpu_target_id = self.run_to_gpu_child_session(source)
        gpu_breakpoint_line = line_number(source, "// GPU BREAKPOINT")
        gpu_breakpoint_ids = self.set_source_breakpoints_on(
            gpu_target_id, source, [gpu_breakpoint_line], wait_for_resolve=False
        )
        first_child = self.dap_server.get_child_sessions()[gpu_target_id]

        self.dap_server.request_restart()
        self.wait_for_cpu_breakpoint(cpu_breakpoint_ids)
        reverse_requests = self.dap_server.reverse_requests
        self.assertEqual(len(reverse_requests), 2)
        self.assertEqual(
            reverse_requests[1]["arguments"]["configuration"]["targetId"],
            gpu_target_id,
        )
        new_child = self.dap_server.get_child_sessions()[gpu_target_id]
        self.assertIsNot(new_child, first_child)

        # Restarting detached the first child session's GPU process, but the
        # session stays open until the client disconnects it. Events for the
        # reused target must go to the new session.
        self.do_continue_on(gpu_target_id)
        self.do_continue()
        self.verify_breakpoint_hit_on(gpu_target_id, gpu_breakpoint_ids)

        # Disconnecting the first child session, as the client does after
        # "terminated", must leave the new session's process alone.
        self.assertTrue(first_child.request_disconnect()["success"])
        self.assertTrue(new_child.request_threads()["success"])
        self.dap_server.request_disconnect(terminateDebuggee=True)
