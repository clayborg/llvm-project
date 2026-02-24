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

    def test_automatic_reverse_request_detection(self):
        """
        Test that we can detect when LLDB automatically sends reverse requests
        """
        program = self.getBuildArtifact("a.out")

        # Build and launch with settings that trigger reverse requests
        self.build_and_launch(program)
        source = "hello_world.hip"
        breakpoint_line = line_number(source, "// CPU BREAKPOINT - BEFORE LAUNCH")
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
        # Enable detailed DAP logging to debug any issues
        program = self.getBuildArtifact("a.out")
        source = "hello_world.hip"
        cpu_breakpoint_line = line_number(source, "// CPU BREAKPOINT")
        gpu_breakpoint_line = line_number(source, "// GPU BREAKPOINT")
        # Launch DAP server
        _, connection = self.start_server(connection="listen://localhost:0")

        self.dap_server = dap_server.DebugAdapterServer(
            connection=connection, log_file=log_file_path
        )
        self.launch(
            program,
            disconnectAutomatically=False,
        )

        # Set CPU breakpoint and stop.
        breakpoint_ids = self.set_source_breakpoints(source, [cpu_breakpoint_line])
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
