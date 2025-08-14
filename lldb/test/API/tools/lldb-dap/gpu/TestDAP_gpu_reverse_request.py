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

def _detect_rocm():
    """Detects rocm target."""
    try:
        proc = Popen(["rocminfo"], stdout=PIPE, stderr=PIPE)
        return "amd"
    except Exception:
        return None

def skipUnlessHasROCm(func):
    """Decorate the item to skip test unless ROCm is available."""
    
    def has_rocm():
        if _detect_rocm() is None:
            return "ROCm not available (rocm-smi not found)"
        return None
    
    return skipTestIfFn(has_rocm)(func)


class TestDAP_gpu_reverse_request(lldbdap_testcase.DAPTestCaseBase):
    """Test DAP session spawning - both basic and GPU scenarios"""

    def setUp(self):
        super().setUp()
    
    @skipUnlessHasROCm
    def test_automatic_reverse_request_detection(self):
        """Test that we can detect when LLDB automatically sends reverse requests"""
        program = self.getBuildArtifact("a.out")
        
        # Build and launch with settings that might trigger reverse requests
        self.build_and_launch(
            program
        )
        source = "hello_world.hip"
        breakpoint_line = line_number(source, "// CPU BREAKPOINT - BEFORE LAUNCH")
        self.set_source_breakpoints(source, [breakpoint_line])  
        self.continue_to_next_stop()
        
        reverse_request_count = len(self.dap_server.reverse_requests)
        self.assertEqual(reverse_request_count, 1, "Should have received one reverse request")
        # If reverse requests were found, validate them
        req = self.dap_server.reverse_requests[0]
        self.assertIn("command", req, "Reverse request should have command")
        self.assertEqual(req["command"], "startDebugging")
        
        self.assertIn("arguments", req, "Reverse request should have arguments")
        self.assertIn("configuration", req["arguments"], "Reverse request should have configuration")
        
        attach_config = req["arguments"]["configuration"]
        self.assertIn("attachCommands", attach_config, "Reverse request should have attachCommands")
        self.assertIn("target select 1", attach_config["attachCommands"])
        self.assertIn("name", attach_config, "Attach config should have name")
        self.assertIn("GPU Session", attach_config["name"])
        self.assertIn("targetIdx", attach_config, "Attach config should have targetIdx")
        self.assertEqual(attach_config["targetIdx"], 1, "Attach config should have targetIdx 1")
        
    @skipUnlessHasROCm
    def test_gpu_breakpoint_hit(self):
        """
        Test that we can hit a breakpoint in GPU debugging session spawned through reverse requests.
        """
        GPU_SESSION_IDX = 0
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
            connection=connection,
            log_file=log_file_path
        )
        self.launch(
            program, disconnectAutomatically=False,
        )
        # Set CPU breakpoint and stop.
        breakpoint_ids = self.set_source_breakpoints(source, [cpu_breakpoint_line])  
        
        self.continue_to_breakpoints(breakpoint_ids, timeout=self.DEFAULT_TIMEOUT)
        # We should have a GPU child session automatically spawned now
        self.assertEqual(len(self.dap_server.get_child_sessions()), 1)
        # Set breakpoint in GPU session
        gpu_breakpoint_ids = self.set_source_breakpoints_on(GPU_SESSION_IDX, source, [gpu_breakpoint_line])
        
        # Resume GPU execution after verifying breakpoint hit
        self.do_continue_on(0)

        # Continue main session
        self.do_continue()  
        # Verify that the GPU breakpoint is hit in the child session
        self.verify_breakpoint_hit_on(GPU_SESSION_IDX, gpu_breakpoint_ids, timeout=self.DEFAULT_TIMEOUT * 2)
        
        # Manually disconnect sessions
        for child_session in self.dap_server.get_child_sessions():
            child_session.request_disconnect()
        self.dap_server.request_disconnect()
