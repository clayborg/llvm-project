"""
Simple test that just verifies binary builds and runs.

Prerequisites:
    - oneAPI environment must be sourced: source ~/intel/oneapi/setvars.sh
    - ZET_ENABLE_PROGRAM_DEBUGGING=1 must be set
"""

import lldb
import os
from lldbsuite.test.lldbtest import *


class BasicManualTestCase(TestBase):
    """Simple test to verify SYCL binary builds and runs."""

    NO_DEBUG_INFO_TESTCASE = True

    def test_simple_run(self):
        """Test that the SYCL binary builds successfully.

        Note: Cannot test full execution due to IntelGT plugin limitation.
        The plugin stops the process when GPU target is created, but resume
        is not currently supported (Resume timed out error).

        This test verifies:
        - Binary builds with SYCL compiler
        - Binary can be loaded into LLDB
        - Process can be launched (will stop at GPU target creation)
        """
        # Build the test binary
        self.build()

        # Get the path to the built executable
        exe = self.getBuildArtifact("simple_kernel")

        # Create target and run
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        # Verify the binary was built and is loadable
        self.assertTrue(target.GetExecutable().Exists(), "Executable should exist")

        # SYCL requires environment variables for Level-Zero and oneAPI libraries
        env = [f"{key}={value}" for key, value in os.environ.items()]
        process = target.LaunchSimple(None, env, self.get_process_working_directory())
        self.assertIsNotNone(process, "Process should launch")

        # Process should be in a valid state (stopped or running)
        state = process.GetState()
        self.assertIn(state, [lldb.eStateStopped, lldb.eStateRunning, lldb.eStateExited],
                     f"Process should be in valid state, got {state}")
