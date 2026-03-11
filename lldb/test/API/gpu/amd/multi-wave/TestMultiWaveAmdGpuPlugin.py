"""
Basic tests for the AMDGPU plugin with a multi-wave kernel.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SHADOW_THREAD_NAME = "AMD Native Shadow Thread"


class BasicAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_breakpoint(self):
        # GPU breakpoint should get hit by at least one thread.
        source = "multi-wave.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        return gpu_threads_at_bp

    def test_num_threads(self):
        """Test that we get the expected number of threads."""
        self.build()

        gpu_threads_at_breakpoint = self.run_to_breakpoint()

        # We launch 960 total threads (8 blocks * 120 threads per block).
        gpu_threads = self.gpu_process.threads
        self.assertEqual(len(gpu_threads), 960)

        # But not all waves may reach the breakpoint at the same time.
        # So here we check that we have at least one wave's worth of threads
        # stopped at the breakpoint. With wave size of 64, this means we should
        # have at least 56 threads (120 = 64 + 56) hitting the breakpoint.
        self.assertGreaterEqual(len(gpu_threads_at_breakpoint), 56)
