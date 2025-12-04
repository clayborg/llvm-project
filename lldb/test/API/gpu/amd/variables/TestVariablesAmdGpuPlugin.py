"""
Variable tests for the AMDGPU plugin.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *
from textwrap import dedent
from lldbsuite.test.decorators import *

SOURCE = "variables.hip"
NUM_THREADS = 32


class VariablesAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_first_gpu_breakpoint(self):
        """Helper to run to common gpu breakpoint"""
        self.build()

        gpu_threads = self.run_to_gpu_breakpoint(
            SOURCE, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")
        self.assertEqual(
            len(gpu_threads),
            NUM_THREADS,
            f"Expected {NUM_THREADS} threads to be stopped",
        )
        self.select_gpu()
        return gpu_threads

    def check_variable(self, thread, name, expected_values):
        """Helper to check the value of a thread local variable."""
        if isinstance(expected_values, str):
            expected_values = [expected_values]
        self.gpu_process.SetSelectedThread(thread)
        self.expect(
            f"frame variable --flat --show-all-children {name}",
            substrs=expected_values,
        )

    def test_private(self):
        """Test reading from the thread private variables."""
        # Note that the kernel is launched with NUM_THREADS threads and one block.
        gpu_threads = self.run_to_first_gpu_breakpoint()
        lane_0, lane_1 = gpu_threads[0], gpu_threads[1]

        # The thread_scalar value is computed as threadIdx.x * 2.
        self.check_variable(lane_0, "thread_scalar", "thread_scalar = 0")
        self.check_variable(lane_1, "thread_scalar", "thread_scalar = 2")

        # The thread_array value is computed as thread_array[i] = threadIdx.x + i.
        self.check_variable(
            lane_0,
            "thread_array",
            [
                "thread_array[0] = 0",
                "thread_array[1] = 1",
                "thread_array[2] = 2",
                "thread_array[3] = 3",
            ],
        )
        self.check_variable(
            lane_1,
            "thread_array",
            [
                "thread_array[0] = 1",
                "thread_array[1] = 2",
                "thread_array[2] = 3",
                "thread_array[3] = 4",
            ],
        )

        # The thread_struct value is computed as (threadIdx.x * 10, threadIdx.x * 20).
        self.check_variable(
            lane_0, "thread_struct", ["thread_struct.x = 0", "thread_struct.y = 0"]
        )
        self.check_variable(
            lane_1, "thread_struct", ["thread_struct.x = 10", "thread_struct.y = 20"]
        )

    def test_shared(self):
        """Test reading from the shared variables."""
        # Note that the kernel is launched with NUM_THREADS threads and one block.
        gpu_threads = self.run_to_first_gpu_breakpoint()
        lane_0 = gpu_threads[0]

        # The shared_scalar value is computed as blockIdx.x + 1000.
        self.check_variable(lane_0, "shared_scalar", "shared_scalar = 1000")

        # The shared_array value is computed as shared_array[threadIdx.x] = threadIdx.x.
        self.check_variable(
            lane_0,
            "shared_array",
            [f"shared_array[{i}] = {i}" for i in range(NUM_THREADS)],
        )

        # The shared_struct value is computed as (blockIdx.x + 100, blockIdx.x + 200).
        self.check_variable(
            lane_0, "shared_struct", ["shared_struct.x = 100", "shared_struct.y = 200"]
        )

    @expectedFailureAll("Read of global variables is not working yet.")
    def test_global(self):
        """Test reading from the device global variables."""
        # Note that the kernel is launched with NUM_THREADS threads and one block.
        gpu_threads = self.run_to_first_gpu_breakpoint()
        lane_0 = gpu_threads[0]

        # The g_scalar value is initialized to 100.
        self.check_variable(lane_0, "g_scalar", "g_scalar = 100")

        # The g_array value is initialized to {10, 20, 30, 40}.
        self.check_variable(
            lane_0,
            "g_array",
            [
                "g_array[0] = 10",
                "g_array[1] = 20",
                "g_array[2] = 30",
                "g_array[3] = 40",
            ],
        )

        # The g_struct value is initialized to (50,  60).
        self.check_variable(lane_0, "g_struct", ["g_struct.x = 50", "g_struct.y = 60"])

    def test_global_copy(self):
        """Test reading from the device global variables copied to locals."""
        # Note that the kernel is launched with NUM_THREADS threads and one block.
        gpu_threads = self.run_to_first_gpu_breakpoint()
        lane_0, lane_1 = gpu_threads[0], gpu_threads[1]

        # The global_scalar_value is initialized to g_scalar (100).
        self.check_variable(lane_0, "global_scalar_value", "global_scalar_value = 100")

        # The global_array_value is initialized to g_array[threadIdx.x % 4] (10, 20, 30, 40).
        self.check_variable(lane_0, "global_array_value", "global_array_value = 10")
        self.check_variable(lane_1, "global_array_value", "global_array_value = 20")

        # The global_struct_value is initialized to g_struct.
        self.check_variable(
            lane_0,
            "global_struct_value",
            ["global_struct_value.x = 50", "global_struct_value.y = 60"],
        )
