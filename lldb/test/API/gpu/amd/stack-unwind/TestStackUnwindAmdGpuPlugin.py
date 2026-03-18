"""
Stack unwinding and caller-frame variable tests for the AMDGPU plugin (live).

stack_unwind.hip launches 1 block / 32 threads with a three-deep call chain:

    stack_unwind_kernel  ->  middle_function  ->  leaf_function

The GPU breakpoint is set inside leaf_function so all three frames are on the
stack.  The test verifies:
  1. The backtrace contains the expected three frames.
  2. Each frame reports the correct function name.
  3. Local variables in non-leaf frames (middle_function) are readable and
     have the expected values.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from amdgpu_testcase import *

SOURCE = "stack_unwind.hip"
NUM_THREADS = 32


class StackUnwindAmdGpuTestCase(AmdGpuTestCaseBase):
    """Live-debugging tests for GPU stack unwinding and caller-frame variables."""

    # -----------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------

    def run_to_leaf_breakpoint(self):
        """Build, launch, and stop at the leaf_function GPU breakpoint."""
        self.build()

        gpu_threads = self.run_to_gpu_breakpoint(
            SOURCE, "// GPU BREAKPOINT"
        )
        self.assertIsNotNone(gpu_threads, "GPU should be stopped at breakpoint")
        self.assertEqual(
            len(gpu_threads),
            NUM_THREADS,
            f"Expected {NUM_THREADS} threads stopped at breakpoint",
        )
        self.select_gpu()
        return gpu_threads

    # -----------------------------------------------------------------
    # Backtrace / frame structure
    # -----------------------------------------------------------------

    def test_backtrace_depth(self):
        """The GPU thread should have at least 3 frames (leaf, middle, kernel)."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]

        num_frames = thread.GetNumFrames()
        self.assertGreaterEqual(
            num_frames,
            3,
            f"Expected >= 3 frames in backtrace, got {num_frames}",
        )

    def test_backtrace_function_names(self):
        """Each frame should report the correct function name."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]

        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        frame2 = thread.GetFrameAtIndex(2)

        self.assertIn(
            "leaf_function",
            frame0.GetFunctionName(),
            f"Frame 0 should be leaf_function, got '{frame0.GetFunctionName()}'",
        )
        self.assertIn(
            "middle_function",
            frame1.GetFunctionName(),
            f"Frame 1 should be middle_function, got '{frame1.GetFunctionName()}'",
        )
        self.assertIn(
            "stack_unwind_kernel",
            frame2.GetFunctionName(),
            f"Frame 2 should be stack_unwind_kernel, got '{frame2.GetFunctionName()}'",
        )

    def test_frame_pc_values(self):
        """Each frame should have a non-zero PC."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]

        for i in range(min(thread.GetNumFrames(), 3)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid(), f"Frame {i} should be valid")
            pc = frame.GetPC()
            self.assertNotEqual(pc, 0, f"Frame {i} PC should be non-zero")

    # -----------------------------------------------------------------
    # Leaf frame (frame 0) variables
    # -----------------------------------------------------------------

    def test_leaf_frame_variables(self):
        """Verify local variables in the leaf frame (frame 0)."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]
        self.gpu_process.SetSelectedThread(thread)

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())

        # tid == 0 for lane 0, so leaf_local = 0 * 100 = 0
        var = frame.FindVariable("leaf_local")
        self.assertTrue(var.IsValid(), "Should find 'leaf_local'")
        self.assertEqual(var.GetValueAsSigned(), 0, "leaf_local = tid * 100 = 0")

        # scalar_arg == middle_scalar == 0 * 10 + 5 = 5
        var = frame.FindVariable("scalar_arg")
        self.assertTrue(var.IsValid(), "Should find 'scalar_arg'")
        self.assertEqual(var.GetValueAsSigned(), 5, "scalar_arg = 5")

    # -----------------------------------------------------------------
    # Middle frame (frame 1) variables — the core of stack unwinding
    # -----------------------------------------------------------------

    def test_middle_frame_scalar(self):
        """Verify scalar variable in the middle frame after unwinding."""
        gpu_threads = self.run_to_leaf_breakpoint()
        lane_0, lane_1 = gpu_threads[0], gpu_threads[1]

        # lane 0: middle_scalar = 0 * 10 + 5 = 5
        self.gpu_process.SetSelectedThread(lane_0)
        frame1 = lane_0.GetFrameAtIndex(1)
        self.assertTrue(frame1.IsValid(), "Middle frame should be valid")
        self.assertIn("middle_function", frame1.GetFunctionName())

        var = frame1.FindVariable("middle_scalar")
        self.assertTrue(var.IsValid(), "Should find 'middle_scalar' in frame 1")
        self.assertEqual(
            var.GetValueAsSigned(), 5, "lane 0: middle_scalar = 0*10+5 = 5"
        )

        # lane 1: middle_scalar = 1 * 10 + 5 = 15
        self.gpu_process.SetSelectedThread(lane_1)
        frame1 = lane_1.GetFrameAtIndex(1)
        var = frame1.FindVariable("middle_scalar")
        self.assertTrue(var.IsValid(), "Should find 'middle_scalar' in frame 1")
        self.assertEqual(
            var.GetValueAsSigned(), 15, "lane 1: middle_scalar = 1*10+5 = 15"
        )

    def test_middle_frame_array(self):
        """Verify array variable in the middle frame after unwinding."""
        gpu_threads = self.run_to_leaf_breakpoint()
        lane_0, lane_1 = gpu_threads[0], gpu_threads[1]

        # lane 0: middle_array[i] = 0 + i + 100 = {100, 101, 102, 103}
        self.gpu_process.SetSelectedThread(lane_0)
        frame1 = lane_0.GetFrameAtIndex(1)
        var = frame1.FindVariable("middle_array")
        self.assertTrue(var.IsValid(), "Should find 'middle_array' in frame 1")
        for i in range(4):
            elem = var.GetChildAtIndex(i)
            self.assertTrue(elem.IsValid(), f"middle_array[{i}] should be valid")
            self.assertEqual(
                elem.GetValueAsSigned(),
                100 + i,
                f"lane 0: middle_array[{i}] = {100 + i}",
            )

        # lane 1: middle_array[i] = 1 + i + 100 = {101, 102, 103, 104}
        self.gpu_process.SetSelectedThread(lane_1)
        frame1 = lane_1.GetFrameAtIndex(1)
        var = frame1.FindVariable("middle_array")
        self.assertTrue(var.IsValid(), "Should find 'middle_array' in frame 1")
        for i in range(4):
            elem = var.GetChildAtIndex(i)
            self.assertEqual(
                elem.GetValueAsSigned(),
                101 + i,
                f"lane 1: middle_array[{i}] = {101 + i}",
            )

    def test_middle_frame_struct(self):
        """Verify struct variable in the middle frame after unwinding."""
        gpu_threads = self.run_to_leaf_breakpoint()
        lane_0, lane_1 = gpu_threads[0], gpu_threads[1]

        # lane 0: middle_struct = {x=0*3=0, y=0*7=0}
        self.gpu_process.SetSelectedThread(lane_0)
        frame1 = lane_0.GetFrameAtIndex(1)
        var = frame1.FindVariable("middle_struct")
        self.assertTrue(var.IsValid(), "Should find 'middle_struct' in frame 1")
        self.assertEqual(
            var.GetChildMemberWithName("x").GetValueAsSigned(),
            0,
            "lane 0: middle_struct.x = 0",
        )
        self.assertEqual(
            var.GetChildMemberWithName("y").GetValueAsSigned(),
            0,
            "lane 0: middle_struct.y = 0",
        )

        # lane 1: middle_struct = {x=1*3=3, y=1*7=7}
        self.gpu_process.SetSelectedThread(lane_1)
        frame1 = lane_1.GetFrameAtIndex(1)
        var = frame1.FindVariable("middle_struct")
        self.assertTrue(var.IsValid(), "Should find 'middle_struct' in frame 1")
        self.assertEqual(
            var.GetChildMemberWithName("x").GetValueAsSigned(),
            3,
            "lane 1: middle_struct.x = 3",
        )
        self.assertEqual(
            var.GetChildMemberWithName("y").GetValueAsSigned(),
            7,
            "lane 1: middle_struct.y = 7",
        )

    # -----------------------------------------------------------------
    # Kernel frame (frame 2) variables
    # -----------------------------------------------------------------

    def test_kernel_frame_variables(self):
        """Verify variables in the kernel (outermost) frame."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]
        self.gpu_process.SetSelectedThread(thread)

        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame2.IsValid(), "Kernel frame should be valid")
        self.assertIn("stack_unwind_kernel", frame2.GetFunctionName())

        var = frame2.FindVariable("tid")
        self.assertTrue(var.IsValid(), "Should find 'tid' in kernel frame")
        self.assertEqual(var.GetValueAsSigned(), 0, "lane 0: tid = 0")

    # -----------------------------------------------------------------
    # frame variable command (--flat) for middle frame
    # -----------------------------------------------------------------

    def test_frame_variable_command_middle(self):
        """Verify 'frame variable' command works for the middle frame."""
        gpu_threads = self.run_to_leaf_breakpoint()
        thread = gpu_threads[0]
        self.gpu_process.SetSelectedThread(thread)

        # Select the middle frame and run frame variable
        self.runCmd(f"frame select 1")
        self.expect(
            "frame variable --flat --show-all-children middle_scalar",
            substrs=["middle_scalar = 5"],
        )
        self.expect(
            "frame variable --flat --show-all-children middle_array",
            substrs=[
                "middle_array[0] = 100",
                "middle_array[1] = 101",
                "middle_array[2] = 102",
                "middle_array[3] = 103",
            ],
        )
        self.expect(
            "frame variable --flat --show-all-children middle_struct",
            substrs=["middle_struct.x = 0", "middle_struct.y = 0"],
        )
