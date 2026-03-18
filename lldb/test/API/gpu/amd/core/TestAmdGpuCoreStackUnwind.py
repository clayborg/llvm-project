"""
Stack unwinding and caller-frame variable tests from AMD GPU core files.

Uses the AmdGpuCoreTestBase infrastructure to automatically generate a GPU
core file via rocgdb, then loads it in LLDB to verify:
  1. The backtrace contains the expected three frames (leaf, middle, kernel).
  2. Each frame reports the correct function name.
  3. Local variables in non-leaf frames (middle_function) are readable and
     have the expected values via DWARF-based stack unwinding.

stack_unwind.hip launches 1 block / 32 threads with a three-deep call chain:

    stack_unwind_kernel  ->  middle_function  ->  leaf_function

The GPU breakpoint is inside leaf_function.

Wave 0, default lane values (threadIdx.x = 0, blockIdx.x = 0):
  leaf_function:
    leaf_local = 0 * 100 = 0
    scalar_arg = 5
  middle_function:
    middle_scalar = 0 * 10 + 5 = 5
    middle_array = {100, 101, 102, 103}
    middle_struct = {x=0, y=0}
  stack_unwind_kernel:
    tid = 0
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.amdgpu_core_testbase import AmdGpuCoreTestBase


class TestAmdGpuCoreStackUnwind(AmdGpuCoreTestBase):
    """Verify GPU stack unwinding and caller-frame variables from core files."""

    HIP_SOURCE = "stack_unwind.hip"
    GPU_BREAKPOINT_PATTERN = "// GPU BREAKPOINT"

    def build(self, **kwargs):
        """Override build to compile stack_unwind.hip instead of the Makefile default."""
        dictionary = kwargs.pop("dictionary", None) or {}
        dictionary["HIP_SOURCES"] = self.HIP_SOURCE
        super().build(dictionary=dictionary, **kwargs)

    # -----------------------------------------------------------------
    # Helper to check a variable in a specific frame via CLI
    # -----------------------------------------------------------------

    def check_frame_variable(self, gpu_process, wave_index, frame_index, name,
                             expected_values):
        """Select a specific frame and check a variable via 'frame variable'."""
        if isinstance(expected_values, str):
            expected_values = [expected_values]
        wave = self.get_wave(gpu_process, wave_index)
        gpu_process.SetSelectedThread(wave)
        self.runCmd(f"frame select {frame_index}")
        self.expect(
            f"frame variable --flat --show-all-children {name}",
            substrs=expected_values,
        )

    # -----------------------------------------------------------------
    # Backtrace structure
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_backtrace_depth(self):
        """The GPU wave should have at least 3 frames."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)

        num_frames = wave.GetNumFrames()
        self.assertGreaterEqual(
            num_frames,
            3,
            f"Expected >= 3 frames in backtrace, got {num_frames}",
        )

    @skipIfRemote
    def test_backtrace_function_names(self):
        """Each frame should report the correct function name."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)

        frame0 = wave.GetFrameAtIndex(0)
        frame1 = wave.GetFrameAtIndex(1)
        frame2 = wave.GetFrameAtIndex(2)

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

    @skipIfRemote
    def test_frame_pc_values(self):
        """Each frame should have a non-zero PC."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)

        for i in range(min(wave.GetNumFrames(), 3)):
            frame = wave.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid(), f"Frame {i} should be valid")
            pc = frame.GetPC()
            self.assertNotEqual(pc, 0, f"Frame {i} PC should be non-zero")

    # -----------------------------------------------------------------
    # Leaf frame (frame 0) variables
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_leaf_frame_local(self):
        """Verify leaf_local in frame 0: tid=0 -> leaf_local = 0."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)

        frame = wave.GetFrameAtIndex(0)
        var = frame.FindVariable("leaf_local")
        self.assertTrue(var.IsValid(), "Should find 'leaf_local'")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'leaf_local' failed: {var.GetError().GetCString()}",
        )
        self.assertEqual(var.GetValueAsSigned(), 0, "leaf_local = 0")

    @skipIfRemote
    def test_leaf_frame_scalar_arg(self):
        """Verify scalar_arg in frame 0: middle_scalar = 5."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)

        frame = wave.GetFrameAtIndex(0)
        var = frame.FindVariable("scalar_arg")
        self.assertTrue(var.IsValid(), "Should find 'scalar_arg'")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'scalar_arg' failed: {var.GetError().GetCString()}",
        )
        self.assertEqual(var.GetValueAsSigned(), 5, "scalar_arg = 5")

    # -----------------------------------------------------------------
    # Middle frame (frame 1) variables — core unwinding test
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_middle_frame_scalar(self):
        """Verify middle_scalar in frame 1: tid=0 -> 5."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)

        frame1 = wave.GetFrameAtIndex(1)
        self.assertTrue(frame1.IsValid(), "Middle frame should be valid")
        self.assertIn("middle_function", frame1.GetFunctionName())

        var = frame1.FindVariable("middle_scalar")
        self.assertTrue(var.IsValid(), "Should find 'middle_scalar' in frame 1")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'middle_scalar' failed: {var.GetError().GetCString()}",
        )
        self.assertEqual(
            var.GetValueAsSigned(), 5, "middle_scalar = 0*10+5 = 5"
        )

    @skipIfRemote
    def test_middle_frame_array(self):
        """Verify middle_array in frame 1: tid=0 -> {100, 101, 102, 103}."""
        gpu_target, gpu_process = self.load_core()
        self.check_frame_variable(
            gpu_process,
            0,
            1,
            "middle_array",
            [
                "middle_array[0] = 100",
                "middle_array[1] = 101",
                "middle_array[2] = 102",
                "middle_array[3] = 103",
            ],
        )

    @skipIfRemote
    def test_middle_frame_struct(self):
        """Verify middle_struct in frame 1: tid=0 -> {x=0, y=0}."""
        gpu_target, gpu_process = self.load_core()
        self.check_frame_variable(
            gpu_process,
            0,
            1,
            "middle_struct",
            ["middle_struct.x = 0", "middle_struct.y = 0"],
        )

    # -----------------------------------------------------------------
    # Kernel frame (frame 2) variables
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_kernel_frame_tid(self):
        """Verify tid in kernel frame: default lane 0 -> tid = 0."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)

        frame2 = wave.GetFrameAtIndex(2)
        self.assertTrue(frame2.IsValid(), "Kernel frame should be valid")
        self.assertIn("stack_unwind_kernel", frame2.GetFunctionName())

        var = frame2.FindVariable("tid")
        self.assertTrue(var.IsValid(), "Should find 'tid' in kernel frame")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'tid' failed: {var.GetError().GetCString()}",
        )
        self.assertEqual(var.GetValueAsSigned(), 0, "tid = 0")

    # -----------------------------------------------------------------
    # frame variable command for middle frame
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_frame_variable_command_middle(self):
        """Verify 'frame variable' CLI works for the middle frame."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)

        # Select the middle frame
        self.runCmd("frame select 1")
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
