"""
Test GPU variable reading from AMD GPU core files.

variables.hip launches 1 block / 32 threads.
  - Wave32 GPU: 1 wave (lanes 0-31), default lane = 0
  - Wave64 GPU: 1 wave (lanes 0-31 active), default lane = 0

Wave 0, default lane values (threadIdx.x = 0, blockIdx.x = 0):
  tid = 0
  thread_scalar = 0*2 = 0
  thread_array = {0, 1, 2, 3}
  thread_struct = {x=0, y=0}
  shared_scalar = 0+1000 = 1000
  shared_array = {0, 1, 2, ..., 31}
  shared_struct = {x=100, y=200}
  global_scalar_value = 100
  global_array_value = g_array[0%4] = 10
  global_struct_value = {x=50, y=60}
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.amdgpu_core_testbase import AmdGpuCoreTestBase

SHARED_ARRAY_SIZE = 32


class TestAmdGpuCoreVariables(AmdGpuCoreTestBase):
    """Verify GPU variable reading from core."""

    # Override base class hooks
    HIP_SOURCE = "variables.hip"
    GPU_BREAKPOINT_PATTERN = "// GPU BREAKPOINT"

    @skipIfRemote
    def test_core_loads_with_gpu_waves(self):
        """Verify core loads and has exactly one GPU wave."""
        gpu_target, gpu_process = self.load_core()
        self.assertEqual(
            gpu_process.GetNumThreads(),
            1,
            "variables.hip launches 1 block / 32 threads = 1 wave",
        )

    @skipIfRemote
    def test_read_variable_smoke(self):
        """Verify DoReadMemory works by reading tid from wave 0."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)
        frame = wave.GetFrameAtIndex(0)

        var = frame.FindVariable("tid")
        self.assertTrue(var.IsValid(), "Should find 'tid' variable")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'tid' failed: {var.GetError().GetCString()}",
        )
        self.assertEqual(
            var.GetValueAsSigned(),
            0,
            "Wave 0, default lane 0: tid should be 0",
        )

    # -----------------------------------------------------------------
    # Wave 0: private (per-lane) variables
    #
    # Private variables are per-lane. At the wave level the debugger
    # shows the default lane (lane 0). threadIdx.x = 0 for lane 0.
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_wave0_private_scalar(self):
        """Wave 0 private scalar: thread_scalar = tid*2 = 0."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "thread_scalar",
            "thread_scalar = 0",
        )

    @skipIfRemote
    def test_wave0_private_array(self):
        """Wave 0 private array: thread_array[i] = tid+i = {0,1,2,3}."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "thread_array",
            [
                "thread_array[0] = 0",
                "thread_array[1] = 1",
                "thread_array[2] = 2",
                "thread_array[3] = 3",
            ],
        )

    @skipIfRemote
    def test_wave0_private_struct(self):
        """Wave 0 private struct: thread_struct = {tid*10, tid*20} = {0,0}."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "thread_struct",
            ["thread_struct.x = 0", "thread_struct.y = 0"],
        )

    # -----------------------------------------------------------------
    # Wave 0: shared (LDS) variables
    #
    # Shared variables are per-block, visible to all waves in the block.
    # blockIdx.x = 0 for the single block launch.
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_wave0_shared_scalar(self):
        """Wave 0 shared scalar: shared_scalar = blockIdx.x+1000 = 1000."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "shared_scalar",
            "shared_scalar = 1000",
        )

    @skipIfRemote
    def test_wave0_shared_array(self):
        """Wave 0 shared array: shared_array[i] = i."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "shared_array",
            [f"shared_array[{i}] = {i}" for i in range(SHARED_ARRAY_SIZE)],
        )

    @skipIfRemote
    def test_wave0_shared_struct(self):
        """Wave 0 shared struct: shared_struct = {100, 200}."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "shared_struct",
            ["shared_struct.x = 100", "shared_struct.y = 200"],
        )

    # -----------------------------------------------------------------
    # Wave 0: local copies of global variables
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_wave0_global_copy_scalar(self):
        """Wave 0 global copy: global_scalar_value = g_scalar = 100."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "global_scalar_value",
            "global_scalar_value = 100",
        )

    @skipIfRemote
    def test_wave0_global_copy_array_element(self):
        """Wave 0 global copy: global_array_value = g_array[0%4] = 10."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "global_array_value",
            "global_array_value = 10",
        )

    @skipIfRemote
    def test_wave0_global_copy_struct(self):
        """Wave 0 global copy: global_struct_value = {50, 60}."""
        gpu_target, gpu_process = self.load_core()
        self.check_wave_variable(
            gpu_process,
            0,
            "global_struct_value",
            ["global_struct_value.x = 50", "global_struct_value.y = 60"],
        )

    # -----------------------------------------------------------------
    # SBValue API path (exercises DoReadMemory differently)
    # -----------------------------------------------------------------

    @skipIfRemote
    def test_wave0_read_array_data(self):
        """Verify raw byte read of array via SBValue for wave 0."""
        gpu_target, gpu_process = self.load_core()
        wave = self.get_wave(gpu_process, 0)
        gpu_process.SetSelectedThread(wave)
        frame = wave.GetFrameAtIndex(0)

        var = frame.FindVariable("thread_array")
        self.assertTrue(var.IsValid())
        self.assertTrue(var.GetError().Success(), var.GetError().GetCString())

        data = var.GetData()
        self.assertTrue(data.IsValid())
        self.assertGreater(data.GetByteSize(), 0)
