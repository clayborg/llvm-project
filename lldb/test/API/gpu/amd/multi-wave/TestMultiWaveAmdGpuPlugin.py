"""
Basic tests for the AMDGPU plugin with a multi-wave kernel.
"""

from collections import Counter, defaultdict

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SHADOW_THREAD_NAME = "AMD Native Shadow Thread"

# multi-wave.hip launches dim3(2, 2, 2) blocks of dim3(5, 4, 6) threads. These
# are properties of the kernel launch, so they hold for any wave size.
NUM_BLOCKS = 2 * 2 * 2
THREADS_PER_BLOCK = 5 * 4 * 6
TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK


class BasicAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_breakpoint(self):
        # GPU breakpoint should get hit by at least one thread.
        source = "multi-wave.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        return gpu_threads_at_bp

    def get_wave_layout(self):
        """Return (wave_size, waves_per_block, lanes_per_block) for this kernel.

        A workgroup is filled with full waves and whatever is left over ends up
        in a shorter final wave, so on wave64 a block of 120 work items is a
        wave of 64 lanes and one of 56, while on wave32 it is three of 32 and
        one of 24.
        """
        wave_size = self.get_wave_size()
        self.assertIn(wave_size, (32, 64), f"implausible wave size {wave_size}")
        waves_per_block = -(-THREADS_PER_BLOCK // wave_size)
        last_wave_lanes = THREADS_PER_BLOCK - (waves_per_block - 1) * wave_size
        lanes_per_block = [wave_size] * (waves_per_block - 1) + [last_wave_lanes]
        return wave_size, waves_per_block, lanes_per_block

    def test_num_threads(self):
        """Test that we get the expected number of threads."""
        self.build()

        gpu_threads_at_breakpoint = self.run_to_breakpoint()

        # We launch 960 total threads (8 blocks * 120 threads per block).
        gpu_threads = self.gpu_process.threads
        self.assertEqual(len(gpu_threads), TOTAL_THREADS)

        # But not all waves may reach the breakpoint at the same time. So here
        # we check that we have at least one wave's worth of threads stopped at
        # the breakpoint. The smallest wave is the partial one at the end of a
        # block, which is 56 threads on wave64 and 24 on wave32.
        _, _, lanes_per_block = self.get_wave_layout()
        self.assertGreaterEqual(len(gpu_threads_at_breakpoint), min(lanes_per_block))

    def test_step_over_from_multi_wave_breakpoint(self):
        """Test stepping when multiple GPU waves are stopped at a breakpoint."""
        self.build()

        source = "multi-wave.hip"
        gpu_threads_at_breakpoint = self.run_to_breakpoint()
        self.step_over_gpu_thread(
            gpu_threads_at_breakpoint[0], line_number(source, "// GPU STEP OVER")
        )

    def test_wave_and_lane_ids_across_waves(self):
        """Test how lane and SIMD ids behave when a kernel spans many waves."""
        self.build()

        self.run_to_breakpoint()
        gpu_threads = self.gpu_process.threads
        self.assertEqual(len(gpu_threads), TOTAL_THREADS)

        # Work out the wave layout this kernel must produce on this hardware.
        wave_size, waves_per_block, lanes_per_block = self.get_wave_layout()

        lanes_by_wave = defaultdict(list)
        for thread in gpu_threads:
            lanes_by_wave[thread.GetSIMD()].append(thread.GetLaneID())

        # Every work item belongs to a real wave, and each block is spread over
        # one wave per wave_size work items.
        self.assertNotIn(lldb.LLDB_INVALID_SIMD_ID, lanes_by_wave)
        self.assertEqual(len(lanes_by_wave), NUM_BLOCKS * waves_per_block)

        # Lane numbering restarts at zero in each wave, so a lane id only
        # identifies a thread when paired with its SIMD id. In particular lane 0
        # appears once per wave rather than once per process.
        for simd_id, lanes in lanes_by_wave.items():
            self.assertEqual(
                sorted(lanes),
                list(range(len(lanes))),
                f"wave {simd_id} should own lanes [0, {len(lanes)})",
            )
        self.assertEqual(
            sum(1 for thread in gpu_threads if thread.GetLaneID() == 0),
            NUM_BLOCKS * waves_per_block,
        )

        # A workgroup that is not a multiple of the wave size ends in a partial
        # wave. On wave64 that means each of the 8 blocks contributes a wave of
        # 64 lanes and one of 56.
        self.assertEqual(
            Counter(len(lanes) for lanes in lanes_by_wave.values()),
            Counter(lanes_per_block * NUM_BLOCKS),
        )

        # Thread ids stay unique across waves even though lane ids do not.
        self.assertEqual(
            len({thread.GetThreadID() for thread in gpu_threads}), TOTAL_THREADS
        )
