"""
Tests on the cuda-samples program matrixMul (see matrixmul.py), on LAUNCH_16:
which threads lldb reports in one warp, and what stepping one GPU thread over a
source line does to the other threads. As with cuda-gdb's `next`, only the
stepped thread's warp should move. While every warp runs during a step, the
step itself can also hang, and waits STEP_TIMEOUT seconds at most (see
matrixmul.py).
"""

from typing import Tuple

import lldb
from lldbsuite.test.decorators import expectedFailureAll

from matrixmul import LAUNCH_16, THREAD, MatrixMulTestCase


class TestNVGPUSampleMatrixMulStepWarps(MatrixMulTestCase):
    def step_one_thread_of_all(self) -> Tuple[lldb.SBBreakpoint, lldb.SBThread]:
        """Run to the first stop of every thread at the load of A's element,
        step THREAD over that line, and return the breakpoint and the
        thread."""
        breakpoint = self.run_to_gpu_marker(
            "// gpu breakpoint at tile load", launch=LAUNCH_16
        )
        self.for_each_thread_at_breakpoint(breakpoint, len(LAUNCH_16.threads))
        thread = self.select_gpu_thread(THREAD[0][:2], THREAD[1][:2])
        # So that a thread that runs during the step cannot end it.
        breakpoint.SetEnabled(False)
        self.step(thread, "// load Bs tile")
        return breakpoint, thread

    def test_threads_of_each_warp(self):
        """lldb reports where each thread runs: on which SM, in which of its
        warps and in which lane of that warp. The 32 threads of each warp of
        the block (see Launch.warp) report one warp of an SM, which no other
        thread reports, and their lanes follow their order in the block: a
        thread in an odd row of a 16-wide block is not in the lane of its
        threadIdx.x."""
        breakpoint = self.run_to_gpu_marker(
            "// gpu breakpoint at tile load", launch=LAUNCH_16
        )
        # Once every thread has stopped at the breakpoint, every block is on
        # the GPU, and none has finished, since no thread stopped there twice.
        self.for_each_thread_at_breakpoint(breakpoint, len(LAUNCH_16.threads))
        threads = set()
        # Each warp of the launch, as (blockIdx, warp in the block), and the
        # warps of an SM that its threads report, as (device, SM, warp).
        hardware_warps = {}
        for thread in self.gpu_process.threads:
            coords = self.thread_coords(thread)
            threads.add(coords)
            device, sm, warp, lane = self.thread_physical_coords(thread)
            self.assertEqual(
                lane, LAUNCH_16.lane(coords), f"lane of {thread.GetName()}"
            )
            hardware_warps.setdefault(LAUNCH_16.warp(coords), set()).add(
                (device, sm, warp)
            )
        self.assertEqual(threads, LAUNCH_16.threads, "threads of the launch")
        warps_of = {}
        for (block, index), found in hardware_warps.items():
            self.assertEqual(
                len(found),
                1,
                f"warp {index} of block {block} runs in {sorted(found)}",
            )
            warps_of.setdefault(found.pop(), []).append((block, index))
        self.assertEqual(
            {found: warps for found, warps in warps_of.items() if len(warps) > 1},
            {},
            "warps of an SM that run more than one warp of the launch",
        )
        self.finish()

    @expectedFailureAll("stepping one GPU thread over a source line resumes every warp")
    def test_step_keeps_other_warps_stopped(self):
        """Stepping one thread over a source line moves its warp, the 32
        threads that run with it, and no other: the other warps stay at the
        breakpoint, as with cuda-gdb's `next`."""
        breakpoint, _ = self.step_one_thread_of_all()
        # The line has a location in each instantiation of the template, and
        # the threads stopped at the one the program launches.
        addresses = {location.GetLoadAddress() for location in breakpoint.locations}
        warp = LAUNCH_16.warp(THREAD)
        moved = []
        for other in self.gpu_process.threads:
            if LAUNCH_16.warp(self.thread_coords(other)) == warp:
                continue
            if other.frame[0].GetPC() not in addresses:
                moved.append(other.GetName())
        self.assertEqual(
            len(moved),
            0,
            f"threads of other warps that ran during the step: {moved[:4]} and more",
        )
        self.finish()

    @expectedFailureAll(
        "threads that ran during a step keep the breakpoint stop reason of "
        "the stop before it"
    )
    def test_moved_threads_lose_breakpoint_stop_reason(self):
        """After one thread steps over a source line, no thread that is no
        longer at the breakpoint reports that it stopped there."""
        breakpoint, _ = self.step_one_thread_of_all()
        addresses = {location.GetLoadAddress() for location in breakpoint.locations}
        stale = [
            other.GetName()
            for other in self.gpu_process.threads
            if other.GetStopReason() == lldb.eStopReasonBreakpoint
            and other.frame[0].GetPC() not in addresses
        ]
        self.assertEqual(
            len(stale),
            0,
            f"threads away from the breakpoint that report it: {stale[:4]} and more",
        )
        self.finish()
