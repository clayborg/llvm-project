"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): running it,
breakpoints in its template kernel by line and by name, set before the run,
and the hits of a breakpoint in each tile and launch.
"""

import lldb
from lldbsuite.test.lldbtest import line_number

from matrixmul import EXE, LAUNCH_16, LAUNCH_32, SOURCE, MatrixMulTestCase


class TestNVGPUSampleMatrixMul(MatrixMulTestCase):
    def test_runs_to_exit(self):
        """matrixMul with the sample's own matrices runs to exit under LLDB,
        and its own check of C passes."""
        self.launch_and_wait_for_gpu_target(self.make_shared_target(EXE))
        self.assert_exit_status(0)
        self.assert_stdout_contains(
            "MatrixA(320,320), MatrixB(640,320)", "Result = PASS"
        )

    def test_breakpoint_on_template_kernel_line(self):
        """A breakpoint on a line of the template kernel, set before the
        program runs, has a location in each instantiation the cubin holds,
        and stops every thread of the launch at the one the program runs,
        MatrixMulCUDA<32>."""
        marker = "// gpu breakpoint at tile load"
        target = self.make_target()
        breakpoint = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, marker)
        )
        gpu_breakpoint = self.run_to_gpu_breakpoint(target, breakpoint)
        locations = {}
        for location in gpu_breakpoint.locations:
            locations[location.GetAddress().GetFunction().GetName()] = location
        self.assertEqual(
            sorted(locations), sorted([LAUNCH_32.kernel, LAUNCH_16.kernel])
        )
        launched = locations[LAUNCH_32.kernel].GetID()

        stopped = self.assert_gpu_stop_at_breakpoint(gpu_breakpoint)
        self.assert_frame_at(stopped[0].frame[0], LAUNCH_32.kernel, SOURCE, marker)

        # A breakpoint stop's reason data is the breakpoint ID, then the
        # location ID. Every thread must be at the <32> location; the frame
        # check above shows which line that location is.
        def check(thread: lldb.SBThread) -> None:
            self.assertEqual(
                thread.GetStopReasonDataAtIndex(1),
                launched,
                f"breakpoint location {thread.GetName()} stopped at",
            )

        self.assertEqual(
            self.for_each_thread_at_breakpoint(
                gpu_breakpoint, len(LAUNCH_32.threads), check
            ),
            LAUNCH_32.threads,
        )
        self.finish()

    def test_breakpoint_by_template_name(self):
        """A breakpoint on the name of the kernel's instantiation,
        MatrixMulCUDA<32>, set before the program runs, stops every thread of
        the launch at the start of the kernel, and not the CPU in the launch
        code that nvcc generates under the kernel's name."""
        target = self.make_target()
        breakpoint = target.BreakpointCreateByName("MatrixMulCUDA<32>")
        gpu_breakpoint = self.run_to_gpu_breakpoint(target, breakpoint)
        self.assertEqual(gpu_breakpoint.GetNumLocations(), 1)
        address = gpu_breakpoint.GetLocationAtIndex(0).GetLoadAddress()
        frame = self.assert_gpu_stop_at_breakpoint(gpu_breakpoint)[0].frame[0]
        self.assertEqual(frame.GetFunctionName(), LAUNCH_32.kernel)
        # The first line after the kernel's prologue, which depends on the code
        # nvcc generates: check only that it comes before the loop.
        line = frame.GetLineEntry().GetLine()
        self.assertGreater(line, line_number(SOURCE, "__global__ void MatrixMulCUDA"))
        self.assertLess(line, line_number(SOURCE, "// gpu breakpoint at tile load"))

        def check(thread: lldb.SBThread) -> None:
            self.assertEqual(thread.frame[0].GetPC(), address, thread.GetName())

        self.assertEqual(
            self.for_each_thread_at_breakpoint(
                gpu_breakpoint, len(LAUNCH_32.threads), check
            ),
            LAUNCH_32.threads,
        )
        self.finish()

    def test_breakpoint_rehits_in_each_tile_and_launch(self):
        """A breakpoint in the kernel's loop over tiles stops every thread in
        each iteration, and again in the next launch of the kernel. lldb counts
        each thread's stop as a hit of the breakpoint."""
        breakpoint = self.run_to_gpu_marker("// gpu breakpoint at tile load")
        # Both tiles of the first launch, then the first of the next launch.
        for stop in range(LAUNCH_32.tiles + 1):
            if stop:
                self.continue_gpu_until_breakpoint(breakpoint)
            tile = stop % LAUNCH_32.tiles

            def check(thread: lldb.SBThread, tile: int = tile) -> None:
                coords = self.thread_coords(thread)
                expected = LAUNCH_32.tile_locals(coords, tile)["a"]
                self.assert_int_variable(thread.frame[0], "a", expected)

            self.for_each_thread_at_breakpoint(
                breakpoint, len(LAUNCH_32.threads), check
            )
            self.assertEqual(
                breakpoint.GetHitCount(),
                (stop + 1) * len(LAUNCH_32.threads),
                f"hit count after stop {stop}",
            )
        self.finish()
