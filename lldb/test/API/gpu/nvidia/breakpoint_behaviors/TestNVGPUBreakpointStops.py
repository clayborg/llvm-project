"""
Test what happens when a GPU breakpoint is hit.
TestNVGPUBreakpointBehaviors tests setting and editing breakpoints.
breakpoint_behaviors.py describes the program.
"""

import lldb
from breakpoint_behaviors import (
    ITERATIONS,
    LAUNCHES,
    SOURCE,
    WARP,
    WIDTHS,
    FINAL_OUTPUT,
    BreakpointBehaviorsTestCase,
)
from lldbsuite.test.decorators import expectedFailureAll


class TestNVGPUBreakpointStops(BreakpointBehaviorsTestCase):
    def test_breakpoint_at_the_current_pc_is_not_hit_until_the_next_pass(self):
        """A breakpoint set at the stopped thread's PC does not stop the
        thread again where it is. The hit there is consumed, and the next
        stop is the next iteration, where both breakpoints report."""
        breakpoint, thread = self.run_to_loop()
        at_pc = self.gpu_target.BreakpointCreateBySBAddress(
            thread.frame[0].GetPCAddress()
        )
        self.assertEqual(
            at_pc.GetNumLocations(), 1, "locations of the breakpoint at the PC"
        )
        self.continue_gpu_until_breakpoint(at_pc)
        thread = self.one_thread_at(at_pc)
        self.assert_iteration(thread, 1)
        self.assertEqual(
            set(self.reported_breakpoints(thread)),
            {breakpoint.GetID(), at_pc.GetID()},
            "breakpoints the stop reports",
        )
        self.assertEqual(breakpoint.GetHitCount(), 2, f"hit count of {breakpoint}")
        self.assertEqual(at_pc.GetHitCount(), 1, f"hit count of {at_pc}")

    def test_auto_continue_breakpoint_counts_hits(self):
        """An auto-continue breakpoint (`breakpoint set -G true`) counts its
        hits without stopping the GPU. The program runs to its end with a hit
        for every iteration of both launches."""
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// breakpoint in loop")
        breakpoint.SetAutoContinue(True)
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue(), "continue CPU process")
        self.assert_exit_status(0)
        self.assertEqual(breakpoint.GetHitCount(), ITERATIONS * LAUNCHES, "hit count")
        self.assert_stdout_contains(FINAL_OUTPUT)

    def test_one_shot_breakpoint_is_deleted_after_its_hit(self):
        """A one-shot breakpoint (`breakpoint set -o true`) stops the GPU once
        and is gone. The program then runs to its end."""
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// breakpoint in loop")
        breakpoint.SetOneShot(True)
        breakpoint_id = breakpoint.GetID()
        self.continue_cpu_and_wait_for_gpu_to_stop()
        # The breakpoint is gone at the stop, so the stop is matched by its ID.
        threads = self.assert_gpu_stop_at_breakpoint(breakpoint_id)
        self.assertEqual(len(threads), 1, "threads stopped at the one-shot breakpoint")
        self.assert_iteration(threads[0], 0)
        self.assertFalse(
            self.gpu_target.FindBreakpointByID(breakpoint_id).IsValid(),
            "the one-shot breakpoint after its hit",
        )
        self.finish()

    def test_python_callback_decides_the_stop(self):
        """A Python callback on the breakpoint runs at each hit with the
        thread's frame and decides whether the GPU stops, as `breakpoint
        command add -s python` does. Here it stops the thread in one
        iteration of each launch and resumes the others."""
        stop_iteration = 3
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// breakpoint in loop")
        self.assertSuccess(
            breakpoint.SetScriptCallbackBody(
                f"return frame.FindVariable('i').GetValueAsSigned() == {stop_iteration}"
            ),
            "setting the callback",
        )
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_iteration(self.one_thread_at(breakpoint), stop_iteration)
        # The hits before the stop iteration count too.
        self.assertEqual(
            breakpoint.GetHitCount(), stop_iteration + 1, "hit count at the first stop"
        )
        self.continue_gpu_until_breakpoint(breakpoint)
        self.assert_iteration(self.one_thread_at(breakpoint), stop_iteration)
        self.assertEqual(
            breakpoint.GetHitCount(),
            ITERATIONS + stop_iteration + 1,
            "hit count at the second stop",
        )
        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.finish()

    def test_thread_specific_breakpoint_stops_one_thread(self):
        """`breakpoint modify -t <tid>` limits a breakpoint to one GPU thread.
        The next stops report that thread alone, in the following
        iterations, and the other threads' hits are not counted."""
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// partner read")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.for_each_thread_at_breakpoint(
            breakpoint, WARP, self.width_check(WIDTHS[0])
        )
        chosen = self.select_gpu_thread(threadidx=[7])
        chosen_coords = self.thread_coords(chosen)
        breakpoint.SetThreadID(chosen.GetThreadID())
        followed = WIDTHS[1:3]
        for width in followed:
            threads = self.continue_gpu_until_breakpoint(breakpoint)
            self.assertEqual(
                [self.thread_coords(thread) for thread in threads],
                [chosen_coords],
                f"threads at the breakpoint with width {width}",
            )
            self.width_check(width)(threads[0])
            others = [
                thread.GetName()
                for thread in self.gpu_process.threads
                if thread.GetStopReason() != lldb.eStopReasonNone
                and self.thread_coords(thread) != chosen_coords
            ]
            self.assertEqual(others, [], "other threads with a stop reason")
        self.assertEqual(breakpoint.GetHitCount(), WARP + len(followed), "hit count")
        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.finish()

    @expectedFailureAll(
        "lldb checks a breakpoint's ignore count once per thread, so at the warp's "
        "first stop a count of 1 is used up by one of the 32 threads and the warp "
        "stops there with every thread reporting the breakpoint"
    )
    def test_ignore_count_on_a_warp_skips_stops(self):
        """An ignore count counts the times the breakpoint's line is reached,
        not the threads that reach it. reduce's one warp of 32 threads
        executes the partner-read line together in each iteration of its
        loop, so an ignore count of 1 skips the whole first iteration. The
        first stop is the second iteration, width 4, with all 32 threads at
        the breakpoint."""
        self.start()
        breakpoint = self.gpu_breakpoint(SOURCE, "// partner read")
        breakpoint.SetIgnoreCount(1)
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.for_each_thread_at_breakpoint(
            breakpoint, WARP, self.width_check(WIDTHS[1])
        )
