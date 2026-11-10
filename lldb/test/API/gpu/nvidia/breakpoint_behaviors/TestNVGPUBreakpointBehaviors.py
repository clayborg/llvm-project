"""
Test setting and editing GPU breakpoints as a user does.
TestNVGPUBreakpointStops tests what happens at a hit.
breakpoint_behaviors.py describes the program.
"""

import re
from typing import Dict

import lldb
from breakpoint_behaviors import (
    EXE,
    LOOP_SUM,
    SOURCE,
    WARP,
    BreakpointBehaviorsTestCase,
)
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number


class TestNVGPUBreakpointBehaviors(BreakpointBehaviorsTestCase):
    def test_ignore_count_skips_iterations(self):
        """An ignore count of N on a breakpoint in the loop skips the first N
        hits. The thread stops in iteration N, and the hit count includes
        the ignored hits, as it does on the CPU."""
        breakpoint, thread = self.run_to_loop(ignore_count=5)
        self.assert_iteration(thread, 5)
        self.assertEqual(
            breakpoint.GetHitCount(), 6, "hit count, ignored hits included"
        )
        self.assertEqual(breakpoint.GetIgnoreCount(), 0, "ignore count left")
        self.continue_gpu_until_breakpoint(breakpoint)
        self.assert_iteration(self.one_thread_at(breakpoint), 6)
        self.assertEqual(breakpoint.GetHitCount(), 7, "hit count after the next stop")

    def test_ignore_count_set_before_the_run_is_copied(self):
        """`breakpoint set -f -l -i 5` before `run`: the copy of the breakpoint
        that lldb makes in the GPU target keeps the ignore count, so the GPU
        stops in iteration 5."""
        target = self.make_shared_target(EXE)
        cpu_breakpoint = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in loop")
        )
        cpu_breakpoint.SetIgnoreCount(5)
        breakpoint = self.run_to_gpu_breakpoint(target, cpu_breakpoint)
        self.assertEqual(
            breakpoint.GetIgnoreCount(), 0, "ignore count left on the copy"
        )
        self.assert_iteration(self.one_thread_at(breakpoint), 5)
        self.assertEqual(breakpoint.GetHitCount(), 6, "hit count of the copy")

    def test_breakpoint_set_after_the_cubin_loads(self):
        """A function breakpoint set while the GPU is stopped, so after its
        cubin is loaded, resolves at once, and the next continue stops in
        the function, called from the loop."""
        self.run_to_loop()
        square = self.gpu_target.BreakpointCreateByName("square")
        self.assertEqual(
            square.GetNumLocations(),
            1,
            "locations of a breakpoint on a loaded function",
        )
        self.continue_gpu_until_breakpoint(square)
        thread = self.one_thread_at(square)
        self.assert_frame_at(
            thread.frame[0], "square(int)", SOURCE, "// breakpoint in square"
        )
        self.assert_int_variable(thread.frame[0], "value", 0)
        self.assert_frame_at(
            thread.frame[1],
            "loop_kernel(int*, int)",
            SOURCE,
            "// breakpoint in loop",
        )
        self.assertEqual(square.GetHitCount(), 1, f"hit count of {square}")

    def test_breakpoint_on_an_inlined_function(self):
        """A breakpoint on a `__forceinline__` device function, by name or by
        a line of its body, has a location in each kernel it is inlined
        into. At a stop there, frame 0 is the inlined function and frame 1
        the kernel at the call."""
        breakpoint, _ = self.run_to_loop()
        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        inlined = self.gpu_target.BreakpointCreateByName("accumulate")
        self.assertEqual(inlined.GetNumLocations(), 2, f"locations of {inlined}")
        copies = {
            location.GetAddress().GetFunction().GetName(): location
            for location in inlined.locations
        }
        self.assertEqual(
            sorted(copies),
            ["loop_kernel(int*, int)", "reduce(int*)"],
            "kernels with a location of the inlined function",
        )
        by_line = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in accumulate")
        )
        self.assertEqual(
            {location.GetLoadAddress() for location in by_line.locations},
            {location.GetLoadAddress() for location in copies.values()},
            "addresses of the breakpoint by line",
        )
        self.gpu_target.BreakpointDelete(by_line.GetID())

        self.continue_gpu_until_breakpoint(inlined)
        thread = self.one_thread_at(inlined)
        frame = thread.frame[0]
        self.assertTrue(frame.IsInlined(), "frame 0 is an inlined frame")
        self.assert_frame_at(
            frame, "accumulate(int, int)", SOURCE, "// breakpoint in accumulate"
        )
        self.assert_int_variable(frame, "value", LOOP_SUM)
        self.assert_frame_at(
            thread.frame[1],
            "loop_kernel(int*, int)",
            SOURCE,
            "// breakpoint after loop",
        )
        self.assertEqual(
            self.reported_breakpoints(thread),
            {inlined.GetID(): copies["loop_kernel(int*, int)"].GetID()},
            "the location the stop reports",
        )

        # With loop_kernel's location disabled, the second launch runs through
        # and the next stop is reduce's first iteration, where every thread's
        # `total` is still its own index.
        copies["loop_kernel(int*, int)"].SetEnabled(False)
        self.continue_gpu_until_breakpoint(inlined)

        def check(thread: lldb.SBThread) -> None:
            frame = thread.frame[0]
            self.assertTrue(frame.IsInlined(), f"frame 0 of {thread.GetName()}")
            self.assert_frame_at(
                frame, "accumulate(int, int)", SOURCE, "// breakpoint in accumulate"
            )
            self.assert_int_variable(frame, "total", self.thread_coords(thread)[1][0])
            self.assert_frame_at(
                thread.frame[1], "reduce(int*)", SOURCE, "// reduce add"
            )

        self.for_each_thread_at_breakpoint(inlined, WARP, check)
        self.assertEqual(inlined.GetHitCount(), 1 + WARP, "hit count")
        self.gpu_target.BreakpointDelete(inlined.GetID())
        self.finish()

    @expectedFailureAll(
        "a second run of the program makes a new GPU target with copies of the "
        "CPU target's breakpoints only, so a breakpoint set on the first run's "
        "GPU target is left in that target and never hit again"
    )
    def test_breakpoint_set_on_the_gpu_target_stops_the_next_run(self):
        """A breakpoint set on the GPU target during a run, as `breakpoint
        set` does after a GPU stop, where the GPU target is selected, stops
        the next run of the program too, as a CPU breakpoint does. After
        `process kill`, `target select` of the CPU target and `run`, the
        GPU stops in the loop again."""
        self.run_to_loop()
        first_gpu_target = self.gpu_target
        self.assertSuccess(self.cpu_process.Kill(), "kill the CPU process")
        # In sync mode the launch returns at start()'s CPU breakpoint, where
        # the new run's GPU target exists.
        self.setAsync(False)
        process = self.cpu_target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertState(process.GetState(), lldb.eStateStopped)
        # lldb makes a GPU target for each run and leaves the first run's
        # behind, where the helpers would find it. Delete it while it is still
        # there: once lldb drops or reuses it, this step passes and the
        # assertion below decides the test.
        if self.dbg.GetNumTargets() == 3:
            self.dbg.DeleteTarget(first_gpu_target)
        # Drop the reference too. An expected failure keeps the method's locals
        # alive past tearDown, and a live SBTarget keeps the target's modules.
        del first_gpu_target
        self.continue_cpu_and_wait_for_gpu_to_stop()
        threads = lldbutil.get_stopped_threads(
            self.gpu_process, lldb.eStopReasonBreakpoint
        )
        self.assertEqual(len(threads), 1, "threads at a breakpoint in the second run")
        self.assert_iteration(threads[0], 0)
        self.gpu_target.DeleteAllBreakpoints()
        self.finish()

    def test_function_breakpoint_spellings(self):
        """The ways a user names a kernel in `breakpoint set` resolve to one
        address, its first statement: the plain name, the name with its
        parameter list, the mangled name, a regular expression, and the
        file and line of the first statement. The next launch reports them
        all at its stop."""
        breakpoint, _ = self.run_to_loop()
        gpu_target = self.gpu_target
        first_statement = line_number(SOURCE, "// first statement")
        spellings: Dict[str, lldb.SBBreakpoint] = {
            "loop_kernel": gpu_target.BreakpointCreateByName("loop_kernel"),
            "loop_kernel(int*, int)": gpu_target.BreakpointCreateByName(
                "loop_kernel(int*, int)"
            ),
            "_Z11loop_kernelPii": gpu_target.BreakpointCreateByName(
                "_Z11loop_kernelPii"
            ),
            "regex ^loop_kernel": gpu_target.BreakpointCreateByRegex("^loop_kernel"),
            f"{SOURCE}:{first_statement}": gpu_target.BreakpointCreateByLocation(
                SOURCE, first_statement
            ),
        }
        addresses: Dict[str, int] = {}
        for spelling, candidate in spellings.items():
            self.assertEqual(
                candidate.GetNumLocations(), 1, f"locations of `{spelling}`"
            )
            address = candidate.GetLocationAtIndex(0).GetAddress()
            self.assertEqual(
                address.GetFunction().GetName(),
                "loop_kernel(int*, int)",
                f"function of `{spelling}`",
            )
            addresses[spelling] = address.GetLoadAddress(gpu_target)
        self.assertEqual(len(set(addresses.values())), 1, f"addresses: {addresses}")

        # Without the loop breakpoint, the next stop is the second launch's.
        gpu_target.BreakpointDelete(breakpoint.GetID())
        self.continue_gpu_until_breakpoint(spellings["loop_kernel"])
        thread = self.one_thread_at(spellings["loop_kernel"])
        self.assertEqual(
            set(self.reported_breakpoints(thread)),
            {candidate.GetID() for candidate in spellings.values()},
            "breakpoints the stop reports",
        )
        for spelling, candidate in spellings.items():
            self.assertEqual(candidate.GetHitCount(), 1, f"hit count of `{spelling}`")

    def test_deleted_breakpoint_stops_nothing(self):
        """Deleting the breakpoint at a stop removes it from the GPU. The rest
        of the launch and the next launch run through."""
        breakpoint, _ = self.run_to_loop()
        self.assertTrue(
            self.gpu_target.BreakpointDelete(breakpoint.GetID()),
            "deleting the breakpoint",
        )
        self.finish()

    def test_disabled_breakpoint_is_skipped_until_enabled(self):
        """Disabled at a stop, the breakpoint lets the rest of the launch
        run, and the CPU reaches its line between the launches; enabled
        again there, it stops the next launch in its first iteration."""
        breakpoint, _ = self.run_to_loop()
        breakpoint.SetEnabled(False)
        between_launches = self.cpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// between launches")
        )
        self.setAsync(True)
        self.assertSuccess(self.gpu_process.Continue(), "continue GPU process")
        self.wait_for_stop(self.cpu_process)
        cpu_threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.cpu_process, between_launches
        )
        self.assertEqual(len(cpu_threads), 1, "CPU threads between the launches")
        self.assert_int_variable(cpu_threads[0].frame[0], "launches", 1)
        self.assertEqual(
            self.pull_state(self.gpu_process),
            lldb.eStateRunning,
            "the GPU process while its breakpoint is disabled",
        )
        self.assertEqual(breakpoint.GetHitCount(), 1, "hit count while disabled")

        breakpoint.SetEnabled(True)
        self.cpu_target.BreakpointDelete(between_launches.GetID())
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_iteration(self.one_thread_at(breakpoint), 0)
        self.assertEqual(breakpoint.GetHitCount(), 2, "hit count in the second launch")
        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.finish()

    def test_continue_sends_no_breakpoint_packets(self):
        """lldb leaves the GPU's breakpoints alone when it resumes the GPU from
        one. Between two stops, it sends the GPU a continue and no breakpoint
        or step packet, where it would send a CPU process the remove,
        single-step and reinsert. The debugger API itself steps a warp that
        stopped at a breakpoint past it on resume."""
        breakpoint, _ = self.run_to_loop()
        before = self.packet_history(self.gpu_process)
        self.continue_gpu_until_breakpoint(breakpoint)
        self.assert_iteration(self.one_thread_at(breakpoint), 1)
        after = self.packet_history(self.gpu_process)
        self.assertEqual(
            after[: len(before)], before, "the packet history must not have wrapped"
        )
        sent = [line for line in after[len(before) :] if "send packet:" in line]
        self.assertTrue(
            any("$c#" in line or "$vCont;c" in line for line in sent),
            f"a continue packet among:\n{sent}",
        )
        # A step is a bare `s` or `S<signal>` packet, or an `s` or `S` action
        # in a `vCont` packet, which lldb puts after the `c` actions.
        step_or_breakpoint = re.compile(r"\$([zZ]0,|[sS][0-9a-f]*#|vCont[^#]*;[sS])")
        mutations = [line for line in sent if step_or_breakpoint.search(line)]
        self.assertEqual(mutations, [], "breakpoint and step packets between two stops")
