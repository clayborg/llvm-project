"""
Tests on the cuda-samples program simpleCooperativeGroups, copied in
simpleCooperativeGroups.cu. Its kernel runs one block of 64 threads and calls
the device function sumReduction twice: once for the whole block, then once
for each of four 16-thread tiles. sumReduction sums the ranks of its group in
shared memory, with a barrier between each store and load. The program checks
nothing itself and always exits 0, so the tests check what it prints, and
check the values lldb reads at its stops against the same computation.

The tests do what a user does to debug the sample: stop in sumReduction, pick
a thread, read its variables and the shared memory, and step through the
loop.
"""

import re
from typing import Callable, List, Optional, Tuple

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "simpleCooperativeGroups.cu"
EXE = "simpleCooperativeGroups"
BLOCK_SIZE = 64
TILE_SIZE = 16

BLOCK_SUM = re.compile(
    r"Sum of all ranks 0\.\.(\d+) in threadBlockGroup is (\d+) \(expected (\d+)\)"
)
TILE_SUM = re.compile(
    r"Sum of all ranks 0\.\.15 in this tiledPartition16 group is (\d+) "
    r"\(expected (\d+)\)"
)


def reduce(size: int) -> Tuple[List[int], List[int]]:
    """Run sumReduction's loop for a group of `size` threads, each with its
    rank as input. Return `val` of each lane when the loop ends, and the
    contents of the group's workspace, which keeps what the last iteration
    stored."""
    val = list(range(size))
    workspace = list(val)
    i = size // 2
    while i > 0:
        workspace = list(val)
        val = [v + workspace[lane + i] if lane < i else v for lane, v in enumerate(val)]
        i //= 2
    return val, workspace


def loop_iterations(size: int) -> List[Tuple[int, List[int]]]:
    """Run sumReduction's loop for a group of `size` threads, each with its
    rank as input. Return `i` and the `val` of each lane at the start of each
    iteration."""
    val = list(range(size))
    iterations = []
    i = size // 2
    while i > 0:
        iterations.append((i, val))
        val = [v + val[lane + i] if lane < i else v for lane, v in enumerate(val)]
        i //= 2
    return iterations


class TestNVGPUSampleSimpleCooperativeGroups(NVGPUTestCaseBase):
    def run_to_marker(
        self, marker: str, condition: Optional[str] = None
    ) -> lldb.SBBreakpoint:
        """Run the program to a breakpoint on the line marked `marker`, with
        `condition` if one is given, and return the GPU target's breakpoint.
        The program makes no CUDA call before its kernel launch, so the
        breakpoint is set before the run (see run_to_gpu_line)."""
        return self.run_to_gpu_line(
            SOURCE, marker, target=self.make_shared_target(EXE), condition=condition
        )

    def check_output(self, stdout: str) -> None:
        """The program printed the sum of the ranks of the block, and of each
        of the four tiles."""
        self.assertEqual(
            BLOCK_SUM.findall(stdout), [("63", "2016", "2016")], f"stdout:\n{stdout}"
        )
        self.assertEqual(
            TILE_SUM.findall(stdout), [("120", "120")] * 4, f"stdout:\n{stdout}"
        )

    def finish(self) -> None:
        """Let the program run to its end and check what it printed."""
        self.continue_gpu_to_exit()
        self.check_output(self.read_stdout())

    def check_every_thread(
        self,
        breakpoint: lldb.SBBreakpoint,
        call: str,
        check: Callable[[lldb.SBThread, int], None],
    ) -> None:
        """Run `check(thread, rank)` on every thread of the block as it stops
        at `breakpoint`, in sumReduction called from the line marked `call`,
        where `rank` is the thread's rank in the block."""

        def check_thread(thread: lldb.SBThread) -> None:
            self.assert_frame_at(thread.frame[1], "cgkernel()", SOURCE, call)
            check(thread, self.thread_coords(thread)[1][0])

        self.for_each_thread_at_breakpoint(breakpoint, BLOCK_SIZE, check_thread)

    def test_runs_to_exit(self):
        """simpleCooperativeGroups runs to exit under LLDB and prints the sums
        of the ranks of the block and of each tile."""
        self.launch_and_wait_for_gpu_target(self.make_shared_target(EXE))
        self.assert_exit_status(0)
        self.check_output(self.read_stdout())

    def test_breakpoint_in_device_function(self):
        """A breakpoint on the name of the device function sumReduction, set
        before the program runs, stops every thread of the block in both of
        the kernel's calls to it, with the kernel as the caller at the line of
        each call."""
        target = self.make_shared_target(EXE)
        # nvcc also leaves a copy of sumReduction in the CPU program, which
        # nothing calls, so the CPU target's location never stops the CPU.
        breakpoint = self.run_to_gpu_breakpoint(
            target, target.BreakpointCreateByName("sumReduction")
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)

        def check(thread: lldb.SBThread, rank: int) -> None:
            self.assertEqual(thread.GetNumFrames(), 2, "sumReduction and the kernel")
            frame = thread.frame[0]
            # cooperative_groups defines its types in a versioned inline
            # namespace, __v1 in CUDA 13.
            self.assertRegex(
                frame.GetFunctionName(),
                r"^sumReduction\(cooperative_groups::(\w+::)?thread_group, "
                r"int\*, int\)$",
            )
            # The first line after the function's prologue, which depends on
            # the code nvcc generates: check only that it is in the function.
            line = frame.GetLineEntry().GetLine()
            self.assertGreater(line, line_number(SOURCE, "__device__ int sumReduction"))
            self.assertLess(
                line, line_number(SOURCE, "// gpu breakpoint after barrier")
            )
            self.assertEqual(
                self.read_vec3_register(frame, "blockDim"), [BLOCK_SIZE, 1, 1]
            )
            self.assertEqual(self.read_vec3_register(frame, "gridDim"), [1, 1, 1])

        self.check_every_thread(breakpoint, "// reduce the block", check)
        self.continue_gpu_until_breakpoint(breakpoint)
        self.check_every_thread(breakpoint, "// reduce each tile", check)
        self.finish()

    def test_shared_memory_at_barrier(self):
        """After the first barrier in sumReduction, every thread of the block
        has stored its rank in the shared array workspace, and lldb reads the
        array from the block's shared memory."""
        breakpoint = self.run_to_marker("// gpu breakpoint after barrier")
        # The first stop on the line is in the first iteration of the loop: no
        # thread passes the barrier at the end of an iteration until every
        # thread of the block has reached it.
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]
        self.assert_frame_at(
            thread.frame[1], "cgkernel()", SOURCE, "// reduce the block"
        )

        # workspace is declared `extern __shared__ int workspace[]`, so its
        # type has no length and lldb shows no elements: read each by index.
        frame = thread.frame[0]
        self.assertEqual(
            [
                self.read_int_variable(frame, f"workspace[{i}]")
                for i in range(BLOCK_SIZE)
            ],
            list(range(BLOCK_SIZE)),
        )
        self.finish()

    def test_variables_in_device_function_and_caller(self):
        """When sumReduction's loop ends, each thread's lane and val, the
        group's workspace through the pointer x, and the locals of the kernel
        that called it, read in the caller's frame, have the values the code
        computed, in both calls."""
        breakpoint = self.run_to_marker("// gpu breakpoint after reduction")

        def check_group(frame: lldb.SBFrame, lane: int, size: int) -> None:
            val, workspace = reduce(size)
            self.assert_int_variable(frame, "lane", lane)
            self.assert_int_variable(frame, "val", val[lane])
            # Read the group's workspace once per group, from its first lane.
            if lane == 0:
                self.assertEqual(
                    [self.read_int_variable(frame, f"x[{i}]") for i in range(size)],
                    workspace,
                )

        def check_block(thread: lldb.SBThread, rank: int) -> None:
            check_group(thread.frame[0], rank, BLOCK_SIZE)
            caller = thread.frame[1]
            self.assert_int_variable(caller, "threadBlockGroupSize", BLOCK_SIZE)
            self.assert_int_variable(caller, "input", rank)
            self.assert_int_variable(caller, "expectedOutput", sum(range(BLOCK_SIZE)))

        def check_tile(thread: lldb.SBThread, rank: int) -> None:
            lane = rank % TILE_SIZE
            check_group(thread.frame[0], lane, TILE_SIZE)
            caller = thread.frame[1]
            self.assert_int_variable(caller, "input", lane)
            self.assert_int_variable(caller, "workspaceOffset", rank - lane)
            self.assert_int_variable(caller, "expectedOutput", sum(range(TILE_SIZE)))

        self.check_every_thread(breakpoint, "// reduce the block", check_block)
        self.continue_gpu_until_breakpoint(breakpoint)
        self.check_every_thread(breakpoint, "// reduce each tile", check_tile)
        self.finish()

    def test_conditional_breakpoint_stops_one_thread(self):
        """A breakpoint with a condition on lane, set before the program runs,
        stops only the thread for which the condition holds, once in each
        iteration of the loop, and nowhere else: lane 17 exists only in the
        call for the whole block, since a tile's lanes are 0 to 15."""
        lane = 17
        breakpoint = self.run_to_marker(
            "// gpu breakpoint after barrier", condition=f"lane == {lane}"
        )
        expected = [(i, val[lane]) for i, val in loop_iterations(BLOCK_SIZE)]
        seen = []
        stopped = self.assert_gpu_stop_at_breakpoint(breakpoint)
        while True:
            self.assertEqual(len(stopped), 1, "threads stopped at the breakpoint")
            self.assertEqual(self.thread_coords(stopped[0])[1], (lane, 0, 0))
            frame = stopped[0].frame[0]
            seen.append(
                (
                    self.read_int_variable(frame, "i"),
                    self.read_int_variable(frame, "val"),
                )
            )
            if len(seen) == len(expected):
                break
            stopped = self.continue_gpu_until_breakpoint(breakpoint)
        self.assertEqual(seen, expected, "(i, val) at each stop")
        self.assertEqual(breakpoint.GetHitCount(), len(expected))

        # The breakpoint stays in place: no other thread stops at it.
        self.continue_gpu_to_exit(keep_breakpoints=True)
        self.check_output(self.read_stdout())

    def test_select_thread_by_cuda_index(self):
        """thread select --threadidx selects the thread with that CUDA thread
        index, in either warp of the block, and its frame then shows that
        thread's variables."""
        breakpoint = self.run_to_marker("// gpu breakpoint after barrier")
        self.assert_gpu_stop_at_breakpoint(breakpoint)
        for index in (17, 40):
            thread = self.select_gpu_thread(threadidx=(index,))
            self.assertEqual(self.thread_coords(thread)[1], (index, 0, 0))
            frame = thread.GetSelectedFrame()
            self.assert_int_variable(frame, "lane", index)
            self.assert_int_variable(frame, "val", index)
        self.finish()

    @expectedFailureAll("stepping over a source line in device code runs on")
    def test_step_over_reduction_loop(self):
        """Stepping over source lines, as `next` does, moves one thread
        through an iteration of sumReduction's loop: to the addition, to the
        barrier at the end of the iteration, and over that barrier, which the
        block's other warp has to reach too, back to the loop's header."""
        lane = 1
        breakpoint = self.run_to_marker(
            "// gpu breakpoint after barrier", condition=f"lane == {lane}"
        )
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]
        for marker in (
            "// add partner's value",
            "// barrier at end of iteration",
            "// reduction loop",
        ):
            self.step_over_gpu_thread(thread, line_number(SOURCE, marker))
            self.assert_int_variable(thread.frame[0], "lane", lane)
        self.finish()

    @expectedFailureAll(
        "expressions read GPU memory in the default address space, not the "
        "shared or generic space"
    )
    def test_expressions_read_shared_memory(self):
        """Expressions, which `p` and `expr` evaluate, read the shared array
        workspace and the group's workspace through the pointer x, as frame
        variable does."""
        breakpoint = self.run_to_marker("// gpu breakpoint after barrier")
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].frame[0]
        # Each thread stored its rank before the first barrier.
        for expression in ("workspace[5]", "x[5]"):
            value = self.evaluate_expression(frame, expression)
            self.assertEqual(value.GetValueAsSigned(), 5, expression)
        self.finish()

    @expectedFailureAll(
        "a pointer stored in a struct in local memory is read through in local "
        "memory, where the struct is, not in the pointer's generic space"
    )
    def test_pointer_member_of_thread_block(self):
        """The kernel's thread_block, read in the caller's frame, holds a
        generic pointer to the block's scratch memory, and lldb reads what it
        points to as memory read -p generic does."""
        breakpoint = self.run_to_marker("// gpu breakpoint after barrier")
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]
        caller = thread.frame[1]
        pointer = caller.GetValueForVariablePath("threadBlockGroup.tile_memory")
        self.assertSuccess(pointer.GetError())

        data = self.read_gpu_memory(pointer.GetValueAsUnsigned(), "generic", 4, thread)
        barrier = caller.GetValueForVariablePath(
            "threadBlockGroup.tile_memory->barriers[0]"
        )
        self.assertSuccess(barrier.GetError(), "reading through the pointer")
        self.assertEqual(barrier.GetValueAsUnsigned(), int.from_bytes(data, "little"))
        self.assertTrue(
            pointer.GetValue().startswith("generic#"),
            f"the pointer is shown in its own space: {pointer.GetValue()}",
        )
        self.finish()

    @expectedFailureAll(
        "a synchronous launch returns at the CPU's target-sync stop, and the "
        "GPU process takes its stop only when an event loop pulls it"
    )
    def test_synchronous_launch_stops_at_gpu_breakpoint(self):
        """In synchronous mode, which a script that handles no events runs
        in, a launch that stops at a breakpoint in device code returns with
        the GPU process stopped there, and the CPU stopped with it."""
        if not self.cpu_stops_with_gpu():
            self.skipTest(
                "the CPU does not stop when the GPU does, so a synchronous launch "
                "returns only when the program exits"
            )
        target = self.make_shared_target(EXE)
        breakpoint = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// gpu breakpoint after barrier")
        )
        self.setAsync(False)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped, "CPU")
        self.assertState(
            self.gpu_process.GetState(),
            lldb.eStateStopped,
            "GPU, when the launch returns",
        )
        self.assert_gpu_stop_at_breakpoint(self.gpu_breakpoint_copy(breakpoint))
