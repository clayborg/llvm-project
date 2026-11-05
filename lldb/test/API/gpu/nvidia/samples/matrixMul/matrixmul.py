"""
What the tests of the cuda-samples program matrixMul share. The program is
copied in matrixMul.cu. Its kernel, the template MatrixMulCUDA<BLOCK_SIZE>,
computes one tile of C = A * B in each block of BLOCK_SIZE x BLOCK_SIZE
threads. For each pair of tiles along a row of A and a column of B, every
thread loads one element of each into the shared arrays As and Bs, waits at a
barrier, adds the products of a row of As and a column of Bs to its Csub, and
waits at a second barrier. Then it stores Csub in C. The program launches the
kernel once, then 300 times more, and prints "Result = PASS" if every element
of C is right.

The tests launch the kernel in one of two ways. Both are a 2x2 grid of blocks,
all resident at once on different SMs, with two tiles for each thread, so
every test has blocks other than (0, 0):

- LAUNCH_32 is the sample's own blocks of 32x32 threads, on 64x64 matrices:
  4,096 threads in MatrixMulCUDA<32>. The tests that check every thread or set
  breakpoints before the run use it, and so does the conditional breakpoint
  test that passes, so that lldb evaluates a condition in 4,096 threads at
  once.
- LAUNCH_16 is blocks of 16x16 threads, on 32x32 matrices: 1,024 threads in
  MatrixMulCUDA<16>, the sample's other instantiation. The tests that follow
  one thread use it, since a stop of all 4,096 threads takes about a second on
  an idle machine and several times that while the other GPU tests run. The
  sample takes no block size, so the tests set its block_size variable from
  lldb before the launch, as a user can, and the copy stays as it is. A warp
  of a 16-wide block spans two rows, so a thread's lane is not its
  threadIdx.x, and blockDim is not the warp size.

A is all 1.0 and B all 0.01, so the tests compute each value they read from the
thread's coordinates.

The tests do what a user does to debug the sample: break in the kernel by
line and by name, pick a thread by its coordinates, read its locals, the
shared tiles and the matrices, evaluate expressions, set conditions, and step
through a tile. They are in seven files, one for each workflow, so that no
one file makes the GPU tests slower.
"""

import functools
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase, ThreadCoords

SOURCE = "matrixMul.cu"
EXE = "matrixMul"

# The block size the sample launches with, and the threads in a warp.
SAMPLE_BLOCK = 32
WARP_SIZE = 32


@dataclass(frozen=True)
class Launch:
    """A launch of the kernel the tests ask the program for: a 2x2 grid of
    square blocks, on square matrices two blocks wide."""

    # The side of a block, the kernel's BLOCK_SIZE.
    block: int

    @property
    def width(self) -> int:
        """The side of the matrices A, B and C."""
        return 2 * self.block

    @property
    def grid(self) -> int:
        """Blocks in x and in y."""
        return self.width // self.block

    @property
    def tiles(self) -> int:
        """Iterations of the kernel's loop over tiles."""
        return self.width // self.block

    @property
    def args(self) -> List[str]:
        """The program's arguments for the matrices."""
        width = self.width
        return [f"-wA={width}", f"-hA={width}", f"-wB={width}", f"-hB={width}"]

    @property
    def kernel(self) -> str:
        """The instantiation of the kernel that runs."""
        return f"void MatrixMulCUDA<{self.block}>(float*, float*, float*, int, int)"

    @property
    def threads(self) -> Set[ThreadCoords]:
        """Every thread of the launch, as (blockIdx, threadIdx)."""
        return {
            ((bx, by, 0), (tx, ty, 0))
            for bx in range(self.grid)
            for by in range(self.grid)
            for tx in range(self.block)
            for ty in range(self.block)
        }

    def tile_locals(self, coords: ThreadCoords, tile: int) -> Dict[str, int]:
        """The kernel's integer locals in the thread at `coords`, in iteration
        `tile` of its loop over tiles."""
        (bx, by, _), (tx, ty, _) = coords
        a_begin = self.width * self.block * by
        b_begin = self.block * bx
        return {
            "bx": bx,
            "by": by,
            "tx": tx,
            "ty": ty,
            "aBegin": a_begin,
            "aEnd": a_begin + self.width - 1,
            "aStep": self.block,
            "bBegin": b_begin,
            "bStep": self.block * self.width,
            "a": a_begin + tile * self.block,
            "b": b_begin + tile * self.block * self.width,
        }

    def warp(self, coords: ThreadCoords) -> Tuple[Tuple[int, int, int], int]:
        """The block of the thread at `coords`, and its warp in the block: a
        block's warps are its threads 32 at a time, row by row."""
        block, (tx, ty, _) = coords
        return block, (ty * self.block + tx) // WARP_SIZE

    def lane(self, coords: ThreadCoords) -> int:
        """The lane of the thread at `coords`: its place in its warp."""
        _, (tx, ty, _) = coords
        return (ty * self.block + tx) % WARP_SIZE


LAUNCH_32 = Launch(block=SAMPLE_BLOCK)
LAUNCH_16 = Launch(block=16)

# The values the host fills A and B with. A is 1.0 so every product is exact,
# and csub() matches the GPU whether or not nvcc fuses the multiply and the add.
VAL_A = 1.0
VAL_B = 0.01

# The thread that the tests following one thread stop, in either launch, and a
# breakpoint condition on the kernel's locals that holds only for that thread.
THREAD = ((1, 1, 0), (3, 2, 0))
THREAD_CONDITION = "bx == 1 && by == 1 && tx == 3 && ty == 2"

# How long a step may take. Every warp runs during a step, so another warp
# can reach the end of the line first. lldb then steps the thread by
# instruction to finish, and the GPU does not stop for an instruction step
# yet, so about one step in five hangs. A step that works stops within a
# second, so the stepping tests wait far less than the default.
STEP_TIMEOUT = 10


def f32(value: float) -> float:
    """`value` rounded to a float, as the program stores it."""
    return struct.unpack("f", struct.pack("f", value))[0]


# The tests check Csub in every thread, so compute each value once.
@functools.lru_cache(maxsize=None)
def csub(products: int) -> float:
    """Csub once the kernel has added `products` products of an element of A
    and one of B to it, in float arithmetic."""
    total = 0.0
    for _ in range(products):
        total = f32(total + f32(VAL_A) * f32(VAL_B))
    return total


class MatrixMulTestCase(NVGPUTestCaseBase):
    """The helpers that depend on matrixMul. Each test file subclasses
    this."""

    def make_target(self, launch: Launch = LAUNCH_32) -> lldb.SBTarget:
        """A target for the program that runs it on `launch`'s matrices.
        run_to_launch then gives the kernel `launch`'s blocks, and finish
        checks the program's output for them."""
        self.kernel_launch = launch
        return self.make_shared_target(EXE, args=launch.args)

    def run_to_launch(self, target: Optional[lldb.SBTarget] = None) -> None:
        """Run the program to its `// cpu breakpoint before launch` line, with
        `target` from make_target, or a new target for LAUNCH_32. CUDA is
        initialized there and no kernel has run yet, so the GPU target exists
        and takes breakpoints for the first launch. The line is the first use
        of block_size, which a launch with blocks other than the sample's sets
        there, as a user can from lldb."""
        _, _, thread, _ = self.run_to_cpu_line(
            SOURCE,
            "// cpu breakpoint before launch",
            target=target or self.make_target(),
        )
        block = self.kernel_launch.block
        if block != SAMPLE_BLOCK:
            error = lldb.SBError()
            thread.frame[0].FindVariable("block_size").SetValueFromCString(
                str(block), error
            )
            self.assertSuccess(error, f"setting block_size to {block}")

    def run_to_gpu_marker(
        self,
        marker: str,
        condition: Optional[str] = None,
        launch: Launch = LAUNCH_32,
    ) -> lldb.SBBreakpoint:
        """Run the program, with `launch`, to the GPU's first stop at a
        breakpoint on the line marked `marker`, with `condition` if one is
        given, and return the breakpoint. The breakpoint is set on the GPU
        target, which lldb creates when the program initializes CUDA, so the
        CPU never stops at it."""
        self.run_to_launch(self.make_target(launch))
        breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, marker)
        )
        if condition:
            breakpoint.SetCondition(condition)
        self.continue_cpu_and_wait_for_gpu_to_stop()
        return breakpoint

    def assert_stopped_threads(
        self, stopped: List[lldb.SBThread], expected: ThreadCoords, what: str
    ) -> None:
        """`stopped`, the threads at a breakpoint, is only the thread at
        `expected`. `what` names the stop in a failure message."""
        self.assertEqual(len(stopped), 1, f"threads stopped at {what}")
        self.assertEqual(self.thread_coords(stopped[0]), expected, what)

    def run_to_thread(
        self,
        marker: str,
        condition: str = THREAD_CONDITION,
        launch: Launch = LAUNCH_16,
    ) -> Tuple[lldb.SBBreakpoint, lldb.SBThread]:
        """Run the program, with `launch`, to the GPU's first stop at a
        breakpoint on the line marked `marker` with `condition`, and return
        the breakpoint and THREAD. The condition must stop THREAD and no other
        thread, and lldb must evaluate it without an error: when it cannot,
        it stops every thread and reports an error for each."""
        errors = self.collect_errors()
        breakpoint = self.run_to_gpu_marker(marker, condition, launch)
        self.assertEqual(errors(), [], f"errors evaluating {condition!r}")
        stopped = self.assert_gpu_stop_at_breakpoint(breakpoint)
        self.assert_stopped_threads(stopped, THREAD, f"the breakpoint at {marker!r}")
        return breakpoint, stopped[0]

    def step(self, thread: lldb.SBThread, marker: str) -> None:
        """Step `thread` over its source line, as `next` does, to the line
        marked `marker`, waiting STEP_TIMEOUT seconds at most."""
        self.step_over_gpu_thread(
            thread, line_number(SOURCE, marker), timeout_seconds=STEP_TIMEOUT
        )

    def check_output(self) -> None:
        """The program multiplied the matrices of the launch the test asked
        for and found C right."""
        launch = getattr(self, "kernel_launch", None)
        self.assertIsNotNone(launch, "check_output needs a target from make_target")
        width = launch.width
        self.assert_stdout_contains(
            f"MatrixA({width},{width}), MatrixB({width},{width})", "Result = PASS"
        )

    def finish(self) -> None:
        """Let the program run to its end, and check that it computed C
        right."""
        self.continue_gpu_to_exit()
        self.check_output()
