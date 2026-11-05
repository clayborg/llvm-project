"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): threads by
their two-dimensional coordinates, and what frame variable reads at a stop in
the kernel: locals, the shared tiles, and the matrices through the kernel's
pointer parameters.
"""

import struct

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.tools.gpu.nvgpu_testcase import ThreadCoords

from matrixmul import (
    LAUNCH_16,
    LAUNCH_32,
    THREAD,
    VAL_A,
    VAL_B,
    MatrixMulTestCase,
    csub,
    f32,
)


class TestNVGPUSampleMatrixMulVariables(MatrixMulTestCase):
    def test_two_dimensional_coordinates_and_locals(self):
        """At the first stop in the kernel, each thread's CUDA built-in
        registers hold its coordinates in the 2x2 grid of 32x32 blocks, and
        its locals and the kernel's integer parameters have the values the
        kernel computed from them."""
        breakpoint = self.run_to_gpu_marker("// gpu breakpoint at tile load")
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].frame[0]
        self.assertEqual(
            self.read_vec3_register(frame, "blockDim"),
            [LAUNCH_32.block, LAUNCH_32.block, 1],
        )
        self.assertEqual(
            self.read_vec3_register(frame, "gridDim"),
            [LAUNCH_32.grid, LAUNCH_32.grid, 1],
        )

        def check(thread: lldb.SBThread) -> None:
            coords = self.thread_coords(thread)
            frame = thread.frame[0]
            registers = tuple(
                tuple(self.read_vec3_register(frame, name))
                for name in ("blockIdx", "threadIdx")
            )
            self.assertEqual(registers, coords, "blockIdx and threadIdx")
            for name, value in LAUNCH_32.tile_locals(coords, 0).items():
                self.assert_int_variable(frame, name, value)
            self.assert_int_variable(frame, "wA", LAUNCH_32.width)
            self.assert_int_variable(frame, "wB", LAUNCH_32.width)
            self.assert_float_variable(frame, "Csub", 0.0)

        self.assertEqual(
            self.for_each_thread_at_breakpoint(
                breakpoint, len(LAUNCH_32.threads), check
            ),
            LAUNCH_32.threads,
        )
        self.finish()

    def test_select_thread_by_two_dimensional_index(self):
        """thread select --blockidx --threadidx selects the thread with those
        CUDA coordinates, in two dimensions, and its frame then shows that
        thread's locals. Coordinates that no thread has select nothing."""
        breakpoint = self.run_to_gpu_marker("// gpu breakpoint at tile load")
        self.assert_gpu_stop_at_breakpoint(breakpoint)

        def check(thread: lldb.SBThread, coords: ThreadCoords) -> None:
            self.assertEqual(self.thread_coords(thread), coords)
            frame = thread.GetSelectedFrame()
            for name in ("bx", "by", "tx", "ty"):
                self.assert_int_variable(
                    frame, name, LAUNCH_32.tile_locals(coords, 0)[name]
                )

        for block, index in (((1, 1), (3, 2)), ((0, 1), (31, 0)), ((1, 0), (0, 31))):
            check(
                self.select_gpu_thread(block, index),
                ((*block, 0), (*index, 0)),
            )
        # Only a thread index: the block is the selected thread's.
        check(self.select_gpu_thread(threadidx=(7, 7)), ((1, 0, 0), (7, 7, 0)))
        # The coordinates by name, which the shell needs quoted.
        self.runCmd('thread select --blockidx "(x=0 y=1)" --threadidx "(x=5 y=6)"')
        check(self.gpu_process.GetSelectedThread(), ((0, 1, 0), (5, 6, 0)))

        self.expect(
            f"thread select --blockidx {LAUNCH_32.grid},0 --threadidx 0,0",
            error=True,
            substrs=["no GPU thread found matching the specified coordinates"],
        )
        self.finish()

    def check_shared_tiles(self, thread: lldb.SBThread) -> None:
        """The shared tiles As and Bs of `thread`'s block hold the values of A
        and B, both as the arrays the kernel declares and in shared memory at
        their addresses."""
        frame = thread.frame[0]
        count = LAUNCH_32.block * LAUNCH_32.block
        for name, value in (("As", VAL_A), ("Bs", VAL_B)):
            array = frame.GetValueForVariablePath(name)
            self.assertSuccess(array.GetError(), f"reading {name}")
            self.assertTrue(
                array.GetLocation().startswith("shared#"),
                f"{name} is in shared memory: {array.GetLocation()}",
            )
            error = lldb.SBError()
            data = array.GetData().ReadRawData(error, 0, 4 * count)
            self.assertSuccess(error, f"the bytes of {name}")
            self.assertEqual(
                struct.unpack(f"<{count}f", data), (f32(value),) * count, name
            )
            memory = self.read_gpu_memory(
                array.GetLoadAddress(), "shared", 4 * count, thread
            )
            self.assertEqual(memory, data, f"shared memory at {name}'s address")

    def test_shared_tiles_at_second_barrier(self):
        """At the barrier after the loop of multiply-adds, in each iteration
        over tiles, every thread's Csub holds the products so far, and lldb
        reads the block's shared tiles As and Bs both as the arrays the kernel
        declares and as raw shared memory."""
        breakpoint = self.run_to_gpu_marker("// second barrier")
        for tile in range(LAUNCH_32.tiles):
            if tile:
                self.continue_gpu_until_breakpoint(breakpoint)
            blocks_checked = set()

            def check(thread: lldb.SBThread, tile: int = tile) -> None:
                block, _ = self.thread_coords(thread)
                expected = csub(LAUNCH_32.block * (tile + 1))
                self.assert_float_variable(thread.frame[0], "Csub", expected)
                if block not in blocks_checked:
                    blocks_checked.add(block)
                    self.check_shared_tiles(thread)

            self.for_each_thread_at_breakpoint(
                breakpoint, len(LAUNCH_32.threads), check
            )
        self.finish()

    @expectedFailureAll(
        "frame variable reads what a kernel's pointer parameter points to in "
        "the parameter's own space, not in generic memory"
    )
    def test_matrix_elements_through_pointer_parameters(self):
        """frame variable reads the elements of A and B that a thread is about
        to load, through the kernel's pointer parameters A and B, as memory
        read at the pointers' values does."""
        _, thread = self.run_to_thread("// gpu breakpoint at tile load")
        frame = thread.frame[0]
        (_, (tx, ty, _)) = THREAD
        indices = LAUNCH_16.tile_locals(THREAD, 0)
        width = LAUNCH_16.width
        for name, index, value in (
            ("A", indices["a"] + width * ty + tx, VAL_A),
            ("B", indices["b"] + width * ty + tx, VAL_B),
        ):
            # The pointer points to global memory, where the element is.
            pointer = frame.GetValueForVariablePath(name)
            self.assertSuccess(pointer.GetError(), f"reading {name}")
            memory = self.read_gpu_memory(
                pointer.GetValueAsUnsigned() + 4 * index, "global", 4
            )
            self.assertEqual(struct.unpack("<f", memory)[0], f32(value), name)
            self.assert_float_variable(frame, f"{name}[{index}]", f32(value))
        self.finish()
