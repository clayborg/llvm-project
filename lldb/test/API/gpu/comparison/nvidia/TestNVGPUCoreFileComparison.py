"""
Compares LLDB and cuda-gdb on NVGPU core files (.nvcudmp).

Both debuggers load the same core (LLDB as an nvgpu-core target, cuda-gdb via
"target cudacore") and their views are compared. Two cores share the same
load/compare plumbing (_load_both):

- test_compare_artificial_core: an artificial core built by NVGPUCoreBuilder
  (no live GPU/CUDA/build), embedding the comparison_artificial.cubin fixture.
  Compares thread count, coordinates, and the faulting thread's registers.
- test_compare_live_core: a live core from comparison_live.cu, needed for a
  real multi-frame backtrace. Compares frame functions, source line, and PCs.

Requires cuda-gdb on PATH; the live-core test also needs nvcc + a GPU. Tests
skip cleanly when these are unavailable.
"""

import os
import pathlib
import re
import shutil
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    CUDBG_GRID_STATUS_ACTIVE,
)

# Put the comparison parent directory on the path to import the shared framework.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from framework.gdb_driver import GdbDriver
from framework.lldb_driver import LldbDriver
from framework.comparator import ResultComparator
from framework.debugger_interface import DebuggerResult, ThreadInfo


def _get_cuda_gdb_path():
    return shutil.which("cuda-gdb")


# Leading "(x,y,z) (x,y,z)" (BlockIdx, ThreadIdx) of a non-coalesced
# cuda-gdb "info cuda threads" row.
_GDB_CUDA_THREAD_ROW = re.compile(
    r"\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)"
)


def _thread_name(block, thread):
    """Coordinate-derived thread name, matching LLDB's FormatThreadName."""
    return (
        f"blockIdx(x={block[0]} y={block[1]} z={block[2]}) "
        f"threadIdx(x={thread[0]} y={thread[1]} z={thread[2]})"
    )


class TestNVGPUCoreFileComparison(NVGPUCoreTestBase):
    """Compares LLDB and cuda-gdb on an artificial NVGPU (.nvcudmp) core."""

    NO_DEBUG_INFO_TESTCASE = True

    # Cubin fixture and the address compare_kernel resolves to (see
    # comparison_artificial.cu).
    CUBIN = "comparison_artificial.cubin"
    KERNEL_PC = 0x00007FFFCF203000

    # Synthetic CUDA handles for the core's single context/module.
    CONTEXT_ID = 0x1000
    MODULE_HANDLE = 0x2000
    GRID_ID = 1

    # One block per SM, each with a single thread at threadIdx (0,0,0).
    NUM_BLOCKS = 4

    # Register-class sizes. RZ/URZ are the always-zero R255/UR255; P7/UP7 are
    # the always-1 constant predicates (PT/UPT).
    NUM_R = 255  # R0..R254
    NUM_UR = 255  # UR0..UR254
    NUM_P = 8  # P0..P7
    NUM_UP = 8  # UP0..UP7

    # Nested device call chain comparison_live.cu walks before faulting
    # (innermost first).
    EXPECTED_FRAMES = ["live_crash_2", "live_crash_1", "live_crash_0", "live_kernel"]

    # Per-thread register value formulas, applied by both the builder and asserts.

    @staticmethod
    def reg_value(t, i):
        return 0x0A000000 | (t << 16) | i

    @staticmethod
    def ureg_value(t, i):
        return 0x0B000000 | (t << 16) | i

    @staticmethod
    def pred_value(t, i):
        return 1 if (i + t) % 2 == 0 else 0

    @staticmethod
    def upred_value(t, i):
        return 1 if (i // 2 + t) % 2 == 0 else 0

    def _expected_registers(self, block):
        """Ground-truth values from the builder formulas for one thread."""
        expected = {f"R{i}": self.reg_value(block, i) for i in range(self.NUM_R)}
        expected["RZ"] = 0
        expected.update(
            {f"UR{i}": self.ureg_value(block, i) for i in range(self.NUM_UR)}
        )
        expected["URZ"] = 0
        expected.update(
            {f"P{i}": self.pred_value(block, i) for i in range(self.NUM_P - 1)}
        )
        expected.update(
            {f"UP{i}": self.upred_value(block, i) for i in range(self.NUM_UP - 1)}
        )
        return expected

    def _build_comparison_core(self):
        """Build an artificial core with NUM_BLOCKS threads (one block per SM)
        stopped at compare_kernel. SM0 faults so both debuggers select it."""
        cubin = pathlib.Path(self.getSourcePath(self.CUBIN)).read_bytes()

        b = NVGPUCoreBuilder()
        dev = b.add_device(
            sm_major=8,
            sm_minor=0,
            num_sms=self.NUM_BLOCKS,
            num_warps_per_sm=1,
            num_lanes_per_warp=32,
            num_regs_per_lane=self.NUM_R,
            num_predicates_per_lane=self.NUM_P,
            num_uniform_regs_per_warp=self.NUM_UR,
            num_uniform_predicates_per_warp=self.NUM_UP,
            sm_type="sm_80",
            dev_name="cmp_device",
        )
        ctx = b.add_context(dev, context_id=self.CONTEXT_ID)
        module = b.add_module(ctx, module_handle=self.MODULE_HANDLE)
        b.add_relocated_cubin(cubin, module=module)
        b.add_grid(
            dev,
            grid_id=self.GRID_ID,
            context=ctx,
            module_handle=self.MODULE_HANDLE,
            grid_dim=(self.NUM_BLOCKS, 1, 1),
            block_dim=(1, 1, 1),
            grid_status=CUDBG_GRID_STATUS_ACTIVE,
        )

        for block in range(self.NUM_BLOCKS):
            faulting = block == 0
            error_pc = self.KERNEL_PC if faulting else None
            sm = b.add_sm(
                dev,
                sm_id=block,
                exception=4 if faulting else 0,
                error_pc=error_pc,
            )
            cta = b.add_cta(sm, grid_id=self.GRID_ID, block_idx=(block, 0, 0))
            warp = b.add_warp(
                cta,
                warp_id=0,
                valid_lanes_mask=1,
                active_lanes_mask=1,
                error_pc=error_pc,
            )
            lane = b.add_lane(
                warp, lane_id=0, thread_idx=(0, 0, 0), pc=self.KERNEL_PC,
                call_depth=1,
            )
            # Fill the full register file with per-thread formula values (t=block)
            # so the comparison exercises every R/UR/P/UP register, not zeros.
            b.set_lane_registers(
                lane, [self.reg_value(block, i) for i in range(self.NUM_R)]
            )
            b.set_warp_uniform_registers(
                warp, [self.ureg_value(block, i) for i in range(self.NUM_UR)]
            )
            b.set_lane_predicates(
                lane, [self.pred_value(block, i) for i in range(self.NUM_P)]
            )
            b.set_warp_uniform_predicates(
                warp, [self.upred_value(block, i) for i in range(self.NUM_UP)]
            )
            # A single terminating frame keeps cuda-gdb from recursing on a
            # missing caller.
            b.set_lane_backtrace(lane, [(0, 0, 0)])

        return b

    def _load_both_live(self):
        """Load a live core in both debuggers."""
        return self._load_both(self.generate_core())

    def _load_both_artificial(self):
        """Load an artificial core in both debuggers, building it on first use."""
        if getattr(self, "_artificial_core_path", None) is None:
            self._artificial_core_path = self.generate_artificial_core(
                self._build_comparison_core(), "comparison.nvcudmp"
            )
        return self._load_both(self._artificial_core_path)

    def _load_both(self, core_path):
        """Load an existing .nvcudmp in both debuggers.

        Returns the cuda-gdb driver, LLDB driver, and shared comparator. Skips
        cleanly if cuda-gdb is unavailable or cannot load the core.
        """
        cuda_gdb_path = _get_cuda_gdb_path()
        if not cuda_gdb_path:
            self.skipTest("cuda-gdb not found in PATH")

        # Drop drivers from the previous sub-test so a test method can load
        # several cores without retaining debugger state.
        if getattr(self, "_active_gdb_driver", None):
            self._active_gdb_driver.cleanup()
            self._active_gdb_driver = None
        if getattr(self, "_active_lldb_driver", None):
            self._active_lldb_driver.cleanup()
            self._active_lldb_driver = None

        # get_all_threads() visits every target, so keep only the current core.
        for target in list(self.dbg):
            self.dbg.DeleteTarget(target)

        lldb_driver = LldbDriver(self.dbg)
        self._active_lldb_driver = lldb_driver
        load = lldb_driver.load_core(core_path)
        self.assertTrue(
            load.success, f"LLDB failed to load core: {load.error_message}"
        )

        gpu_process = lldb_driver.process
        gpu_target = lldb_driver.target
        self.assertEqual(gpu_process.GetState(), lldb.eStateStopped)
        self.assertIn("nvptx", gpu_target.GetTriple(), "GPU target not found in core")
        self.dbg.SetSelectedTarget(gpu_target)

        gdb_driver = GdbDriver(cuda_gdb_path, prompt="(cuda-gdb)")
        self._active_gdb_driver = gdb_driver
        for command in (
            "set cuda coalescing off",
            "set cuda ptx_cache off",
            "set width 0",
            f"target cudacore {core_path}",
        ):
            result = gdb_driver.execute_command(command)
            self.assertTrue(
                result.success,
                f"cuda-gdb command '{command}' failed: {result.error_message}",
            )

        return gdb_driver, lldb_driver, ResultComparator(pc_tolerance=0)

    def tearDown(self):
        if getattr(self, "_active_gdb_driver", None):
            self._active_gdb_driver.cleanup()
            self._active_gdb_driver = None
        if getattr(self, "_active_lldb_driver", None):
            self._active_lldb_driver.cleanup()
            self._active_lldb_driver = None
        NVGPUCoreTestBase.tearDown(self)

    def _gdb_threads(self, gdb_driver):
        """Parse non-coalesced 'info cuda threads' into generic thread data."""
        result = gdb_driver.execute_command("info cuda threads")
        self.assertTrue(
            result.success,
            f"cuda-gdb 'info cuda threads' failed: {result.error_message}",
        )
        threads = []
        for line in result.raw_output.splitlines():
            match = _GDB_CUDA_THREAD_ROW.search(line)
            if not match:
                continue
            bx, by, bz, tx, ty, tz = (int(g) for g in match.groups())
            threads.append(
                ThreadInfo(
                    id=len(threads),
                    name=_thread_name((bx, by, bz), (tx, ty, tz)),
                )
            )
        return DebuggerResult(threads=threads)

    def test_compare_artificial_core(self):
        """Compare thread and register state from an artificial NVGPU core."""
        self._test_thread_list()
        self._test_registers()

    def test_compare_live_core(self):
        """Compare backtraces from a core generated by a live CUDA process."""
        self._test_backtrace()

    def _test_thread_list(self):
        """LLDB and cuda-gdb agree on the GPU thread list (count and
        coordinates) for the same artificial core."""
        gdb_driver, lldb_driver, comparator = self._load_both_artificial()
        gdb_result = self._gdb_threads(gdb_driver)
        lldb_result = lldb_driver.get_all_threads()
        gdb_names = [thread.name for thread in gdb_result.threads]
        lldb_names = [thread.name for thread in lldb_result.threads]

        self.assertEqual(len(lldb_names), self.NUM_BLOCKS)
        comparison = comparator.compare_threads(gdb_result, lldb_result)
        self.assertTrue(comparison.is_equivalent, comparison.get_summary())

    def _lldb_faulting_thread(self, gpu_process):
        """The single thread with a stop reason (only one CTA faults)."""
        faulting = [
            t
            for t in gpu_process
            if t.GetStopReason() != lldb.eStopReasonNone
        ]
        self.assertEqual(len(faulting), 1)
        return faulting[0]

    def _test_backtrace(self):
        """LLDB and cuda-gdb agree on the faulting thread's backtrace (call
        chain and PC) for the same live core."""
        gdb_driver, lldb_driver, comparator = self._load_both_live()
        thread = self._lldb_faulting_thread(lldb_driver.process)
        lldb_driver.select_thread(thread.GetThreadID())

        lldb_result = lldb_driver.get_backtrace()
        gdb_result = gdb_driver.get_backtrace()
        lldb_frames = lldb_result.backtrace
        gdb_frames = gdb_result.backtrace

        n = len(self.EXPECTED_FRAMES)
        self.assertEqual([f.function for f in lldb_frames[:n]], self.EXPECTED_FRAMES)
        self.assertEqual([f.function for f in gdb_frames[:n]], self.EXPECTED_FRAMES)

        comparison = comparator.compare_backtrace(
            DebuggerResult(backtrace=gdb_frames[:n]),
            DebuggerResult(backtrace=lldb_frames[:n]),
        )
        self.assertTrue(comparison.is_equivalent, comparison.get_summary())

    def _test_registers(self):
        """LLDB and cuda-gdb expose the identical SASS register file (R/UR/P/UP
        plus RZ/URZ) with identical values on the faulting thread, matching the
        builder's formula values.

        The compared common set is R0..R254, RZ, UR0..UR254, URZ, P0..P6,
        UP0..UP6. P7/UP7 (the always-1 PT/UPT) are excluded: cuda-gdb does not
        surface them under those names."""
        gdb_driver, lldb_driver, comparator = self._load_both_artificial()
        thread = self._lldb_faulting_thread(lldb_driver.process)
        lldb_driver.select_thread(thread.GetThreadID())

        names = set()
        names.update(f"R{i}" for i in range(self.NUM_R))
        names.add("RZ")
        names.update(f"UR{i}" for i in range(self.NUM_UR))
        names.add("URZ")
        names.update(f"P{i}" for i in range(self.NUM_P - 1))
        names.update(f"UP{i}" for i in range(self.NUM_UP - 1))

        lldb_result = lldb_driver.get_registers(register_names=sorted(names))
        gdb_result = gdb_driver.get_registers(register_names=sorted(names))
        self.assertTrue(lldb_result.success)
        self.assertTrue(gdb_result.success)
        self.assertEqual(set(gdb_result.registers), names)
        self.assertEqual(set(lldb_result.registers), names)

        expected = self._expected_registers(block=0)
        for name in sorted(names):
            self.assertEqual(
                gdb_result.registers[name].value,
                expected[name],
                f"cuda-gdb {name}",
            )
            self.assertEqual(
                lldb_result.registers[name].value,
                expected[name],
                f"LLDB {name}",
            )

        comparison = comparator.compare_registers(
            gdb_result,
            lldb_result,
            register_names=sorted(names),
        )
        self.assertTrue(comparison.is_equivalent, comparison.get_summary())
