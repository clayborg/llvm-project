"""
SM-level exceptions with no surviving kernel context.

When a kernel faults but its warps run to completion before the coredump is
written, the SM row still records the exception type and error PC while every
warp is gone, so no lane is left to attribute the fault to. The reader stands
in for it with a lane-less thread whose single frame is the SM's error PC (see
`ThreadNVGPUCore`), which is what turns an otherwise threadless core into one
that names the fault and its location.

The live equivalent is the cuda-gdb testsuite's bug4247218, which races
cudaDeviceSynchronize() against context destruction; an artificial core
reproduces the resulting state deterministically and without a GPU.
"""

import pathlib
import struct

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    cudbg_exception,
    CTA_ROW_SIZE,
    CUDBG_SHT_CTA_TABLE,
    CUDBG_SHT_GRID_TABLE,
    GRID_ROW_SIZE,
)


class TestNVGPUArtificialSMException(NVGPUCoreTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Addresses inside the elfv8.cubin fixture, so an error PC symbolicates
    # (values from ../disass/TestNVGPUDisass.py). The lane and error PCs differ
    # so a frame can be traced back to the row it came from.
    CUBIN_PATH = "../disass/elfv8.cubin"
    SYMBOL_NAME = "acosf"
    LANE_PC = 0x00007FFFCF280300
    ERROR_PC = 0x00007FFFCF280340

    # Two faults, so a core can tell one SM's exception from another's. Both
    # predate every revision gate in CUDAExceptionToString, so the tests below
    # can assert the description LLDB prints for them verbatim.
    WARP_ILLEGAL_INSTRUCTION = cudbg_exception("WARP_ILLEGAL_INSTRUCTION")
    DEVICE_ILLEGAL_ADDRESS = cudbg_exception("DEVICE_ILLEGAL_ADDRESS")

    GLOBAL_ADDR = 0x100000000

    CONTEXT_ID = 0x101
    GRID_ID = 1
    MODULE_HANDLE = 0x2000

    def _builder(self, *, with_grid=False):
        """A device carrying the cubin fixture and one global region, with no
        SMs yet. Returns the builder and the device to hang SMs off.

        No context or grid table by default, which is the shape this feature
        exists for: a kernel that faulted but ran to completion leaves no
        resident grid, and the context may already be tearing down, so the
        dump can record an SM's exception with no launch state around it. A
        stand-in must not depend on either table -- it resolves its fault site
        through the root-level cubin image alone.

        Pass ``with_grid=True`` when the core also carries lane-backed
        threads, which do resolve their launch dimensions through a grid.
        """
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        if with_grid:
            context = b.add_context(dev, context_id=self.CONTEXT_ID)
            b.add_grid(
                dev, grid_id=self.GRID_ID, context=context,
                grid_dim=(8, 1, 1), block_dim=(32, 1, 1),
            )
        b.add_relocated_cubin(
            pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()
        )
        b.add_global_memory(self.GLOBAL_ADDR, struct.pack("<I", 0xDEADBEEF))
        return b, dev

    def _add_lane(self, b, sm, *, block_idx, exception=0, warp_error_pc=None):
        """Give `sm` a single-lane CTA, so it has a surviving thread."""
        cta = b.add_cta(sm, grid_id=self.GRID_ID, block_idx=block_idx)
        warp = b.add_warp(
            cta, valid_lanes_mask=1, active_lanes_mask=1,
            error_pc=warp_error_pc,
        )
        lane = b.add_lane(warp, lane_id=0, pc=self.LANE_PC, exception=exception)
        b.set_lane_registers(lane, [0] * 32)

    @staticmethod
    def _sm_name(device_idx, sm_id):
        return f"device {device_idx} SM {sm_id} (no kernel context)"

    @staticmethod
    def _lane_name(block_idx):
        return (
            f"blockIdx(x={block_idx[0]} y={block_idx[1]} z={block_idx[2]}) "
            "threadIdx(x=0 y=0 z=0)"
        )

    @staticmethod
    def _thread_names(process):
        return [thread.GetName() for thread in process]

    def test_stands_in_for_exception_after_warps_exit(self):
        """A faulted SM whose warps have all exited still names the exception
        and its location: one thread, one frame at the SM's error PC,
        symbolicated against the embedded cubin.

        The core carries no context, grid, CTA, warp or lane table -- only the
        SM row and a root-level cubin -- which is what a dump taken after the
        kernel completed looks like."""
        b, dev = self._builder()
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        _, process = self.generate_and_load_artificial_core(
            b, name="exited.nvcudmp"
        )

        self.assertEqual(process.GetNumThreads(), 1)
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetName(), self._sm_name(0, 0))
        self.assertEqual(
            process.GetSelectedThread().GetThreadID(), thread.GetThreadID()
        )
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn(
            "CUDA Exception: Device Illegal Address",
            thread.GetStopDescription(256),
        )

        # The error PC is the whole call stack: there is no lane to unwind
        # from, and no local memory or backtrace table to unwind with.
        self.assertEqual(thread.GetNumFrames(), 1)
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetPC(), self.ERROR_PC)
        self.assertEqual(frame.GetFunctionName(), self.SYMBOL_NAME)
        self.assertEqual(
            frame.FindRegister("PC").GetValueAsUnsigned(), self.ERROR_PC
        )
        self.assertEqual(
            frame.FindRegister("errorPC").GetValueAsUnsigned(), self.ERROR_PC
        )

    def test_only_the_fault_pc_is_readable(self):
        """PC and errorPC come from the SM row and read back; every other
        register is refused rather than reading as a zero the corefile never
        recorded, so register-backed expressions can't compute on a value
        that does not exist."""
        b, dev = self._builder()
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        _, process = self.generate_and_load_artificial_core(
            b, name="pconly.nvcudmp"
        )
        frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)

        for name in ("PC", "errorPC"):
            reg = frame.FindRegister(name)
            self.assertTrue(reg.IsValid(), f"{name} should be readable")
            self.assertEqual(reg.GetValueAsUnsigned(), self.ERROR_PC, name)

        # One per register class, plus the CUDA built-ins: a lane register, a
        # predicate, their uniform counterparts, an alias of R1/R2, and the
        # coordinates that only a real lane could supply.
        for name in ("R0", "R1", "RZ", "P0", "UR0", "UP0", "SP", "FP", "RA",
                     "threadIdx", "blockIdx", "blockDim", "gridDim",
                     "warpSize"):
            self.assertEqual(
                frame.FindRegister(name).GetValueAsUnsigned(0xDEAD), 0xDEAD,
                f"{name} should be unreadable on the SM-exception stand-in",
            )

    def test_matches_a_real_exited_kernel_core(self):
        """Reproduce the section shape of a real dump -- the one the cuda-gdb
        bug4247218 test leaves behind -- and check the stand-in survives it.

        A real producer differs from the minimal cores above in three ways
        that all touch this feature: the CUDA context outlives the kernel, the
        grid and per-SM CTA tables are still emitted but hold no rows, and
        cubins are reached through a module table instead of sitting at the
        root. Building without them leaves the zero-row-table and
        module-linked-cubin paths untested.
        """
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        context = b.add_context(dev, context_id=self.CONTEXT_ID)
        module = b.add_module(context, module_handle=self.MODULE_HANDLE)
        cubin = pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()
        b.add_relocated_cubin(cubin, module=module)
        # A real dump carries the unrelocated image beside the relocated one.
        # Only the relocated image has resolvable addresses, so the reader must
        # symbolicate through that one and leave this copy alone.
        b.add_unrelocated_cubin(cubin, module=module)
        b.add_global_memory(self.GLOBAL_ADDR, struct.pack("<I", 0xDEADBEEF))
        # Only one SM records the fault, as on real hardware where the rest of
        # the SMs ran the same kernel without faulting.
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        b.add_sm(dev, sm_id=1)
        # The grid completed and every warp exited, so these tables are
        # present with zero rows. The builder only emits populated tables, so
        # go through the raw-section escape hatch.
        b.add_raw_section(
            name=".cudbg.gridtbl.dev0", sh_type=CUDBG_SHT_GRID_TABLE,
            content=b"", link=".cudbg.devtbl", info=0,
            entsize=GRID_ROW_SIZE,
        )
        for sm_id in range(2):
            b.add_raw_section(
                name=f".cudbg.ctatbl.dev0.sm{sm_id}",
                sh_type=CUDBG_SHT_CTA_TABLE, content=b"",
                link=".cudbg.smtbl.dev0", info=sm_id, entsize=CTA_ROW_SIZE,
            )

        _, process = self.generate_and_load_artificial_core(
            b, name="realshape.nvcudmp"
        )

        # The empty tables contribute no CTAs or grids, so the faulted SM is
        # still the only thread and still resolves its fault site.
        self.assertEqual(process.GetNumThreads(), 1)
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetName(), self._sm_name(0, 0))
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn(
            "CUDA Exception: Device Illegal Address",
            thread.GetStopDescription(256),
        )
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetPC(), self.ERROR_PC)
        self.assertEqual(frame.GetFunctionName(), self.SYMBOL_NAME)
        self.expect(
            f"memory read {self.GLOBAL_ADDR:#x} --format x --size 4 -c 1",
            substrs=["0xdeadbeef"],
        )

    def test_exception_without_error_pc(self):
        """An SM that recorded a fault but no error PC has no site to report,
        yet the exception itself still reaches the user."""
        b, dev = self._builder()
        b.add_sm(
            dev, sm_id=4, exception=cudbg_exception("WARP_OUT_OF_RANGE_ADDRESS")
        )
        _, process = self.generate_and_load_artificial_core(
            b, name="noerrorpc.nvcudmp"
        )

        self.assertEqual(process.GetNumThreads(), 1)
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetName(), self._sm_name(0, 4))
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn(
            "CUDA Exception: Warp Out-of-range Address",
            thread.GetStopDescription(256),
        )
        self.assertEqual(thread.GetNumFrames(), 1)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), 0)

    def test_idle_sm_gets_no_thread(self):
        """Only a faulted SM stands in. An SM that is merely empty -- no lanes
        and no exception -- is not a stop reason and gets no thread."""
        b, dev = self._builder()
        b.add_sm(dev, sm_id=0)
        b.add_sm(
            dev, sm_id=1, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        b.add_sm(dev, sm_id=2)
        _, process = self.generate_and_load_artificial_core(
            b, name="idle.nvcudmp"
        )

        self.assertEqual(self._thread_names(process), [self._sm_name(0, 1)])

    def test_surviving_lane_suppresses_stand_in(self):
        """An SM with both an exception and a surviving lane gets no stand-in:
        the lane already borrows the SM's exception, so a second thread would
        double-report the same fault."""
        b, dev = self._builder(with_grid=True)
        sm = b.add_sm(
            dev, sm_id=0, exception=self.WARP_ILLEGAL_INSTRUCTION,
            error_pc=self.ERROR_PC,
        )
        # A valid warp errorPC is what lets an active lane borrow sm.exception.
        self._add_lane(b, sm, block_idx=(0, 0, 0), warp_error_pc=self.ERROR_PC)
        _, process = self.generate_and_load_artificial_core(
            b, name="survivinglane.nvcudmp"
        )

        self.assertEqual(
            self._thread_names(process), [self._lane_name((0, 0, 0))]
        )
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn(
            "CUDA Exception: Warp Illegal Instruction",
            thread.GetStopDescription(256),
        )

    def test_lane_exception_outranks_sm_exception(self):
        """Both fault kinds in one core: each gets its own thread, but the
        faulting lane is selected because it carries thread coordinates and
        registers that the SM-level stand-in cannot. This is the order
        cuda-gdb uses, which only consults SM state after finding no lane
        exception."""
        b, dev = self._builder(with_grid=True)
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        sm1 = b.add_sm(dev, sm_id=1)
        self._add_lane(
            b, sm1, block_idx=(7, 0, 0), exception=self.WARP_ILLEGAL_INSTRUCTION
        )
        _, process = self.generate_and_load_artificial_core(
            b, name="bothfaults.nvcudmp"
        )

        self.assertEqual(
            self._thread_names(process),
            [self._sm_name(0, 0), self._lane_name((7, 0, 0))],
        )
        self.assertEqual(
            process.GetSelectedThread().GetName(), self._lane_name((7, 0, 0))
        )

    def test_stand_in_appears_in_aggregated_thread_list(self):
        """The default `thread list` aggregates by fault location and labels
        rows by coordinates. A stand-in has none, so it gets a row labelled by
        name -- without that it is dropped from the default view whenever
        lane-backed threads are present, hiding the very fault this thread
        exists to report.

        Both threads here resolve to the same function, so this also pins down
        that a coordinate-less thread never merges into a coordinate-bearing
        row: they stay two entries.

        The stand-in's SM has no grid of its own, so this also covers the two
        kinds coexisting in one core: a gridless SM-level fault alongside a
        lane that does resolve through a grid."""
        b, dev = self._builder(with_grid=True)
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        sm1 = b.add_sm(dev, sm_id=1)
        self._add_lane(
            b, sm1, block_idx=(7, 0, 0),
            exception=self.WARP_ILLEGAL_INSTRUCTION,
        )
        self.generate_and_load_artificial_core(b, name="aggregated.nvcudmp")

        self.expect(
            "thread list",
            substrs=[
                "1 thread(s): device 0 SM 0 (no kernel context), "
                "stop reason = CUDA Exception: Device Illegal Address",
                "1 thread(s): blockIdx(x=7 y=0 z=0) threadIdx(x=0 y=0 z=0), "
                "stop reason = CUDA Exception: Warp Illegal Instruction",
                self.SYMBOL_NAME,
            ],
        )

    def test_stand_ins_at_one_site_collapse_to_one_row(self):
        """Several SMs faulting at the same site aggregate into a single row,
        so a device-wide fault does not print one row per SM. The individual
        SMs stay available under `thread list -v`."""
        b, dev = self._builder()
        for sm_id in range(3):
            b.add_sm(
                dev, sm_id=sm_id, exception=self.DEVICE_ILLEGAL_ADDRESS,
                error_pc=self.ERROR_PC,
            )
        _, process = self.generate_and_load_artificial_core(
            b, name="collapsed.nvcudmp"
        )
        self.assertEqual(process.GetNumThreads(), 3)

        self.expect(
            "thread list",
            substrs=[
                "3 thread(s): no kernel context, "
                "stop reason = CUDA Exception: Device Illegal Address",
                self.SYMBOL_NAME,
            ],
        )
        self.expect(
            "thread list -v",
            substrs=[self._sm_name(0, sm_id) for sm_id in range(3)],
        )

    def test_stand_in_has_no_thread_scoped_memory(self):
        """The stand-in has no lane, CTA, or grid, so every thread-scoped
        address space reports a miss rather than reading unrelated state.
        Global memory is not thread-scoped and stays readable."""
        b, dev = self._builder()
        b.add_sm(
            dev, sm_id=0, exception=self.DEVICE_ILLEGAL_ADDRESS,
            error_pc=self.ERROR_PC,
        )
        _, process = self.generate_and_load_artificial_core(
            b, name="standinmemory.nvcudmp"
        )
        process.SetSelectedThread(process.GetThreadAtIndex(0))

        self.expect(
            f"memory read {self.GLOBAL_ADDR:#x} --format x --size 4 -c 1",
            substrs=["0xdeadbeef"],
        )
        for space in ("local", "shared", "const", "param", "generic"):
            self.expect(
                f"memory read -p {space} 0x200000 --format x --size 4 -c 1",
                error=True,
                substrs=[f"does not contain address space '{space}'"],
            )
