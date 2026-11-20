"""
Caller frames of a lane unwound through the SASS default unwind plan.

A lane with neither a backtrace table nor local memory is unwound with DWARF
CFI, which falls back to the SASS default plan (the caller's PC is RA) where
the cubin has no usable CFI. A kernel entry has no caller, so its RA holds
unrelated data: a caller PC is only real if it falls inside loaded code.
"""

import pathlib

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import NVGPUCoreBuilder


class TestNVGPUArtificialUnwind(NVGPUCoreTestCaseBase):
    # Addresses inside the elfv8.cubin fixture (values from
    # ../disass/TestNVGPUDisass.py).
    CUBIN_PATH = "../disass/elfv8.cubin"
    LANE_PC = 0x00007FFFCF280300
    # Inside acos_main, the kernel that calls acosf.
    CALLER_PC = 0x00007FFFCF27EE40
    # Any non-zero, 8-byte aligned value is a valid call frame address.
    STACK_POINTER = 0x1000

    CONTEXT_ID = 0x101
    GRID_ID = 1

    def _load_lane_with_return_address(self, return_address, name):
        """Load a core with one lane at LANE_PC whose RA (R21:R20) is
        `return_address`, and return that lane's thread."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        context = b.add_context(dev, context_id=self.CONTEXT_ID)
        b.add_grid(
            dev,
            grid_id=self.GRID_ID,
            context=context,
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
        )
        b.add_relocated_cubin(
            pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()
        )
        sm = b.add_sm(dev, sm_id=0)
        cta = b.add_cta(sm, grid_id=self.GRID_ID, block_idx=(0, 0, 0))
        warp = b.add_warp(cta, valid_lanes_mask=1, active_lanes_mask=1)
        lane = b.add_lane(warp, lane_id=0, pc=self.LANE_PC)
        regs = [0] * 32
        regs[1] = self.STACK_POINTER
        regs[20] = return_address & 0xFFFFFFFF
        regs[21] = return_address >> 32
        b.set_lane_registers(lane, regs)
        _, process = self.generate_and_load_artificial_core(b, name=name)
        self.assertEqual(process.GetNumThreads(), 1)
        return process.GetThreadAtIndex(0)

    def test_return_address_outside_code_is_not_a_caller(self):
        """An RA that points outside every loaded cubin does not become a
        caller frame."""
        thread = self._load_lane_with_return_address(
            0xBAD17CD7BAD17CD7, "garbage-ra.nvcudmp"
        )
        self.assertEqual(thread.GetNumFrames(), 1)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), self.LANE_PC)

    def test_return_address_into_code_is_a_caller(self):
        """An RA that points into loaded code is still unwound to."""
        thread = self._load_lane_with_return_address(self.CALLER_PC, "code-ra.nvcudmp")
        self.assertGreaterEqual(thread.GetNumFrames(), 2)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), self.LANE_PC)
        self.assertEqual(thread.GetFrameAtIndex(1).GetPC(), self.CALLER_PC)
