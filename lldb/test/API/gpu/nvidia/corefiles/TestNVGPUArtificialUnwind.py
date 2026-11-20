"""
Caller frames of a lane in a GPU core.

Where a function has no usable CFI, the SASS default unwind plan takes the
caller's PC from RA (R21:R20). A kernel has no caller, so its RA can hold
unrelated data, and a caller PC is only real if it falls in a code section of a
loaded cubin.

elfv8.cubin carries CFI for both of its functions, and the CIE of the kernel,
acos_main, has no return-address column, so a walk with CFI ends at the kernel
by itself. The tests that exercise the default plan hide the CFI.
"""

import pathlib

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import NVGPUCoreBuilder


class TestNVGPUArtificialUnwind(NVGPUCoreTestCaseBase):
    # Addresses inside the elfv8.cubin fixture.
    CUBIN_PATH = "../disass/elfv8.cubin"
    # Entry of acosf, which has an FDE.
    ACOSF = 0x00007FFFCF280300
    # Entry of the kernel acos_main, and the end of its .text section.
    ACOS_MAIN = 0x00007FFFCF27EE00
    ACOS_MAIN_END = 0x00007FFFCF27FA00
    # acos_main calls acosf at 0x7FFFCF27F700, so this is the real return
    # address.
    RETURN_ADDRESS = 0x00007FFFCF27F710
    GARBAGE_RETURN_ADDRESS = 0xBAD17CD7BAD17CD7
    # Any non-zero, 8-byte aligned value is a valid call frame address. The
    # core maps memory there, so that the unwinder's own checks of the frame
    # pass and only the check of the caller PC can end the walk.
    STACK_POINTER = 0x1000

    CONTEXT_ID = 0x101
    GRID_ID = 1

    def _load_lane(self, name, *, lane_pc, return_address, hide_cfi=False):
        """Load a core with one lane at `lane_pc` whose RA (R21:R20) is
        `return_address`, and return that lane's thread. `hide_cfi` renames
        .debug_frame so that the lane unwinds with the SASS default plan."""
        cubin = pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()
        if hide_cfi:
            self.assertIn(b".debug_frame", cubin)
            cubin = cubin.replace(b".debug_frame", b".nodbg_frame")
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
        b.add_relocated_cubin(cubin)
        b.add_global_memory(self.STACK_POINTER, bytes(16))
        sm = b.add_sm(dev, sm_id=0)
        cta = b.add_cta(sm, grid_id=self.GRID_ID, block_idx=(0, 0, 0))
        warp = b.add_warp(cta, valid_lanes_mask=1, active_lanes_mask=1)
        lane = b.add_lane(warp, lane_id=0, pc=lane_pc)
        regs = [0] * 32
        regs[1] = self.STACK_POINTER
        regs[20] = return_address & 0xFFFFFFFF
        regs[21] = return_address >> 32
        b.set_lane_registers(lane, regs)
        _, process = self.generate_and_load_artificial_core(b, name=name)
        self.assertEqual(process.GetNumThreads(), 1)
        return process.GetThreadAtIndex(0)

    def _assert_kernel_has_no_caller(self, return_address, name):
        """A lane at the kernel's entry, unwound with the default plan, has no
        caller frame when RA is `return_address`."""
        thread = self._load_lane(
            name,
            lane_pc=self.ACOS_MAIN,
            return_address=return_address,
            hide_cfi=True,
        )
        self.assertEqual(thread.GetNumFrames(), 1)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), self.ACOS_MAIN)

    def test_return_address_outside_code_is_not_a_caller(self):
        """An RA that points outside every loaded cubin does not become a
        caller frame."""
        self._assert_kernel_has_no_caller(
            self.GARBAGE_RETURN_ADDRESS, "garbage-ra.nvcudmp"
        )

    def test_return_address_in_segment_outside_code_is_not_a_caller(self):
        """A core loads each cubin's PT_LOAD segment at address 0 as an
        executable container, so a small RA falls inside it. With no code
        section there, it does not become a caller frame either."""
        self._assert_kernel_has_no_caller(0x400, "segment-ra.nvcudmp")

    def test_return_address_into_code_is_a_caller(self):
        """An RA that points into loaded code is unwound to. The walk then ends
        at the kernel, whose CIE has no return-address column."""
        thread = self._load_lane(
            "code-ra.nvcudmp",
            lane_pc=self.ACOSF,
            return_address=self.RETURN_ADDRESS,
        )
        self.assertEqual(thread.GetNumFrames(), 2)
        self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), self.ACOSF)
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "acosf")
        self.assertEqual(thread.GetFrameAtIndex(1).GetPC(), self.RETURN_ADDRESS)
        self.assertIn("acos_main", thread.GetFrameAtIndex(1).GetFunctionName())

    def test_return_address_at_end_of_code_is_a_caller(self):
        """An RA at the end of a code section belongs to the call just before
        it, as the unwinder resolves it, so it is unwound to."""
        thread = self._load_lane(
            "section-end-ra.nvcudmp",
            lane_pc=self.ACOSF,
            return_address=self.ACOS_MAIN_END,
        )
        self.assertEqual(thread.GetNumFrames(), 2)
        self.assertEqual(thread.GetFrameAtIndex(1).GetPC(), self.ACOS_MAIN_END)
        self.assertIn("acos_main", thread.GetFrameAtIndex(1).GetFunctionName())
