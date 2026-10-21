"""Hardware lane IDs need not equal their row indices in a CUDA core."""

import struct

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    CUDBG_SHT_DEV_REGS,
    CUDBG_SHT_LN_TABLE,
    LANE_ROW_SIZE,
    NVGPUCoreBuilder,
    cudbg_exception,
    pack_row,
)


class TestNVGPUCompactLanes(NVGPUCoreTestCaseBase):
    def _compact_core(self, hardware_lanes, valid_mask, active_mask, fault=True):
        """Emit compact rows directly: the normal builder pads to lane IDs."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=4)
        b.add_grid(dev, grid_id=7, grid_dim=(44, 1, 1), block_dim=(32, 1, 1))
        sm = b.add_sm(
            dev,
            exception=cudbg_exception("WARP_ILLEGAL_INSTRUCTION") if fault else 0,
            error_pc=0x100040 if fault else None,
        )
        cta = b.add_cta(sm, grid_id=7, block_idx=(43, 0, 0))
        b.add_warp(
            cta,
            valid_lanes_mask=valid_mask,
            active_lanes_mask=active_mask,
            error_pc=0x100040 if fault else None,
        )
        table_name = ".cudbg.lntbl.compact"
        rows = [
            pack_row(
                "CudbgThreadTableEntry",
                virtualPC=0x100000 + row * 16,
                ln=hardware_lane,
                threadIdxX=hardware_lane,
            )
            for row, hardware_lane in enumerate(hardware_lanes)
        ]
        b.add_raw_section(
            name=table_name,
            sh_type=CUDBG_SHT_LN_TABLE,
            content=b"".join(rows),
            link=".cudbg.wptbl.dev0.sm0.cta0",
            info=0,
            entsize=LANE_ROW_SIZE,
        )
        for row in range(len(rows)):
            b.add_raw_section(
                name=f".cudbg.regs.compact.row{row}",
                sh_type=CUDBG_SHT_DEV_REGS,
                content=struct.pack(
                    "<4I", *(0xAA00 + row * 16 + i for i in range(4))
                ),
                link=table_name,
                info=row,
            )
        return b

    def test_compact_nonzero_lane_is_not_an_sm_stand_in(self):
        """A hardware-valid lane in row zero retains its PC and registers."""
        for hardware_lane in (9, 29):
            with self.subTest(hardware_lane=hardware_lane):
                b = self._compact_core(
                    [hardware_lane], 1 << hardware_lane, 1 << hardware_lane
                )
                _, process = self.generate_and_load_artificial_core(
                    b, f"compact{hardware_lane}.nvcudmp"
                )
                self.assertEqual(process.GetNumThreads(), 1)
                thread = process.GetSelectedThread()
                self.assertEqual(
                    thread.GetName(),
                    f"blockIdx(x=43 y=0 z=0) threadIdx(x={hardware_lane} y=0 z=0)",
                )
                self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
                self.assertIn(
                    "Warp Illegal Instruction", thread.GetStopDescription(256)
                )
                frame = thread.GetFrameAtIndex(0)
                self.assertEqual(frame.GetPC(), 0x100000)
                self.assertEqual(
                    frame.FindRegister("R0").GetValueAsUnsigned(), 0xAA00
                )

    def test_reordered_lanes_use_hardware_active_mask(self):
        """Row zero/lane29 faults; row one/lane0 is valid but quiescent."""
        b = self._compact_core([29, 0], (1 << 29) | 1, 1 << 29)
        _, process = self.generate_and_load_artificial_core(b, "reordered.nvcudmp")
        self.assertEqual(process.GetNumThreads(), 2)
        fault = process.GetThreadAtIndex(0)
        quiet = process.GetThreadAtIndex(1)
        self.assertEqual(fault.GetStopReason(), lldb.eStopReasonException)
        self.assertEqual(quiet.GetStopReason(), lldb.eStopReasonNone)
        self.assertEqual(
            process.GetSelectedThread().GetThreadID(), fault.GetThreadID()
        )
        self.runCmd("thread select --blockidx '(43,0,0)' --threadidx '(29,0,0)'")
        self.expect("thread info", substrs=["threadIdx(x=29 y=0 z=0)"])
        self.assertEqual(fault.GetFrameAtIndex(0).GetPC(), 0x100000)
        self.assertEqual(quiet.GetFrameAtIndex(0).GetPC(), 0x100010)
        self.assertEqual(
            quiet.GetFrameAtIndex(0).FindRegister("R0").GetValueAsUnsigned(), 0xAA10
        )

    def test_out_of_range_hardware_lane_is_skipped(self):
        """Reject lane IDs outside the 32-bit masks without shifting by 32."""
        b = self._compact_core([0, 32], 1, 0, fault=False)
        _, process = self.generate_and_load_artificial_core(b, "badlane.nvcudmp")
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(
            process.GetThreadAtIndex(0).GetName(),
            "blockIdx(x=43 y=0 z=0) threadIdx(x=0 y=0 z=0)",
        )
