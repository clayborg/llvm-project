"""Test CUDA thread coordinates from artificial NVGPU core files."""

import json

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    cudbg_exception,
)
from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase


class TestNVGPUCoreThreadCoords(NVGPUCoreTestCaseBase):
    CONTEXT_ID = 0x8000000100000001
    GRID_ID = 0xF000000500000005
    DECOY_GRID_ID = 0xE000000400000004
    BLOCK_IDX = (8, 9, 10)
    THREAD_IDX = ((11, 12, 13), (21, 22, 23))
    LANE_IDS = (3, 7)
    PC = 0x100000

    def _build_core(
        self,
        *,
        matching_grid=True,
        cluster_idx=(4, 5, 6),
        cta_cluster_dim=(2, 3, 4),
        grid_cluster_dim=(90, 91, 92),
        grid_preferred_cluster_dim=(93, 94, 95),
        cta_entry_size=None,
        grid_entry_size=None,
        exception=0,
    ):
        builder = NVGPUCoreBuilder()
        builder.add_device()  # Keep the populated device ID nonzero.
        device = builder.add_device(
            num_regs_per_lane=1, sm_major=9, sm_type="sm_90"
        )
        context = builder.add_context(device, context_id=self.CONTEXT_ID)

        # Put a nonmatching grid first so lookup must use the 64-bit grid ID.
        builder.add_grid(
            device,
            grid_id=self.DECOY_GRID_ID,
            cluster_dim=(80, 81, 82),
            preferred_cluster_dim=(83, 84, 85),
            table_entry_size=grid_entry_size,
        )
        if matching_grid:
            builder.add_grid(
                device,
                grid_id=self.GRID_ID,
                context=context,
                cluster_dim=grid_cluster_dim,
                preferred_cluster_dim=grid_preferred_cluster_dim,
                table_entry_size=grid_entry_size,
            )

        sm = builder.add_sm(device, sm_id=17)
        cta = builder.add_cta(
            sm,
            grid_id=self.GRID_ID,
            block_idx=self.BLOCK_IDX,
            cluster_idx=cluster_idx,
            cluster_dim=cta_cluster_dim,
            table_entry_size=cta_entry_size,
        )
        warp = builder.add_warp(
            cta,
            warp_id=23,
            valid_lanes_mask=(1 << self.LANE_IDS[0]) | (1 << self.LANE_IDS[1]),
            active_lanes_mask=1 << self.LANE_IDS[0],
        )
        for lane_id, thread_idx in zip(self.LANE_IDS, self.THREAD_IDX):
            lane = builder.add_lane(
                warp,
                lane_id=lane_id,
                thread_idx=thread_idx,
                pc=self.PC,
                exception=exception if lane_id == self.LANE_IDS[0] else 0,
            )
            builder.set_lane_registers(lane, [0])
        return builder

    @staticmethod
    def _thread_name(thread_idx):
        return (
            "blockIdx(x=8 y=9 z=10) "
            f"threadIdx(x={thread_idx[0]} y={thread_idx[1]} z={thread_idx[2]})"
        )

    def _thread(self, process, thread_idx):
        name = self._thread_name(thread_idx)
        matches = [thread for thread in process if thread.GetName() == name]
        self.assertEqual(len(matches), 1, f"expected one thread named {name!r}")
        return matches[0]

    def _json_info(self, thread):
        self.runCmd(f"thread info --json {thread.GetIndexID()}")
        output = self.res.GetOutput()
        json_start = output.find("{")
        self.assertNotEqual(json_start, -1, output)
        return json.loads(output[json_start:])

    def test_coordinates(self):
        _, process = self.generate_and_load_artificial_core(
            self._build_core(), "coordinates.nvcudmp"
        )

        first = self._thread(process, self.THREAD_IDX[0])
        self.assertEqual(
            self._json_info(first),
            {
                "cuda": {
                    "block_idx": {"x": 8, "y": 9, "z": 10},
                    "cluster_dim": {"x": 2, "y": 3, "z": 4},
                    "cluster_idx": {"x": 4, "y": 5, "z": 6},
                    "context_id": self.CONTEXT_ID,
                    "device": 1,
                    "grid_id": self.GRID_ID,
                    "is_active": True,
                    "lane": 3,
                    "sm": 17,
                    "thread_idx": {"x": 11, "y": 12, "z": 13},
                    "warp": 23,
                }
            },
        )

        second_cuda = self._json_info(
            self._thread(process, self.THREAD_IDX[1])
        )["cuda"]
        self.assertEqual(
            second_cuda["thread_idx"], {"x": 21, "y": 22, "z": 23}
        )
        self.assertIs(second_cuda["is_active"], False)

    def test_incomplete_block_coordinates_have_neutral_label(self):
        # A 16-byte CTA prefix retains the grid ID but truncates blockIdx,
        # leaving the block coordinates incomplete.
        _, process = self.generate_and_load_artificial_core(
            self._build_core(cta_entry_size=16),
            "incomplete-block-coordinates.nvcudmp",
        )

        self.assertEqual(process.GetNumThreads(), 2)
        actual_thread_indices = set()
        actual_lanes = set()
        for thread in process:
            cuda = self._json_info(thread)["cuda"]
            self.assertEqual(cuda["context_id"], self.CONTEXT_ID)
            self.assertEqual(cuda["grid_id"], self.GRID_ID)
            self.assertNotIn("block_idx", cuda)
            actual_thread_indices.add(tuple(cuda["thread_idx"][axis] for axis in "xyz"))
            actual_lanes.add(cuda["lane"])
            self.assertEqual(thread.GetFrameAtIndex(0).GetPC(), self.PC)

        self.assertEqual(actual_thread_indices, set(self.THREAD_IDX))
        self.assertEqual(actual_lanes, set(self.LANE_IDS))

        self.runCmd("thread list")
        output = self.res.GetOutput()
        self.assertIn(
            "2 thread(s) at pc=0x100000: CUDA thread coordinates unavailable",
            output,
        )
        self.assertNotIn("blockIdx(", output)
        self.assertNotIn("threadIdx(", output)

    def test_thread_format_printf_suffixes(self):
        self.generate_and_load_artificial_core(
            self._build_core(), "thread-format.nvcudmp"
        )
        self.addTearDownHook(
            lambda: self.runCmd("settings clear thread-format")
        )

        # The first % introduces the suffix; a following % starts the raw
        # printf format.
        self.runCmd(
            "settings set thread-format "
            "'device=${thread.info.cuda.device%%02llu} "
            "lane=${thread.info.cuda.lane%%02llu} "
            "block.x=${thread.info.cuda.block_idx.x%%03llu} "
            "thread.x=${thread.info.cuda.thread_idx.x%%03llu} "
            "grid=${thread.info.cuda.grid_id%%llu}\\n'"
        )
        self.expect(
            "thread list -v",
            substrs=[
                "device=01 lane=03 block.x=008 thread.x=011 "
                "grid=17293822590577541125",
                "device=01 lane=07 block.x=008 thread.x=021 "
                "grid=17293822590577541125",
            ],
        )

    def test_cluster_coordinates(self):
        cases = [
            (
                "current-clustered-cta.nvcudmp",
                {},
                ((4, 5, 6), (2, 3, 4)),
            ),
            (
                "cta-preferred-dims-override-grid-ordinary.nvcudmp",
                {
                    "cluster_idx": (2, 1, 2),
                    "cta_cluster_dim": (4, 6, 5),
                    "grid_cluster_dim": (2, 3, 5),
                    "grid_preferred_cluster_dim": (4, 6, 5),
                },
                ((2, 1, 2), (4, 6, 5)),
            ),
            (
                "noncluster-launch.nvcudmp",
                {
                    "cluster_idx": (0, 0, 0),
                    "cta_cluster_dim": (0, 0, 0),
                    "grid_cluster_dim": (0, 0, 0),
                },
                ((0, 0, 0), (0, 0, 0)),
            ),
            (
                "40-byte-cta-grid-dim-fallback.nvcudmp",
                {
                    "cluster_idx": (1, 2, 3),
                    "cta_cluster_dim": (101, 102, 103),
                    "grid_cluster_dim": (14, 15, 16),
                    "cta_entry_size": 40,
                },
                ((1, 2, 3), (14, 15, 16)),
            ),
            (
                "40-byte-cta-without-grid.nvcudmp",
                {
                    "matching_grid": False,
                    "cluster_idx": (1, 2, 3),
                    "cta_cluster_dim": (101, 102, 103),
                    "cta_entry_size": 40,
                },
                None,
            ),
            (
                "single-block-cluster.nvcudmp",
                {
                    "cluster_idx": self.BLOCK_IDX,
                    "cta_cluster_dim": (1, 1, 1),
                    "grid_cluster_dim": (1, 1, 1),
                },
                (self.BLOCK_IDX, (1, 1, 1)),
            ),
        ]

        for name, options, expected in cases:
            with self.subTest(name=name):
                builder = self._build_core(**options)
                _, process = self.generate_and_load_artificial_core(builder, name)
                cuda = self._json_info(
                    self._thread(process, self.THREAD_IDX[0])
                )["cuda"]
                if expected is None:
                    self.assertNotIn("cluster_idx", cuda)
                    self.assertNotIn("cluster_dim", cuda)
                    continue

                expected_idx, expected_dim = expected
                actual_idx = tuple(cuda["cluster_idx"][axis] for axis in "xyz")
                self.assertEqual(actual_idx, expected_idx)
                actual_dim = tuple(cuda["cluster_dim"][axis] for axis in "xyz")
                self.assertEqual(actual_dim, expected_dim)

    def test_missing_grid_preserves_thread_state(self):
        _, process = self.generate_and_load_artificial_core(
            self._build_core(matching_grid=False, exception=4),
            "missing-grid.nvcudmp",
        )
        expected_name = self._thread_name(self.THREAD_IDX[0])
        thread = self._thread(process, self.THREAD_IDX[0])

        cuda = self._json_info(thread)["cuda"]
        self.assertEqual(
            cuda,
            {
                "block_idx": {"x": 8, "y": 9, "z": 10},
                "cluster_dim": {"x": 2, "y": 3, "z": 4},
                "cluster_idx": {"x": 4, "y": 5, "z": 6},
                "device": 1,
                "grid_id": self.GRID_ID,
                "is_active": True,
                "lane": 3,
                "sm": 17,
                "thread_idx": {"x": 11, "y": 12, "z": 13},
                "warp": 23,
            },
        )
        stream = lldb.SBStream()
        self.assertFalse(
            thread.GetInfoItemByPathAsString("cuda.context_id", stream)
        )
        self.assertEqual(stream.GetData(), "")
        self.assertEqual(thread.GetName(), expected_name)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn(
            "CUDA Exception: Warp Illegal Instruction",
            thread.GetStopDescription(256),
        )

    def test_known_zero_and_false_coordinates_are_serialized(self):
        builder = NVGPUCoreBuilder()
        device = builder.add_device(num_regs_per_lane=1)
        builder.add_grid(
            device,
            grid_id=0,
            cluster_dim=(0, 0, 0),
        )
        sm = builder.add_sm(device, sm_id=0)
        cta = builder.add_cta(
            sm,
            grid_id=0,
            block_idx=(0, 0, 0),
            cluster_dim=(0, 0, 0),
        )
        warp = builder.add_warp(
            cta,
            warp_id=0,
            valid_lanes_mask=1,
            active_lanes_mask=0,
        )
        lane = builder.add_lane(
            warp,
            lane_id=0,
            thread_idx=(0, 0, 0),
            pc=self.PC,
        )
        builder.set_lane_registers(lane, [0])

        _, process = self.generate_and_load_artificial_core(
            builder, "zero-coordinates.nvcudmp"
        )
        self.assertEqual(
            self._json_info(process.GetThreadAtIndex(0)),
            {
                "cuda": {
                    "block_idx": {"x": 0, "y": 0, "z": 0},
                    "cluster_dim": {"x": 0, "y": 0, "z": 0},
                    "cluster_idx": {"x": 0, "y": 0, "z": 0},
                    "context_id": 0,
                    "device": 0,
                    "grid_id": 0,
                    "is_active": False,
                    "lane": 0,
                    "sm": 0,
                    "thread_idx": {"x": 0, "y": 0, "z": 0},
                    "warp": 0,
                }
            },
        )

    def test_thread_selection_patterns_inherit_metadata(self):
        _, process = self.generate_and_load_artificial_core(
            self._build_core(), "coordinate-selection.nvcudmp"
        )
        first = self._thread(process, self.THREAD_IDX[0])
        second = self._thread(process, self.THREAD_IDX[1])

        self.assertTrue(process.SetSelectedThread(first))
        # An explicit group is a per-axis pattern. Its unspecified y/z axes
        # stay wildcards, while the omitted block group is inherited in full
        # from the selected thread's typed metadata.
        self.runCmd("thread select --threadidx '(x=21)'")
        self.assertEqual(
            process.GetSelectedThread().GetThreadID(), second.GetThreadID()
        )

        # Conversely, an omitted thread group inherits the selected thread's
        # complete threadIdx. The first lane has the same block but must not
        # win simply because it appears first in the process thread list.
        self.runCmd("thread select --blockidx '(x=8)'")
        self.assertEqual(
            process.GetSelectedThread().GetThreadID(), second.GetThreadID()
        )

    def test_stop_attribution_precedence_and_active_lane_gating(self):
        builder = NVGPUCoreBuilder()
        device = builder.add_device(num_regs_per_lane=1)
        context = builder.add_context(device, context_id=self.CONTEXT_ID)
        builder.add_grid(device, grid_id=self.GRID_ID, context=context)
        sm = builder.add_sm(
            device,
            exception=cudbg_exception("DEVICE_ILLEGAL_ADDRESS"),
            error_pc=self.PC,
        )
        cta = builder.add_cta(
            sm, grid_id=self.GRID_ID, block_idx=self.BLOCK_IDX
        )

        warp_illegal = cudbg_exception("WARP_ILLEGAL_INSTRUCTION")
        cases = [
            ((31, 1, 0), True, warp_illegal, "Warp Illegal Instruction"),
            ((32, 1, 0), False, 0, None),
            ((33, 1, 0), True, 0, "Device Illegal Address"),
        ]
        for warp_id, (thread_idx, active, lane_exception, _) in enumerate(cases):
            warp = builder.add_warp(
                cta,
                warp_id=warp_id,
                valid_lanes_mask=1,
                active_lanes_mask=1 if active else 0,
                error_pc=self.PC,
                is_warp_broken=True,
            )
            lane = builder.add_lane(
                warp,
                lane_id=0,
                thread_idx=thread_idx,
                pc=self.PC,
                exception=lane_exception,
            )
            builder.set_lane_registers(lane, [0])

        _, process = self.generate_and_load_artificial_core(
            builder, "stop-attribution.nvcudmp"
        )

        for thread_idx, active, _, expected_exception in cases:
            with self.subTest(thread_idx=thread_idx):
                thread = self._thread(process, thread_idx)
                expected_reason = (
                    lldb.eStopReasonException
                    if expected_exception
                    else lldb.eStopReasonNone
                )
                self.assertEqual(thread.GetStopReason(), expected_reason)
                if expected_exception:
                    self.assertIn(
                        f"CUDA Exception: {expected_exception}",
                        thread.GetStopDescription(256),
                    )
                self.assertIs(
                    self._json_info(thread)["cuda"]["is_active"], active
                )
                self.assertEqual(thread.IsActive(), active)
