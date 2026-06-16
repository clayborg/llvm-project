"""
Negative / malformed-core tests for the nvgpu-core reader.

Procedural generation makes structurally broken cores easy to construct.
These exercise the rejection and recovery paths in
ObjectFileELF::BuildNVGPUSectionList and ProcessNVGPUCore::DoLoadCore:
bad parent links, wrong parent kinds, out-of-range row indices, zero entry
sizes, missing device hierarchy, leaf truncation, and leaf-driven lane
sparseness.
"""

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    CUDBG_SHT_SM_TABLE,
    CUDBG_SHT_CTA_TABLE,
    CUDBG_SHT_DEV_REGS,
    CUDBG_SHT_GLOBAL_MEM,
    SM_ROW_SIZE,
    CTA_ROW_SIZE,
)

# Substring of the load error every "malformed/no-hierarchy" core surfaces:
# BuildNVGPUSectionList installs no nvgpucore root, so DoLoadCore bails.
NO_ROOT_ERROR = "did not produce a nvgpucore root section"


class TestNVGPUArtificialMalformed(NVGPUCoreTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _minimal(self, b):
        """Valid one-thread scaffold; tests bolt malformed sections onto it."""
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(dev, exception=4, error_pc=0x100000)
        cta = b.add_cta(sm, block_idx=(0, 0, 0))
        warp = b.add_warp(
            cta, valid_lanes_mask=1, active_lanes_mask=1, error_pc=0x100000
        )
        lane = b.add_lane(warp, lane_id=0, pc=0x100000)
        b.set_lane_registers(lane, [0] * 32)
        return dev, sm, cta, warp, lane

    def test_minimal_scaffold_loads(self):
        """Positive control: the bare _minimal() scaffold the malformed tests
        build on loads as a one-thread core."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        _, gpu_process = self.generate_and_load_artificial_core(
            b, "scaffold.nvcudmp"
        )
        self.assertEqual(gpu_process.GetNumThreads(), 1)

    def _assert_rejected_with_log(self, builder, name, log_substrs):
        """Load a core expecting failure; assert NO_ROOT_ERROR plus the
        condition-specific reject reason on the Modules log channel."""
        core_path = self.generate_artificial_core(builder, name)
        for target in list(self.dbg):
            self.dbg.DeleteTarget(target)
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid())

        log_file = self.getBuildArtifact(name + ".module.log")
        self.runCmd("log enable lldb module -f '%s'" % log_file)
        error = lldb.SBError()
        try:
            process = target.LoadCore(core_path, error)
        finally:
            self.runCmd("log disable lldb module")

        self.assertFalse(process.IsValid())
        self.assertIn(NO_ROOT_ERROR, error.GetCString())

        with open(log_file) as f:
            log_text = f.read()
        for substr in log_substrs:
            self.assertIn(
                substr, log_text,
                f"expected {substr!r} in module log for {name}",
            )

    def test_missing_device_hierarchy(self):
        """A device table with no SMs underneath produces no hierarchy."""
        b = NVGPUCoreBuilder()
        b.add_device()
        self._assert_rejected_with_log(
            b, "nohier.nvcudmp",
            ["no NVGPU device hierarchy found", "rejecting corefile"],
        )

    def test_invalid_parent_link(self):
        """An SM table with sh_link == 0 has no resolvable parent table."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            entsize=SM_ROW_SIZE,  # no Link => sh_link == 0
        )
        self._assert_rejected_with_log(
            b, "badlink.nvcudmp", ["invalid sh_link", "rejecting corefile"]
        )

    def test_wrong_parent_kind(self):
        """A CTA table parented by a device table skips the SM level."""
        b = NVGPUCoreBuilder()
        dev = b.add_device()
        b.add_sm(dev)
        b.add_raw_section(
            name=".cudbg.ctatbl.bad",
            sh_type=CUDBG_SHT_CTA_TABLE,
            content=b"\x00" * CTA_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,
            entsize=CTA_ROW_SIZE,
        )
        self._assert_rejected_with_log(
            b, "badkind.nvcudmp",
            ["resolves to a parent of", "expected type", "rejecting corefile"],
        )

    def test_out_of_range_row_index(self):
        """A leaf whose sh_info exceeds its parent table's row count is bad."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.regs.bad",
            sh_type=CUDBG_SHT_DEV_REGS,
            content=b"\x00" * 128,
            link=".cudbg.lntbl.dev0.sm0.cta0.wp0",
            info=99,
        )
        self._assert_rejected_with_log(
            b, "badinfo.nvcudmp",
            ["out of range for table", "rejecting corefile"],
        )

    def test_zero_entsize_table(self):
        """A container table with sh_entsize == 0 has no addressable rows."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,  # no EntSize => sh_entsize == 0
        )
        self._assert_rejected_with_log(
            b, "badentsize.nvcudmp", ["zero sh_entsize", "rejecting corefile"]
        )

    def test_truncated_leaf_is_skipped(self):
        """A leaf claiming to extend past EOF is dropped, but the core still
        loads and the rest of its state stays usable."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        # Claim a far larger sh_size than the actual content / file.
        b.add_raw_section(
            name=".cudbg.global.trunc",
            sh_type=CUDBG_SHT_GLOBAL_MEM,
            content=b"\xef\xbe\xad\xde",
            address=0x100000000,
            shsize=0x100000,
        )
        _, gpu_process = self.generate_and_load_artificial_core(
            b, name="trunc.nvcudmp"
        )
        # Core still loads with its single thread.
        self.assertEqual(gpu_process.GetNumThreads(), 1)
        # The truncated global region was skipped and is unreadable.
        self.expect(
            "memory read 0x100000000 --format x --size 4 -c 1",
            error=True,
            substrs=["core file does not contain"],
        )

    def test_sparse_lanes_materialize_by_leaf(self):
        """Only lanes with a per-lane leaf become threads, even when the warp
        marks many lanes valid."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(dev, exception=4, error_pc=0x100000)
        cta = b.add_cta(sm, block_idx=(1, 0, 0))
        # Lanes 0..7 marked valid/active, but only lane 7 gets a regs leaf.
        warp = b.add_warp(
            cta, valid_lanes_mask=0xFF, active_lanes_mask=0xFF,
            error_pc=0x100000,
        )
        lane = b.add_lane(warp, lane_id=7, thread_idx=(7, 0, 0), pc=0x100000)
        b.set_lane_registers(lane, [0] * 32)

        _, gpu_process = self.generate_and_load_artificial_core(
            b, name="sparse.nvcudmp"
        )
        self.assertEqual(gpu_process.GetNumThreads(), 1)
        self.assertEqual(
            gpu_process.GetThreadAtIndex(0).GetName(),
            "blockIdx(x=1 y=0 z=0) threadIdx(x=7 y=0 z=0)",
        )
