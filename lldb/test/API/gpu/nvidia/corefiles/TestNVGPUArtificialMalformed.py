"""
Negative / malformed-core tests for the nvgpu-core reader.

Procedural generation makes structurally broken cores easy to construct.
These exercise the rejection and recovery paths in
ObjectFileELF::BuildNVGPUSectionList and ProcessNVGPUCore::DoLoadCore:
bad parent links, wrong parent kinds, out-of-range row indices, zero entry
sizes, table sizes that cover later sections, tables cut off by the end of the
file, missing device hierarchy, leaf truncation, and leaf-driven lane
sparseness.
"""


import os

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    CUDBG_SHT_SM_TABLE,
    CUDBG_SHT_CTA_TABLE,
    CUDBG_SHT_LN_TABLE,
    CUDBG_SHT_DEV_REGS,
    CUDBG_SHT_GLOBAL_MEM,
    cudbg_exception,
    SM_ROW_SIZE,
    CTA_ROW_SIZE,
    LANE_ROW_SIZE,
)

# A section size that ends past the end of every core these tests build. It is
# small on purpose. A reader that walks every row that such a table claims
# still finishes, and the test then fails because the core loads. With a size
# near 2^63 that reader would allocate until memory runs out.
PAST_END_OF_FILE = 0x100000

# Substring of the load error every "malformed/no-hierarchy" core surfaces:
# BuildNVGPUSectionList installs no nvgpucore root, so DoLoadCore bails.
NO_ROOT_ERROR = "did not produce a nvgpucore root section"

# The load error names no cause. These are the warnings in which
# BuildNVGPUSectionList tells the user why it refused the core. The first is
# for a malformed section, the second for a core with no SM under any device.
MALFORMED_WARNING = "GPU corefile appears malformed"
NO_HIERARCHY_WARNING = "GPU corefile contains no device hierarchy"

# Any real fault will do for these scaffolds; take the code from the SDK
# header rather than writing the number down.
WARP_ILLEGAL_INSTRUCTION = cudbg_exception("WARP_ILLEGAL_INSTRUCTION")


class TestNVGPUArtificialMalformed(NVGPUCoreTestCaseBase):
    def _minimal(self, b):
        """Valid one-thread scaffold; tests bolt malformed sections onto it."""
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(
            dev, exception=WARP_ILLEGAL_INSTRUCTION, error_pc=0x100000
        )
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

    def _assert_rejected_with_log(
        self, builder, name, log_substrs, warning=MALFORMED_WARNING
    ):
        """Load a core that lldb should refuse. Assert NO_ROOT_ERROR, the
        warning that tells the user why, and the condition-specific reject
        reason on the Modules log channel."""
        core_path = self.generate_artificial_core(builder, name)
        for target in list(self.dbg):
            self.dbg.DeleteTarget(target)
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid())

        warnings = self.collect_warnings()
        log_file = self.getBuildArtifact(name + ".module.log")
        self.runCmd("log enable lldb module -f '%s'" % log_file)
        error = lldb.SBError()
        try:
            process = target.LoadCore(core_path, error)
        finally:
            self.runCmd("log disable lldb module")

        self.assertFalse(process.IsValid(), f"{name} should not load")
        self.assertIn(NO_ROOT_ERROR, error.GetCString())
        self.assertIn(
            warning,
            "\n".join(warnings()),
            f"the warning that tells the user why lldb refused {name}",
        )

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
            warning=NO_HIERARCHY_WARNING,
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

    def test_table_size_that_covers_later_sections_is_rejected(self):
        """A container table whose sh_size covers the start of the section
        after it has a corrupted size. lldb refuses the core and creates no
        container for the rows that the size claims. This size also ends past
        the end of the file."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
            shsize=PAST_END_OF_FILE,
        )
        self._assert_rejected_with_log(
            b,
            "bigtable.nvcudmp",
            ["overlaps another section", "rejecting corefile"],
        )

    def test_table_size_that_ends_inside_the_file_is_rejected(self):
        """A corrupted sh_size that covers the start of the section after the
        table is refused even when it ends inside the file. The file holds
        every byte that such a size claims, but the rows past the table's own
        are the bytes of the sections after it."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
            shsize=2 * SM_ROW_SIZE,
        )
        # The section after the table is longer than the one extra row that
        # the table's size claims, so the size ends inside the file.
        b.add_raw_section(
            name=".cudbg.global.after",
            sh_type=CUDBG_SHT_GLOBAL_MEM,
            content=b"\x00" * (2 * SM_ROW_SIZE),
            address=0x100000000,
        )
        self._assert_rejected_with_log(
            b,
            "widetable.nvcudmp",
            ["overlaps another section", "rejecting corefile"],
        )

    def test_table_sharing_offset_with_later_section_is_rejected(self):
        """An empty-content table shares its sh_offset with the non-empty
        section that follows it, just as real cores place empty CTA or
        backtrace tables at the start of the next section. A corrupted sh_size
        that covers that following section must be refused even though both
        begin at the same offset."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"",
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
            shsize=2 * SM_ROW_SIZE,
        )
        # Same size as the corrupted table, so the table ends right where
        # yaml2obj's .strtab starts and only the shared-offset section sits
        # inside it.
        b.add_raw_section(
            name=".cudbg.global.after",
            sh_type=CUDBG_SHT_GLOBAL_MEM,
            content=b"\x00" * (2 * SM_ROW_SIZE),
            address=0x100000000,
        )
        self._assert_rejected_with_log(
            b,
            "sameoffset.nvcudmp",
            ["overlaps another section", "rejecting corefile"],
        )

    def test_zero_sized_table_sharing_offset_with_next_table_loads(self):
        """A valid zero-sized table shares its sh_offset with the non-empty
        table that follows it, as the driver writes for idle SMs and empty
        backtraces. It does not overlap that table and the core loads cleanly."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.empty",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"",
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
            shsize=0,
        )
        b.add_raw_section(
            name=".cudbg.smtbl.valid",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
        )
        warnings = self.collect_warnings()
        gpu_target, gpu_process = self.generate_and_load_artificial_core(
            b, "zerotable.nvcudmp"
        )
        self.assertEqual(warnings(), [], "no warning for a valid empty table")
        device = (
            gpu_target.FindModule(
                lldb.SBFileSpec(self.getBuildArtifact("zerotable.nvcudmp"))
            )
            .FindSection("nvgpucore")
            .FindSubSection("dev0")
        )
        self.assertTrue(device.IsValid(), "the device's section")
        self.assertEqual(
            device.GetNumSubSections(),
            2,
            "the scaffold SM and the non-empty table after the zero-sized one",
        )
        self.assertEqual(gpu_process.GetNumThreads(), 1)

    def test_table_starting_inside_another_section_is_rejected(self):
        """A table whose sh_offset starts inside an earlier region (here the
        64-byte ELF header, ending right where the first section begins) has a
        corrupted offset and is refused even when no section starts after it
        within its sh_size."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.smtbl.bad",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * SM_ROW_SIZE,
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
            shoffset=0x10,
        )
        self._assert_rejected_with_log(
            b,
            "badoffset.nvcudmp",
            ["overlaps another section", "rejecting corefile"],
        )

    def test_lane_table_covering_later_sections_is_rejected(self):
        """A lazily-consulted container table (device or lane table) whose
        sh_size covers later sections is also refused."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        b.add_raw_section(
            name=".cudbg.lntbl.bad",
            sh_type=CUDBG_SHT_LN_TABLE,
            content=b"\x00" * LANE_ROW_SIZE,
            link=".cudbg.wptbl.dev0.sm0.cta0",
            info=0,
            entsize=LANE_ROW_SIZE,
            shsize=PAST_END_OF_FILE,
        )
        self._assert_rejected_with_log(
            b,
            "biglanetable.nvcudmp",
            ["overlaps another section", "rejecting corefile"],
        )

    def test_table_cut_off_by_the_end_of_the_file(self):
        """A dump that was cut off inside a container table still loads. The
        table gets a container for each row that the file holds, and lldb
        warns that the core is incomplete."""
        b = NVGPUCoreBuilder()
        self._minimal(b)
        rows_written = 3
        rows_kept = 1
        b.add_raw_section(
            name=".cudbg.smtbl.cut",
            sh_type=CUDBG_SHT_SM_TABLE,
            content=b"\x00" * (rows_written * SM_ROW_SIZE),
            link=".cudbg.devtbl",
            info=0,
            entsize=SM_ROW_SIZE,
        )
        core_path = self.generate_artificial_core(b, "cut.nvcudmp", headers_first=True)
        # Only the one-byte .strtab that yaml2obj adds follows the table in
        # the file. Cut the file inside the row that follows the rows to keep.
        rows_cut = rows_written - rows_kept
        os.truncate(
            core_path,
            os.path.getsize(core_path) - rows_cut * SM_ROW_SIZE + SM_ROW_SIZE // 2,
        )
        warnings = self.collect_warnings()
        gpu_target, gpu_process = self.load_core(core_path)
        self.assertIn(
            ".cudbg.smtbl.cut",
            "\n".join(warnings()),
            "a warning that names the table that the file does not hold",
        )
        device = (
            gpu_target.FindModule(lldb.SBFileSpec(core_path))
            .FindSection("nvgpucore")
            .FindSubSection("dev0")
        )
        self.assertTrue(device.IsValid(), "the device's section")
        self.assertEqual(
            device.GetNumSubSections(),
            1 + rows_kept,
            "SM sections of the device, which are the scaffold's SM and the "
            "rows of the second table that the file holds",
        )
        self.assertEqual(gpu_process.GetNumThreads(), 1, "the scaffold's one thread")

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

    def test_missing_grid_row_keeps_lane_registers(self):
        """A CTA whose gridId64 matches no grid row costs only the two
        grid-derived registers. Everything the lane, warp, CTA and device rows
        supply survives -- in particular the PC, without which the fault site
        could not be symbolicated."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=4)
        sm = b.add_sm(
            dev, exception=WARP_ILLEGAL_INSTRUCTION, error_pc=0x100040
        )
        # No grid table is emitted at all, so this gridId64 resolves to nothing.
        cta = b.add_cta(sm, grid_id=7, block_idx=(3, 0, 0))
        warp = b.add_warp(
            cta, valid_lanes_mask=1, active_lanes_mask=1, error_pc=0x100040
        )
        lane = b.add_lane(warp, lane_id=0, thread_idx=(2, 0, 0), pc=0x100000)
        b.set_lane_registers(lane, [0xAA00, 0xAA01, 0xAA02, 0xAA03])

        _, gpu_process = self.generate_and_load_artificial_core(
            b, name="nogridrow.nvcudmp"
        )
        thread = gpu_process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)

        self.assertEqual(self.read_register(frame, "PC"), 0x100000)
        self.assertEqual(self.read_register(frame, "errorPC"), 0x100040)
        self.assertEqual(
            [self.read_register(frame, f"R{i}") for i in range(4)],
            [0xAA00, 0xAA01, 0xAA02, 0xAA03],
        )
        self.assertEqual(self.read_register(frame, "warpSize"), 32)
        self.assertEqual(
            thread.GetName(),
            "blockIdx(x=3 y=0 z=0) threadIdx(x=2 y=0 z=0)",
        )

        # The launch dimensions are the one thing the grid row would have
        # supplied, so they alone read back as zero.
        self.expect(
            "register read blockDim gridDim",
            substrs=[
                "blockDim = {0x00000000 0x00000000 0x00000000}",
                "gridDim = {0x00000000 0x00000000 0x00000000}",
            ],
        )

    def test_sparse_lanes_materialize_by_leaf(self):
        """Only lanes with a per-lane leaf become threads, even when the warp
        marks many lanes valid."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(
            dev, exception=WARP_ILLEGAL_INSTRUCTION, error_pc=0x100000
        )
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
