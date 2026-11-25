"""
Loading an NVIDIA GPU core regenerated at the path of a core that the same
debugger loaded before.

Each cubin module is made from the cubin's bytes in its core file. A regenerated
core whose cubin has the same offset and size as before, and whose file has the
same modification time, must still get a module made from its own bytes, not
from the earlier core's.
"""

import os
import pathlib

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import (
    NVGPUCoreBuilder,
    with_inverted_code,
)


class TestNVGPUArtificialCoreRegenerated(NVGPUCoreTestCaseBase):
    CUBIN_PATH = "../disass/elfv8.cubin"
    LANE_PC = 0x00007FFFCF280300  # acosf in elfv8.cubin
    CODE_SIZE = 16  # One SASS instruction.
    CORE_NAME = "regenerated.nvcudmp"

    def _builder(self, cubin):
        """A one-thread core, stopped at LANE_PC, that embeds cubin."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(dev, sm_id=0)
        cta = b.add_cta(sm, block_idx=(0, 0, 0))
        warp = b.add_warp(cta, valid_lanes_mask=1, active_lanes_mask=1)
        lane = b.add_lane(warp, lane_id=0, pc=self.LANE_PC)
        b.set_lane_registers(lane, [0] * 32)
        b.add_relocated_cubin(cubin)
        return b

    def _code_at_lane_pc(self, target):
        """The code at LANE_PC, as the target's cubin module holds it."""
        address = target.ResolveLoadAddress(self.LANE_PC)
        section = address.GetSection()
        self.assertTrue(section.IsValid(), "no cubin section holds LANE_PC")
        data = section.GetSectionData(address.GetOffset(), self.CODE_SIZE)
        error = lldb.SBError()
        code = data.ReadRawData(error, 0, self.CODE_SIZE)
        self.assertSuccess(error)
        return code

    def test_regenerated_core_gets_its_own_cubin(self):
        """A core regenerated at the same path with the same modification
        time, whose cubin has the same size but different code, gets a cubin
        module made from the new core."""
        cubin = pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()
        core_path = self.generate_artificial_core(
            self._builder(cubin), name=self.CORE_NAME
        )
        target, _ = self.load_core(core_path)
        old_code = self._code_at_lane_pc(target)
        old_mtime_ns = os.stat(core_path).st_mtime_ns

        # Write the new core to a new file rather than over the old one, which
        # the old target's cubin module may still map. Give it the old core's
        # modification time, as a core regenerated within the file system's
        # timestamp resolution, or copied with its time kept, would have.
        os.remove(core_path)
        new_cubin = with_inverted_code(cubin, self.LANE_PC, self.CODE_SIZE)
        self.assertEqual(
            self.generate_artificial_core(
                self._builder(new_cubin), name=self.CORE_NAME
            ),
            core_path,
        )
        os.utime(core_path, ns=(old_mtime_ns, old_mtime_ns))
        self.assertEqual(os.stat(core_path).st_mtime_ns, old_mtime_ns)

        # load_core deletes the old target first.
        target, _ = self.load_core(core_path)
        self.assertEqual(
            self._code_at_lane_pc(target),
            bytes(byte ^ 0xFF for byte in old_code),
            "the regenerated core's cubin module holds the old core's code",
        )
