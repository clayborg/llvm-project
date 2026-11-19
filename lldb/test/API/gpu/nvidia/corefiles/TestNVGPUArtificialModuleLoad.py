"""
Module-load notifications while loading an NVIDIA GPU core.

A core can embed thousands of cubins, and each load notification re-probes
language runtimes and breakpoints against every loaded module, so the reader
announces all cubin modules together, once, after their load addresses are
set.
"""

import pathlib
import struct

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import NVGPUCoreBuilder

SHF_ALLOC = 0x2


def without_loadable_sections(cubin):
    """Clear SHF_ALLOC on every section and drop the program headers, so the
    cubin still parses but none of its sections gets a load address."""
    b = bytearray(cubin)
    struct.pack_into("<H", b, 0x38, 0)  # e_phnum
    (shoff,) = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x3A)
    for i in range(shnum):
        flags_offset = shoff + i * shentsize + 8
        (flags,) = struct.unpack_from("<Q", b, flags_offset)
        struct.pack_into("<Q", b, flags_offset, flags & ~SHF_ALLOC)
    return bytes(b)


class TestNVGPUArtificialModuleLoad(NVGPUCoreTestCaseBase):
    CUBIN_PATHS = ("../disass/elfv7.cubin", "../disass/elfv8.cubin")
    LANE_PC = 0x00007FFFCF280300  # acosf in elfv8.cubin

    def _builder(self):
        """A one-thread core embedding both cubin fixtures, plus a copy of
        one that has no loadable sections."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(dev, sm_id=0)
        cta = b.add_cta(sm, block_idx=(0, 0, 0))
        warp = b.add_warp(cta, valid_lanes_mask=1, active_lanes_mask=1)
        lane = b.add_lane(warp, lane_id=0, pc=self.LANE_PC)
        b.set_lane_registers(lane, [0] * 32)
        cubins = [
            pathlib.Path(self.getSourcePath(path)).read_bytes()
            for path in self.CUBIN_PATHS
        ]
        for cubin in cubins:
            b.add_relocated_cubin(cubin)
        b.add_relocated_cubin(without_loadable_sections(cubins[0]))
        return b

    @staticmethod
    def _is_core_module(module):
        """The core registers its own module, which holds only the synthetic
        nvgpucore section tree."""
        return (
            module.GetNumSections() == 1
            and module.GetSectionAtIndex(0).GetName() == "nvgpucore"
        )

    def test_cubins_are_announced_once_after_loading(self):
        """All cubin modules, loadable or not, arrive in a single
        modules-loaded event."""
        listener = lldb.SBListener("module-load-events")
        listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitModulesLoaded,
        )
        target, _ = self.generate_and_load_artificial_core(
            self._builder(), name="moduleload.nvcudmp"
        )

        cubins = [m for m in target.module_iter() if not self._is_core_module(m)]
        self.assertEqual(len(cubins), len(self.CUBIN_PATHS) + 1)

        events_with_cubins = []
        event = lldb.SBEvent()
        while listener.GetNextEvent(event):
            modules = [
                lldb.SBTarget.GetModuleAtIndexFromEvent(i, event)
                for i in range(lldb.SBTarget.GetNumModulesFromEvent(event))
            ]
            if any(m in cubins for m in modules):
                events_with_cubins.append(modules)

        self.assertEqual(len(events_with_cubins), 1, "cubins announced more than once")
        for cubin in cubins:
            self.assertIn(cubin, events_with_cubins[0], "a cubin was never announced")
