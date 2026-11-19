"""
Module-load notifications while loading an NVIDIA GPU core.

A core can embed thousands of cubins, and each load notification re-probes
language runtimes and breakpoints against every loaded module, so the reader
announces all cubin modules together, once, after their load addresses are
set.
"""

import pathlib

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testcase import NVGPUCoreTestCaseBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import NVGPUCoreBuilder


class TestNVGPUArtificialModuleLoad(NVGPUCoreTestCaseBase):
    CUBIN_PATHS = ("../disass/elfv7.cubin", "../disass/elfv8.cubin")
    LANE_PC = 0x00007FFFCF280300  # acosf in elfv8.cubin

    def _builder(self):
        """A one-thread core embedding both cubin fixtures."""
        b = NVGPUCoreBuilder()
        dev = b.add_device(num_regs_per_lane=32)
        sm = b.add_sm(dev, sm_id=0)
        cta = b.add_cta(sm, block_idx=(0, 0, 0))
        warp = b.add_warp(cta, valid_lanes_mask=1, active_lanes_mask=1)
        lane = b.add_lane(warp, lane_id=0, pc=self.LANE_PC)
        b.set_lane_registers(lane, [0] * 32)
        for path in self.CUBIN_PATHS:
            b.add_relocated_cubin(pathlib.Path(self.getSourcePath(path)).read_bytes())
        return b

    @staticmethod
    def _has_loaded_section(module, target):
        return any(
            section.GetLoadAddress(target) != lldb.LLDB_INVALID_ADDRESS
            for section in module.section_iter()
        )

    def test_cubins_are_announced_once_after_loading(self):
        """All cubin modules arrive in a single modules-loaded event, and each
        already has its load address by the time the core is loaded."""
        listener = lldb.SBListener("module-load-events")
        listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitModulesLoaded,
        )
        target, _ = self.generate_and_load_artificial_core(
            self._builder(), name="moduleload.nvcudmp"
        )

        # The core registers its own module too; it has no loadable sections.
        cubins = [
            m for m in target.module_iter() if self._has_loaded_section(m, target)
        ]
        self.assertEqual(len(cubins), len(self.CUBIN_PATHS))

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
            self.assertIn(cubin, events_with_cubins[0])
