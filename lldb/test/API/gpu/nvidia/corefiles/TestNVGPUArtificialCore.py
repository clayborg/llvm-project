"""
Exhaustively test the nvgpu-core plugin against an artificial GPU core file.

One core (built in-memory by NVGPUCoreBuilder, serialized via yaml2obj) covers
every register across multiple threads with distinct formula-derived values,
every reader-supported memory space, multiple constant banks across grids, and
every stop-reason mechanism -- with no live GPU or CUDA runtime.
"""

import pathlib
import struct

import lldb

from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase
from lldbsuite.test.tools.gpu.nvgpu_core_builder import NVGPUCoreBuilder


class TestNVGPUArtificialCore(NVGPUCoreTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # PC points inside the elfv8.cubin fixture so the primary thread drives
    # both register reads and cubin symbolication (values from
    # ../disass/TestNVGPUDisass.py).
    CUBIN_PATH = "../disass/elfv8.cubin"
    PC = 0x00007FFFCF280300
    SYMBOL_NAME = "acosf"

    # Device-advertised register-class sizes (the bound the reader clamps to).
    NUM_R = 255  # R0..R254; RZ is the separate always-zero R255
    NUM_UR = 255  # UR0..UR254; URZ is the separate always-zero UR255
    NUM_P = 8  # P0..P7; P7 == PT, always 1
    NUM_UP = 8  # UP0..UP7; UP7 == UPT, always 1

    GLOBAL_ADDR = 0x100000000
    GLOBAL_ADDR2 = 0x100001000
    MANAGED_ADDR = 0x200000000
    MANAGED_ADDR2 = 0x200001000
    SHARED_ADDR = 0x300000
    SHARED_ADDR2 = 0x301000
    LOCAL_ADDR = 0x200000
    LOCAL_ADDR2 = 0x201000
    CONST_ADDR = 0x400000  # grid 1, bank 0
    CONST_ADDR2 = 0x401000  # grid 1, bank 1
    CONST_ADDR_G2 = 0x500000  # grid 2, bank 0
    ABSENT_ADDR = 0x100100000  # backed by nothing

    PC_T1 = 0x00007FFFCF281000
    ERROR_PC_T1 = 0x00000000ABCD1000

    # Revision-independent exception codes -> exact string; safe to assert
    # verbatim across CUDBG API revisions.
    EXACT_EXCEPTIONS = {
        4: "Warp Illegal Instruction",
        5: "Warp Out-of-range Address",
        6: "Warp Misaligned Address",
        7: "Warp Invalid Address Space",
        8: "Warp Invalid PC",
        9: "Warp Hardware Stack Overflow",
        10: "Device Illegal Address",
        12: "Warp Assert",
        14: "Warp Illegal Address",
        17: "Cluster Out-of-range Address",
        18: "Cluster Block Not Present",
        19: "Warp Stack Canary",
    }
    # Revision-gated exception codes; only the "CUDA Exception:" prefix is
    # asserted since older builds map them to "Device Unknown Exception".
    GATED_EXCEPTIONS = list(range(20, 36))

    # Per-thread value formulas; the builder fills leaves and tests assert with
    # the same formula. P7/UP7 are the always-1 constant predicates (PT/UPT).

    @staticmethod
    def reg_value(t, i):
        return 0x0A000000 | (t << 16) | i

    @staticmethod
    def ureg_value(t, i):
        return 0x0B000000 | (t << 16) | i

    @staticmethod
    def pred_value(t, i):
        if i == 7:
            return 1
        return 1 if (i + t) % 2 == 0 else 0

    @staticmethod
    def upred_value(t, i):
        if i == 7:
            return 1
        return 1 if (i // 2 + t) % 2 == 0 else 0

    @staticmethod
    def _thread_name(block, thread):
        return (
            f"blockIdx(x={block[0]} y={block[1]} z={block[2]}) "
            f"threadIdx(x={thread[0]} y={thread[1]} z={thread[2]})"
        )

    def _fill_register_thread(self, b, lane, warp, t):
        """Fill a lane and its single-lane warp with full register leaves."""
        b.set_lane_registers(lane, [self.reg_value(t, i) for i in range(self.NUM_R)])
        b.set_warp_uniform_registers(
            warp, [self.ureg_value(t, i) for i in range(self.NUM_UR)]
        )
        b.set_lane_predicates(lane, [self.pred_value(t, i) for i in range(self.NUM_P)])
        b.set_warp_uniform_predicates(
            warp, [self.upred_value(t, i) for i in range(self.NUM_UP)]
        )

    def _build_full_core(self):
        """Two register-bearing threads across two grids, a fan of lanes
        covering every exception code, a trap warp, and a no-reason lane, plus
        multiple memory sections and constant banks."""
        cubin = pathlib.Path(self.getSourcePath(self.CUBIN_PATH)).read_bytes()

        b = NVGPUCoreBuilder()
        dev = b.add_device(
            sm_major=8,
            sm_minor=0,
            num_regs_per_lane=self.NUM_R,
            num_predicates_per_lane=self.NUM_P,
            num_uniform_regs_per_warp=self.NUM_UR,
            num_uniform_predicates_per_warp=self.NUM_UP,
        )
        # SM exception 4 drives the "borrow from SM" stop reason for active
        # lanes on a warp whose errorPC is valid.
        sm = b.add_sm(dev, exception=4, error_pc=self.PC)

        # Primary thread (grid 1): cubin anchor and SM-borrow stop case.
        primary_block = (3, 0, 0)
        primary_thread = (2, 0, 0)
        cta0 = b.add_cta(sm, grid_id=1, block_idx=primary_block)
        warp0 = b.add_warp(
            cta0, valid_lanes_mask=1, active_lanes_mask=1, error_pc=self.PC
        )
        lane0 = b.add_lane(warp0, lane_id=0, thread_idx=primary_thread, pc=self.PC)
        self._fill_register_thread(b, lane0, warp0, t=0)
        b.add_shared_memory(
            cta0, self.SHARED_ADDR, struct.pack("<II", 0xABCD0000, 0xABCD0001)
        )
        b.add_local_memory(lane0, self.LOCAL_ADDR, struct.pack("<I", 0xFEEDFACE))

        # Second register-bearing thread in grid 2 (for per-grid const-bank
        # isolation), with its own warp carrying distinct uniforms and errorPC.
        t1_block = (1, 0, 0)
        t1_thread = (5, 1, 0)
        cta1 = b.add_cta(sm, grid_id=2, block_idx=t1_block)
        warp1 = b.add_warp(
            cta1, valid_lanes_mask=1, active_lanes_mask=1, error_pc=self.ERROR_PC_T1
        )
        lane1 = b.add_lane(warp1, lane_id=0, thread_idx=t1_thread, pc=self.PC_T1)
        self._fill_register_thread(b, lane1, warp1, t=1)
        b.add_shared_memory(
            cta1, self.SHARED_ADDR2, struct.pack("<II", 0xBEEF0000, 0xBEEF0001)
        )
        b.add_local_memory(lane1, self.LOCAL_ADDR2, struct.pack("<I", 0xFEED1111))

        self.reg_threads = [
            {
                "name": self._thread_name(primary_block, primary_thread),
                "t": 0,
                "pc": self.PC,
                "error_pc": self.PC,
            },
            {
                "name": self._thread_name(t1_block, t1_thread),
                "t": 1,
                "pc": self.PC_T1,
                "error_pc": self.ERROR_PC_T1,
            },
        ]

        # One warp whose lanes each carry a definitive per-lane exception code,
        # plus one quiet lane. The primary thread's stop comes from SM-borrow.
        self.stop_threads = []
        self.stop_threads.append(
            {
                "name": self.reg_threads[0]["name"],
                "reason": lldb.eStopReasonException,
                "substr": "CUDA Exception: Warp Illegal Instruction",
            }
        )

        exc_lanes = [(code, True) for code in self.EXACT_EXCEPTIONS] + [
            (code, False) for code in self.GATED_EXCEPTIONS
        ]
        num_exc = len(exc_lanes)
        none_lane_id = num_exc
        valid_mask = (1 << (num_exc + 1)) - 1
        warp_exc = b.add_warp(
            cta1, valid_lanes_mask=valid_mask, active_lanes_mask=valid_mask
        )
        for lane_id, (code, exact) in enumerate(exc_lanes):
            thread_idx = (lane_id, 3, 0)
            lane = b.add_lane(
                warp_exc, lane_id=lane_id, thread_idx=thread_idx, exception=code
            )
            # A lane only materializes a thread if it has a per-lane leaf.
            b.set_lane_registers(lane, [0])
            if exact:
                substr = "CUDA Exception: " + self.EXACT_EXCEPTIONS[code]
            else:
                substr = "CUDA Exception:"
            self.stop_threads.append(
                {
                    "name": self._thread_name(t1_block, thread_idx),
                    "reason": lldb.eStopReasonException,
                    "substr": substr,
                }
            )
        # No exception, no errorPC, unbroken warp -> no stop reason.
        none_thread = (0, 5, 0)
        none_lane = b.add_lane(
            warp_exc, lane_id=none_lane_id, thread_idx=none_thread
        )
        b.set_lane_registers(none_lane, [0])
        self.stop_threads.append(
            {
                "name": self._thread_name(t1_block, none_thread),
                "reason": lldb.eStopReasonNone,
                "substr": None,
            }
        )

        # An inline trap surfaces as isWarpBroken -> SIGTRAP "trap".
        trap_block = (2, 0, 0)
        trap_thread = (0, 2, 0)
        cta2 = b.add_cta(sm, grid_id=1, block_idx=trap_block)
        warp_trap = b.add_warp(
            cta2, valid_lanes_mask=1, active_lanes_mask=1, is_warp_broken=True
        )
        trap_lane = b.add_lane(warp_trap, lane_id=0, thread_idx=trap_thread)
        b.set_lane_registers(trap_lane, [0])
        self.stop_threads.append(
            {
                "name": self._thread_name(trap_block, trap_thread),
                "reason": lldb.eStopReasonSignal,
                "substr": "trap",
            }
        )

        self.expected_num_threads = len(self.reg_threads) + num_exc + 2

        # Two banks in grid 1, one in grid 2, each backed by global memory at
        # its address. Const reads resolve the selected thread's CTA -> grid.
        grid1 = b.add_grid(dev, grid_id=1)
        b.add_constbank(grid1, addr=self.CONST_ADDR, size=8, bank_id=0)
        b.add_constbank(grid1, addr=self.CONST_ADDR2, size=8, bank_id=1)
        grid2 = b.add_grid(dev, grid_id=2)
        b.add_constbank(grid2, addr=self.CONST_ADDR_G2, size=8, bank_id=0)
        b.add_global_memory(self.CONST_ADDR, struct.pack("<II", 0xC0FFEE00, 0xC0FFEE01))
        b.add_global_memory(
            self.CONST_ADDR2, struct.pack("<II", 0xC0FFEE10, 0xC0FFEE11)
        )
        b.add_global_memory(
            self.CONST_ADDR_G2, struct.pack("<II", 0x6710D000, 0x6710D001)
        )

        b.add_relocated_cubin(cubin)
        b.add_global_memory(self.GLOBAL_ADDR, struct.pack("<II", 0xDEADBEEF, 0x1))
        b.add_global_memory(self.GLOBAL_ADDR2, struct.pack("<II", 0xD15EA5E0, 0x2))
        b.add_managed_memory(self.MANAGED_ADDR, struct.pack("<II", 0xCAFEBABE, 0x2))
        b.add_managed_memory(self.MANAGED_ADDR2, struct.pack("<II", 0xCAFED00D, 0x3))
        return b

    def setUp(self):
        NVGPUCoreTestBase.setUp(self)
        self.core_target, self.core_process = (
            self.generate_and_load_artificial_core(self._build_full_core())
        )
        self.core_process.SetSelectedThread(self.core_process.GetThreadAtIndex(0))

    def _thread_by_name(self, name):
        for index in range(self.core_process.GetNumThreads()):
            thread = self.core_process.GetThreadAtIndex(index)
            if thread.GetName() == name:
                return thread
        self.fail(f"no thread named {name!r}")

    def _frame_for(self, name):
        return self._thread_by_name(name).GetFrameAtIndex(0)

    def _reg(self, frame, name):
        reg = frame.FindRegister(name)
        self.assertTrue(reg.IsValid(), f"{name} should be valid")
        return reg.GetValueAsUnsigned()

    def _select(self, name):
        self.core_process.SetSelectedThread(self._thread_by_name(name))

    def test_load_and_plugin(self):
        """Loads as an nvgpu-core process with one thread per valid lane."""
        self.assertEqual(self.core_process.GetPluginName(), "nvgpu-core")
        self.assertEqual(self.core_process.GetState(), lldb.eStateStopped)
        self.assertEqual(
            self.core_process.GetNumThreads(), self.expected_num_threads
        )
        self.assertIn("nvptx", self.core_target.GetTriple())

    def test_all_general_registers(self):
        """R0-R254 read back per-thread formula values; RZ reads 0."""
        for spec in self.reg_threads:
            frame = self._frame_for(spec["name"])
            t = spec["t"]
            for i in range(self.NUM_R):
                self.assertEqual(
                    self._reg(frame, f"R{i}"),
                    self.reg_value(t, i),
                    f"R{i} in thread t={t}",
                )
            self.assertEqual(self._reg(frame, "RZ"), 0, f"RZ in thread t={t}")

    def test_register_aliases_and_composite(self):
        """SP/FP alias R1/R2, RA is the R20:R21 composite, PC/errorPC come from
        the lane/warp rows -- all distinct per thread."""
        for spec in self.reg_threads:
            frame = self._frame_for(spec["name"])
            t = spec["t"]
            self.assertEqual(self._reg(frame, "SP"), self.reg_value(t, 1))
            self.assertEqual(self._reg(frame, "FP"), self.reg_value(t, 2))
            self.assertEqual(
                self._reg(frame, "RA"),
                self.reg_value(t, 20) | (self.reg_value(t, 21) << 32),
            )
            self.assertEqual(self._reg(frame, "PC"), spec["pc"])
            self.assertEqual(self._reg(frame, "errorPC"), spec["error_pc"])

    def test_all_uniform_registers(self):
        """UR0-UR254 read back per-thread formula values; URZ reads 0."""
        for spec in self.reg_threads:
            frame = self._frame_for(spec["name"])
            t = spec["t"]
            for i in range(self.NUM_UR):
                self.assertEqual(
                    self._reg(frame, f"UR{i}"),
                    self.ureg_value(t, i),
                    f"UR{i} in thread t={t}",
                )
            self.assertEqual(self._reg(frame, "URZ"), 0, f"URZ in thread t={t}")

    def test_all_predicates(self):
        """P0-P6 follow the per-thread pattern; P7 (PT) is always 1."""
        for spec in self.reg_threads:
            frame = self._frame_for(spec["name"])
            t = spec["t"]
            for i in range(self.NUM_P):
                self.assertEqual(
                    self._reg(frame, f"P{i}"),
                    self.pred_value(t, i),
                    f"P{i} in thread t={t}",
                )
            self.assertEqual(self._reg(frame, "P7"), 1, "P7 (PT) is always 1")

    def test_all_uniform_predicates(self):
        """UP0-UP6 follow the per-thread pattern; UP7 (UPT) is always 1."""
        for spec in self.reg_threads:
            frame = self._frame_for(spec["name"])
            t = spec["t"]
            for i in range(self.NUM_UP):
                self.assertEqual(
                    self._reg(frame, f"UP{i}"),
                    self.upred_value(t, i),
                    f"UP{i} in thread t={t}",
                )
            self.assertEqual(self._reg(frame, "UP7"), 1, "UP7 (UPT) is always 1")

    def test_all_stop_reasons(self):
        """Per-lane exceptions over every code, the SM-borrow path, an inline
        trap (SIGTRAP), and a quiet lane with no stop reason; each thread is
        located by its coordinate-derived name."""
        for entry in self.stop_threads:
            thread = self._thread_by_name(entry["name"])
            self.assertEqual(
                thread.GetStopReason(),
                entry["reason"],
                f"stop reason for {entry['name']}",
            )
            if entry["substr"] is not None:
                self.assertIn(entry["substr"], thread.GetStopDescription(256))

    def test_cubin_symbolication(self):
        """The embedded relocated cubin symbolicates the primary thread's PC."""
        self.assertGreater(
            self.core_target.GetNumModules(), 0, "Core should load a cubin module"
        )
        frame = self._frame_for(self.reg_threads[0]["name"])
        self.assertEqual(frame.GetFunctionName(), self.SYMBOL_NAME)
        self.expect(f"image lookup -a {self.PC:#x}", substrs=[self.SYMBOL_NAME])
        self.expect(
            f"disassemble -a {self.PC:#x}", substrs=[self.SYMBOL_NAME, "MOV"]
        )

    def test_multiple_memory_sections(self):
        """Multiple global/managed sections plus per-thread shared/local; an
        absent address fails with the reader's diagnostic."""
        self.expect(
            f"memory read {self.GLOBAL_ADDR:#x} --format x --size 4 -c 2",
            substrs=["0xdeadbeef 0x00000001"],
        )
        self.expect(
            f"memory read {self.GLOBAL_ADDR2:#x} --format x --size 4 -c 2",
            substrs=["0xd15ea5e0 0x00000002"],
        )
        self.expect(
            f"memory read {self.MANAGED_ADDR:#x} --format x --size 4 -c 2",
            substrs=["0xcafebabe 0x00000002"],
        )
        self.expect(
            f"memory read {self.MANAGED_ADDR2:#x} --format x --size 4 -c 2",
            substrs=["0xcafed00d 0x00000003"],
        )

        # Shared/local resolve against the selected thread.
        self._select(self.reg_threads[0]["name"])
        self.expect(
            f"memory read -p shared {self.SHARED_ADDR:#x} --format x --size 4 -c 2",
            substrs=["0xabcd0000 0xabcd0001"],
        )
        self.expect(
            f"memory read -p local {self.LOCAL_ADDR:#x} --format x --size 4 -c 1",
            substrs=["0xfeedface"],
        )
        self._select(self.reg_threads[1]["name"])
        self.expect(
            f"memory read -p shared {self.SHARED_ADDR2:#x} --format x --size 4 -c 2",
            substrs=["0xbeef0000 0xbeef0001"],
        )
        self.expect(
            f"memory read -p local {self.LOCAL_ADDR2:#x} --format x --size 4 -c 1",
            substrs=["0xfeed1111"],
        )

        self.expect(
            f"memory read {self.ABSENT_ADDR:#x} --format x --size 4 -c 1",
            error=True,
            substrs=["core file does not contain"],
        )

    def test_multiple_const_banks_and_grids(self):
        """Constant banks are grid-scoped: a grid-1 thread sees its two banks
        but not grid 2's bank (and vice-versa); past-the-bank addresses fail."""
        self._select(self.reg_threads[0]["name"])  # grid 1
        self.expect(
            f"memory read -p const {self.CONST_ADDR:#x} --format x --size 4 -c 2",
            substrs=["0xc0ffee00 0xc0ffee01"],
        )
        self.expect(
            f"memory read -p const {self.CONST_ADDR2:#x} --format x --size 4 -c 2",
            substrs=["0xc0ffee10 0xc0ffee11"],
        )
        self.expect(
            f"memory read -p const {self.CONST_ADDR + 8:#x} --format x --size 4 -c 1",
            error=True,
            substrs=["is not within any constant bank"],
        )
        self.expect(
            f"memory read -p const {self.CONST_ADDR_G2:#x} --format x --size 4 -c 1",
            error=True,
            substrs=["is not within any constant bank"],
        )

        self._select(self.reg_threads[1]["name"])  # grid 2
        self.expect(
            f"memory read -p const {self.CONST_ADDR_G2:#x} --format x --size 4 -c 2",
            substrs=["0x6710d000 0x6710d001"],
        )
        self.expect(
            f"memory read -p const {self.CONST_ADDR:#x} --format x --size 4 -c 1",
            error=True,
            substrs=["is not within any constant bank"],
        )
