"""
Procedural builder for artificial NVGPU (EM_CUDA + ET_CORE) core files.

The builder keeps a small in-memory model of a GPU core -- devices, SMs,
CTAs, warps, lanes, their register/predicate/memory leaves, and embedded
cubin images -- and serializes it to an ELF YAML document that ``yaml2obj``
turns into a ``.nvcudmp``. The result is consumed by the real ``nvgpu-core``
LLDB process plugin, so tests exercise ``ObjectFileELF::BuildNVGPUSectionList``
and the ``ProcessNVGPUCore`` reader without a live GPU or CUDA runtime.

Row layouts mirror the ``Cudbg*TableEntry`` structs decoded by
``lldb/source/Plugins/Process/nvgpu-core/CudbgEntryParser.cpp``. Fields the
decoder reads past the emitted bytes read back as zero, so a packer may emit
only the prefix it needs (e.g. ``pack_grid_row`` writes just the 8-byte
``gridId64``). A test needing non-zero later fields must extend the packer.

The synthetic section tree the reader expects (and that ``write_yaml`` wires
up via ``sh_link`` / ``sh_info``)::

    nvgpucore
      devN                              CUDBG_SHT_DEV_TABLE
        gridN                           CUDBG_SHT_GRID_TABLE
          constbank
        smN                             CUDBG_SHT_SM_TABLE
          ctaN                          CUDBG_SHT_CTA_TABLE
            shared
            warpN                       CUDBG_SHT_WP_TABLE
              uregs, upreds
              laneN                     CUDBG_SHT_LN_TABLE
                regs, preds, local
      global, managed, cubin, ucubin, metadata
"""

import binascii
import struct
from dataclasses import dataclass, field
from typing import Optional

# CUDA debugger section types (SHT_LOUSER + n). Written numerically in YAML.
SHT_LOUSER            = 0x80000000
CUDBG_SHT_MANAGED_MEM = SHT_LOUSER + 1
CUDBG_SHT_GLOBAL_MEM  = SHT_LOUSER + 2
CUDBG_SHT_LOCAL_MEM   = SHT_LOUSER + 3
CUDBG_SHT_SHARED_MEM  = SHT_LOUSER + 4
CUDBG_SHT_DEV_REGS    = SHT_LOUSER + 5
CUDBG_SHT_ELF_IMG     = SHT_LOUSER + 6
CUDBG_SHT_RELF_IMG    = SHT_LOUSER + 7
CUDBG_SHT_BT          = SHT_LOUSER + 8
CUDBG_SHT_DEV_TABLE   = SHT_LOUSER + 9
CUDBG_SHT_CTX_TABLE   = SHT_LOUSER + 10
CUDBG_SHT_SM_TABLE    = SHT_LOUSER + 11
CUDBG_SHT_GRID_TABLE  = SHT_LOUSER + 12
CUDBG_SHT_CTA_TABLE   = SHT_LOUSER + 13
CUDBG_SHT_WP_TABLE    = SHT_LOUSER + 14
CUDBG_SHT_LN_TABLE    = SHT_LOUSER + 15
CUDBG_SHT_MOD_TABLE   = SHT_LOUSER + 16
CUDBG_SHT_DEV_PRED    = SHT_LOUSER + 17
CUDBG_SHT_PARAM_MEM   = SHT_LOUSER + 18
CUDBG_SHT_DEV_UREGS   = SHT_LOUSER + 19
CUDBG_SHT_DEV_UPRED   = SHT_LOUSER + 20
CUDBG_SHT_CB_TABLE    = SHT_LOUSER + 21
CUDBG_SHT_META_DATA   = SHT_LOUSER + 22
CUDBG_SHT_CBU_BAR     = SHT_LOUSER + 23

# Decoded row sizes (bytes), matching CudbgEntryParser.cpp at the latest CUDBG
# API revision. These double as the section EntSize (row stride).
DEVICE_ROW_SIZE    = 84
SM_ROW_SIZE        = 48
CTA_ROW_SIZE       = 52
WARP_ROW_SIZE      = 80
LANE_ROW_SIZE      = 64
CONSTBANK_ROW_SIZE = 16
GRID_ROW_SIZE      = 8
META_ROW_SIZE      = 24

# Default driver/CUDA toolkit version stamped into generated core files.
DEFAULT_DRIVER_BRANCH = 580
DEFAULT_CUDA_MAJOR    = 13
DEFAULT_CUDA_MINOR    = 0


def _hex(data):
    """Hex-encode bytes for an ELF YAML ``Content`` field."""
    return binascii.hexlify(data).decode("ascii")


def u32_words(words):
    """Pack an iterable of ints into little-endian uint32 words."""
    words = tuple(words)
    return struct.pack("<" + "I" * len(words), *words)


def _padded(row, size):
    """Zero-fill a row prefix out to its full decoded size."""
    return row.ljust(size, b"\x00")


def _table(rows, size):
    """Concatenate rows, padding each to a fixed stride."""
    return b"".join(_padded(row, size) for row in rows)


def pack_metadata_row(
    *,
    driver_branch=DEFAULT_DRIVER_BRANCH,
    cuda_major=DEFAULT_CUDA_MAJOR,
    cuda_minor=DEFAULT_CUDA_MINOR,
):
    return struct.pack(
        "<QIIII",
        0,  # generatorName
        driver_branch,
        0,  # driverVersionMinor
        cuda_major,
        cuda_minor,
    )


@dataclass
class _Section:
    """A single ELF section to emit. ``link`` is a section name (resolved to
    an index by yaml2obj) or None for ``sh_link == 0``. ``shsize`` overrides
    the recorded sh_size (ELFYAML ShSize), e.g. to claim a section extends
    beyond its actual content / the file for truncation tests."""

    name: str
    sh_type: int
    content: str
    link: Optional[str] = None
    info: int = 0
    address: Optional[int] = None
    entsize: Optional[int] = None
    shsize: Optional[int] = None


@dataclass
class _Device:
    idx: int
    row_bytes: bytes
    num_regs_per_lane: int = 0
    sms: list = field(default_factory=list)
    grids: list = field(default_factory=list)

    @property
    def tag(self):
        return f"dev{self.idx}"


@dataclass
class _SM:
    device: _Device
    row_index: int
    row_bytes: bytes
    ctas: list = field(default_factory=list)

    @property
    def tag(self):
        return f"{self.device.tag}.sm{self.row_index}"


@dataclass
class _CTA:
    sm: _SM
    row_index: int
    row_bytes: bytes
    warps: list = field(default_factory=list)
    shared: Optional[tuple] = None  # (address, data)

    @property
    def tag(self):
        return f"{self.sm.tag}.cta{self.row_index}"


@dataclass
class _Warp:
    cta: _CTA
    row_index: int
    row_bytes: bytes
    lanes: dict = field(default_factory=dict)  # lane_id -> _Lane
    uregs: Optional[bytes] = None
    upreds: Optional[bytes] = None

    @property
    def tag(self):
        return f"{self.cta.tag}.wp{self.row_index}"


@dataclass
class _Lane:
    row_bytes: bytes
    regs: Optional[bytes] = None
    preds: Optional[bytes] = None
    local: Optional[tuple] = None  # (address, data)


@dataclass
class _Grid:
    row_index: int
    row_bytes: bytes
    constbanks: list = field(default_factory=list)  # list of row_bytes


class NVGPUCoreBuilder:
    """Builds an artificial NVGPU core file. See module docstring."""

    def __init__(self):
        self.devices = []
        self.global_mem = []  # list of (address, data, name)
        self.managed_mem = []
        self.relocated_cubins = []  # list of (bytes, name)
        self.unrelocated_cubins = []
        self.metadata = pack_metadata_row()  # raw bytes, or None to omit
        self.raw_sections = []  # list of _Section escape-hatch entries

    # -- hierarchy ---------------------------------------------------------

    def add_device(
        self,
        *,
        sm_major=8,
        sm_minor=0,
        num_sms=1,
        num_warps_per_sm=1,
        num_lanes_per_warp=32,
        num_regs_per_lane=256,
        num_predicates_per_lane=8,
        num_uniform_regs_per_warp=0,
        num_uniform_predicates_per_warp=0,
        instruction_size=16,
    ):
        dev_id = len(self.devices)
        row = struct.pack(
            "<QQQ" + "I" * 15,
            0,  # devName (string-table index; unused)
            0,  # devType (string-table index; unused)
            0,  # smType (string-table index; unused)
            dev_id,
            0,  # pciBusId
            0,  # pciDevId
            num_sms,
            num_warps_per_sm,
            num_lanes_per_warp,
            num_regs_per_lane,
            num_predicates_per_lane,
            sm_major,
            sm_minor,
            instruction_size,
            0,  # status
            num_uniform_regs_per_warp,
            num_uniform_predicates_per_warp,
            0,  # numConvergenceBarriersPrWarp
        )
        # Remember the advertised register count so warp rows can default
        # numRegs to it.
        dev = _Device(dev_id, row, num_regs_per_lane=num_regs_per_lane)
        self.devices.append(dev)
        return dev

    def add_sm(self, device, *, sm_id=0, exception=0, error_pc=None):
        row = struct.pack(
            "<IIIIQ",
            sm_id,
            0,  # padding0
            exception,
            0 if error_pc is None else 1,  # errorPCValid
            0 if error_pc is None else error_pc,
        )
        sm = _SM(device, len(device.sms), row)
        device.sms.append(sm)
        return sm

    def add_grid(self, device, *, grid_id=1):
        grid = _Grid(len(device.grids), struct.pack("<Q", grid_id))
        device.grids.append(grid)
        return grid

    def add_cta(self, sm, *, grid_id=1, block_idx=(0, 0, 0)):
        row = struct.pack(
            "<QIII",
            grid_id,
            block_idx[0],
            block_idx[1],
            block_idx[2],
        )
        cta = _CTA(sm, len(sm.ctas), row)
        sm.ctas.append(cta)
        return cta

    def add_warp(
        self,
        cta,
        *,
        warp_id=0,
        valid_lanes_mask=1,
        active_lanes_mask=1,
        error_pc=None,
        is_warp_broken=False,
        num_regs=None,
    ):
        if num_regs is None:
            num_regs = cta.sm.device.num_regs_per_lane
        row = struct.pack(
            "<QIIIIIII",
            0 if error_pc is None else error_pc,
            warp_id,
            valid_lanes_mask,
            active_lanes_mask,
            1 if is_warp_broken else 0,
            0 if error_pc is None else 1,  # errorPCValid
            0,  # padding0
            num_regs,
        )
        warp = _Warp(cta, len(cta.warps), row)
        cta.warps.append(warp)
        return warp

    def add_lane(
        self,
        warp,
        *,
        lane_id=0,
        thread_idx=(0, 0, 0),
        pc=0,
        exception=0,
        call_depth=1,
    ):
        row = struct.pack(
            "<QQIIIIII",
            pc,  # virtualPC
            pc,  # physPC
            lane_id,
            thread_idx[0],
            thread_idx[1],
            thread_idx[2],
            exception,
            call_depth,
        )
        lane = _Lane(row)
        warp.lanes[lane_id] = lane
        return lane

    # -- per-lane / per-warp register and predicate leaves -----------------

    def set_lane_registers(self, lane, words):
        lane.regs = u32_words(words)

    def set_lane_predicates(self, lane, words):
        lane.preds = u32_words(words)

    def set_warp_uniform_registers(self, warp, words):
        warp.uregs = u32_words(words)

    def set_warp_uniform_predicates(self, warp, words):
        warp.upreds = u32_words(words)

    # -- memory ------------------------------------------------------------

    def add_global_memory(self, addr, data, name=None):
        self.global_mem.append((addr, bytes(data), name))

    def add_managed_memory(self, addr, data, name=None):
        self.managed_mem.append((addr, bytes(data), name))

    def add_local_memory(self, lane, addr, data):
        lane.local = (addr, bytes(data))

    def add_shared_memory(self, cta, addr, data):
        cta.shared = (addr, bytes(data))

    # -- grid constant banks ----------------------------------------------

    def add_constbank(self, grid, *, addr, size, bank_id=0):
        grid.constbanks.append(struct.pack("<QII", addr, size, bank_id))

    # -- images ------------------------------------------------------------

    def add_relocated_cubin(self, cubin_bytes, name=None):
        self.relocated_cubins.append((bytes(cubin_bytes), name))

    def add_unrelocated_cubin(self, cubin_bytes, name=None):
        self.unrelocated_cubins.append((bytes(cubin_bytes), name))

    def set_metadata(self, data):
        self.metadata = None if data is None else bytes(data)

    def set_metadata_version(
        self,
        *,
        driver_branch=DEFAULT_DRIVER_BRANCH,
        cuda_major=DEFAULT_CUDA_MAJOR,
        cuda_minor=DEFAULT_CUDA_MINOR,
    ):
        self.metadata = pack_metadata_row(
            driver_branch=driver_branch,
            cuda_major=cuda_major,
            cuda_minor=cuda_minor,
        )

    # -- escape hatch ------------------------------------------------------

    def add_raw_section(self, *, name, sh_type, content, link=None, info=0,
                        address=None, entsize=None, shsize=None):
        """Append an arbitrary section verbatim. ``content`` may be bytes or a
        hex string. ``shsize`` overrides the recorded sh_size (e.g. to claim
        the section extends past the file for truncation tests). Useful for
        negative / truncation / unsupported-section tests."""
        if isinstance(content, bytes):
            content = _hex(content)
        self.raw_sections.append(
            _Section(name, sh_type, content, link=link, info=info,
                     address=address, entsize=entsize, shsize=shsize)
        )

    # -- serialization -----------------------------------------------------

    def _build_sections(self):
        """Flatten the model into an ordered list of _Section objects with
        sh_link / sh_info wired up per the synthetic-tree contract."""
        sections = []

        def emit(name, sht, data, *, link=None, info=0, address=None,
                 entsize=None):
            if data is None:
                return
            sections.append(
                _Section(name, sht, _hex(data), link=link, info=info,
                         address=address, entsize=entsize)
            )

        emit(".cudbg.meta", CUDBG_SHT_META_DATA, self.metadata,
             entsize=META_ROW_SIZE)

        # One device table, one row per device.
        devtbl = ".cudbg.devtbl"
        if self.devices:
            emit(devtbl, CUDBG_SHT_DEV_TABLE,
                 _table((d.row_bytes for d in self.devices), DEVICE_ROW_SIZE),
                 entsize=DEVICE_ROW_SIZE)

        for dev in self.devices:
            self._emit_device(emit, dev, devtbl)

        # Root-level memory and image leaves (sh_link == 0).
        for i, (addr, data, name) in enumerate(self.global_mem):
            emit(name or f".cudbg.global.{i}", CUDBG_SHT_GLOBAL_MEM, data,
                 address=addr)
        for i, (addr, data, name) in enumerate(self.managed_mem):
            emit(name or f".cudbg.managed.{i}", CUDBG_SHT_MANAGED_MEM, data,
                 address=addr)
        for i, (data, name) in enumerate(self.relocated_cubins):
            emit(name or f".cudbg.relfimg.{i}", CUDBG_SHT_RELF_IMG, data)
        for i, (data, name) in enumerate(self.unrelocated_cubins):
            emit(name or f".cudbg.elfimg.{i}", CUDBG_SHT_ELF_IMG, data)

        sections.extend(self.raw_sections)
        return sections

    def _emit_device(self, emit, dev, devtbl):
        """Emit the grid subtree and SM table for one device."""
        if dev.grids:
            self._emit_grids(emit, dev, devtbl)

        if not dev.sms:
            return

        smtbl = f".cudbg.smtbl.{dev.tag}"
        emit(smtbl, CUDBG_SHT_SM_TABLE,
             _table((sm.row_bytes for sm in dev.sms), SM_ROW_SIZE),
             link=devtbl, info=dev.idx, entsize=SM_ROW_SIZE)

        for sm in dev.sms:
            self._emit_sm(emit, sm, smtbl)

    def _emit_grids(self, emit, dev, devtbl):
        """Emit a device's grid table (sibling of the SM subtree) and the
        constant-bank tables hanging off it."""
        gridtbl = f".cudbg.gridtbl.{dev.tag}"
        emit(gridtbl, CUDBG_SHT_GRID_TABLE,
             _table((g.row_bytes for g in dev.grids), GRID_ROW_SIZE),
             link=devtbl, info=dev.idx, entsize=GRID_ROW_SIZE)
        for grid in dev.grids:
            if grid.constbanks:
                emit(f".cudbg.cbtbl.{dev.tag}.grid{grid.row_index}",
                     CUDBG_SHT_CB_TABLE,
                     _table(grid.constbanks, CONSTBANK_ROW_SIZE),
                     link=gridtbl, info=grid.row_index,
                     entsize=CONSTBANK_ROW_SIZE)

    def _emit_sm(self, emit, sm, smtbl):
        """Emit the CTA table for one SM."""
        if not sm.ctas:
            return
        ctatbl = f".cudbg.ctatbl.{sm.tag}"
        emit(ctatbl, CUDBG_SHT_CTA_TABLE,
             _table((c.row_bytes for c in sm.ctas), CTA_ROW_SIZE),
             link=smtbl, info=sm.row_index, entsize=CTA_ROW_SIZE)

        for cta in sm.ctas:
            self._emit_cta(emit, cta, ctatbl)

    def _emit_cta(self, emit, cta, ctatbl):
        """Emit a CTA's shared memory leaf and warp table."""
        if cta.shared is not None:
            addr, data = cta.shared
            emit(f".cudbg.shared.{cta.tag}", CUDBG_SHT_SHARED_MEM,
                 data, link=ctatbl, info=cta.row_index, address=addr)
        if not cta.warps:
            return
        wptbl = f".cudbg.wptbl.{cta.tag}"
        emit(wptbl, CUDBG_SHT_WP_TABLE,
             _table((w.row_bytes for w in cta.warps), WARP_ROW_SIZE),
             link=ctatbl, info=cta.row_index, entsize=WARP_ROW_SIZE)

        for warp in cta.warps:
            self._emit_warp(emit, warp, wptbl)

    def _emit_warp(self, emit, warp, wptbl):
        """Emit a warp's uniform reg/pred leaves and lane table."""
        emit(f".cudbg.uregs.{warp.tag}", CUDBG_SHT_DEV_UREGS,
             warp.uregs, link=wptbl, info=warp.row_index)
        emit(f".cudbg.upreds.{warp.tag}", CUDBG_SHT_DEV_UPRED,
             warp.upreds, link=wptbl, info=warp.row_index)
        if not warp.lanes:
            return
        # Lane table sized to hold the highest lane id; rows without
        # per-lane leaves never materialize threads.
        rows = [warp.lanes[i].row_bytes if i in warp.lanes else b""
                for i in range(max(warp.lanes) + 1)]
        lntbl = f".cudbg.lntbl.{warp.tag}"
        emit(lntbl, CUDBG_SHT_LN_TABLE, _table(rows, LANE_ROW_SIZE),
             link=wptbl, info=warp.row_index, entsize=LANE_ROW_SIZE)

        for lane_id, lane in warp.lanes.items():
            self._emit_lane(emit, warp, lane_id, lane, lntbl)

    def _emit_lane(self, emit, warp, lane_id, lane, lntbl):
        """Emit a lane's register, predicate, and local-memory leaves."""
        suffix = f".{warp.tag}.ln{lane_id}"
        emit(".cudbg.regs" + suffix, CUDBG_SHT_DEV_REGS,
             lane.regs, link=lntbl, info=lane_id)
        emit(".cudbg.preds" + suffix, CUDBG_SHT_DEV_PRED,
             lane.preds, link=lntbl, info=lane_id)
        if lane.local is not None:
            addr, data = lane.local
            emit(".cudbg.local" + suffix, CUDBG_SHT_LOCAL_MEM, data,
                 link=lntbl, info=lane_id, address=addr)

    def write_yaml(self, path):
        sections = self._build_sections()
        lines = [
            "--- !ELF",
            "FileHeader:",
            "  Class:   ELFCLASS64",
            "  Data:    ELFDATA2LSB",
            "  Type:    ET_CORE",
            "  Machine: EM_CUDA",
            "Sections:",
        ]
        for sec in sections:
            lines.append(f"  - Name:    {sec.name}")
            lines.append(f"    Type:    {sec.sh_type:#x}")
            if sec.link is not None:
                lines.append(f"    Link:    {sec.link}")
                lines.append(f"    Info:    {sec.info}")
            if sec.address is not None:
                lines.append(f"    Address: {sec.address:#x}")
            if sec.entsize is not None:
                lines.append(f"    EntSize: {sec.entsize}")
            lines.append(f'    Content: "{sec.content}"')
            if sec.shsize is not None:
                lines.append(f"    ShSize:  {sec.shsize}")
        lines.append("...")
        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
