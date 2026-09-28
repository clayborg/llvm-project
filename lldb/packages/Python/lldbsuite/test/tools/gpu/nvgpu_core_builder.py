"""
Procedural builder for artificial NVGPU (EM_CUDA + ET_CORE) core files.

The builder keeps a small in-memory model of a GPU core -- devices, SMs,
CTAs, warps, lanes, their register/predicate/memory leaves, and embedded
cubin images -- and serializes it to an ELF YAML document that ``yaml2obj``
turns into a ``.nvcudmp``. The result is consumed by the real ``nvgpu-core``
LLDB process plugin, so tests exercise ``ObjectFileELF::BuildNVGPUSectionList``
and the ``ProcessNVGPUCore`` reader without a live GPU or CUDA runtime.

Row layouts mirror the ``Cudbg*TableEntry`` structs in ``cudacoredump.h`` and
decoded by ``lldb/source/Plugins/Process/nvgpu-core/CudbgEntryParser.cpp``.
Fields a reader decodes past the emitted bytes read back as zero.

The section tree, wired up via ``sh_link`` / ``sh_info``::

    nvgpucore
      devN                              CUDBG_SHT_DEV_TABLE
        ctxN                            CUDBG_SHT_CTX_TABLE
          modN                          CUDBG_SHT_MOD_TABLE
            cubin, ucubin
        gridN                           CUDBG_SHT_GRID_TABLE
          param
          constbank
        smN                             CUDBG_SHT_SM_TABLE
          ctaN                          CUDBG_SHT_CTA_TABLE
            shared
            warpN                       CUDBG_SHT_WP_TABLE
              uregs, upreds
              laneN                     CUDBG_SHT_LN_TABLE
                regs, preds, local, bt
      strtab, global, managed, cubin, ucubin, metadata
"""

import binascii
import functools
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Optional

import lldbsuite

# CUDA debugger API headers vendored at lldb/third-party/cuda. Every value the
# SDK owns -- section types, row layouts, field offsets, enumerators, the
# version stamp -- is read out of them below rather than written down here, so
# a header update cannot leave the generated cores describing a format the
# reader no longer expects.
#
# A build can compile against a different copy of these headers via
# NVGPU_DEBUGGER_INCLUDE_DIR, which the test suite has no way to learn. That is
# safe in practice: these values are stable across in-major revisions, and a
# test that writes one into a core file also asserts what LLDB makes of it, so
# an SDK that disagreed would fail an assertion rather than pass quietly.
_CUDA_HEADER_DIR = os.path.join(lldbsuite.lldb_root, "third-party", "cuda")
CUDADEBUGGER_H = os.path.join(_CUDA_HEADER_DIR, "cudadebugger.h")
CUDACOREDUMP_H = os.path.join(_CUDA_HEADER_DIR, "cudacoredump.h")


@functools.lru_cache(maxsize=None)
def _header_text(path):
    """Header contents with comments removed, so the scans below cannot trip
    over prose or commented-out declarations."""
    with open(path) as header:
        text = header.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _cudbg_constant(table, key, source):
    """Look ``key`` up in something parsed out of a header, naming the
    alternatives on a miss so a typo points at the real spelling."""
    if key not in table:
        raise KeyError(
            f"{key} is not defined in {source}; known: "
            f"{', '.join(sorted(table))}"
        )
    return table[key]


def _header_define(path, name):
    """Value of an object-like #define holding an integer literal."""
    define = re.search(
        r"^\s*#\s*define\s+%s\s+(0[xX][0-9a-fA-F]+|\d+)" % re.escape(name),
        _header_text(path),
        re.M,
    )
    if define is None:
        raise AssertionError(f"no #define {name} found in {path}")
    return int(define.group(1), 0)


def _enumerator_value(expr, path):
    """Evaluate the initializer of an enumerator.

    Covers the two forms these headers use: an integer literal with optional
    width suffix, and an offset from a #define such as ``SHT_LOUSER + 11``.
    """
    literal = re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]*", expr)
    if literal:
        return int(literal.group(1), 0)
    offset_from_define = re.fullmatch(r"(\w+)\s*\+\s*(\d+)", expr)
    if offset_from_define:
        return _header_define(path, offset_from_define.group(1)) + int(
            offset_from_define.group(2)
        )
    raise AssertionError(
        f"cannot evaluate enumerator initializer '{expr}' from {path}"
    )


@functools.lru_cache(maxsize=None)
def _cudbg_enum(path, enum_name):
    """Map a named C enum's enumerators to their values.

    Enumerators without an initializer continue from the previous one, as C
    defines them, so enums that spell out only some values still resolve.
    """
    body = re.search(
        r"typedef enum\s*\{([^{}]*)\}\s*%s\s*;" % re.escape(enum_name),
        _header_text(path),
        re.S,
    )
    if body is None:
        raise AssertionError(f"no {enum_name} enum found in {path}")
    values = {}
    next_value = 0
    for enumerator in body.group(1).split(","):
        name, _, initializer = (part.strip() for part in
                                enumerator.strip().partition("="))
        if not name:
            continue
        if initializer:
            next_value = _enumerator_value(initializer, path)
        values[name] = next_value
        next_value += 1
    return values


def cudbg_faults():
    """Names of every CUDBGException_t value that is a real fault.

    Excludes the enumerators that describe the absence of one: no-exception,
    the reserved holes left by codes the SDK retired, and the catch-all. Lets
    a test cover whatever faults the SDK defines instead of a numeric range
    that silently stops short as codes are added.
    """
    prefix = "CUDBG_EXCEPTION_"
    names = (
        name.removeprefix(prefix)
        for name in _cudbg_enum(CUDADEBUGGER_H, "CUDBGException_t")
    )
    return sorted(
        name
        for name in names
        if name not in ("NONE", "UNKNOWN")
        and not name.startswith("RESERVED_")
    )


def cudbg_exception(name):
    """Value of CUDBG_EXCEPTION_<name> from the vendored cudadebugger.h.

    ``name`` is the enumerator without the CUDBG_EXCEPTION_ prefix, e.g.
    ``cudbg_exception("WARP_ILLEGAL_INSTRUCTION")``.
    """
    return _cudbg_constant(
        _cudbg_enum(CUDADEBUGGER_H, "CUDBGException_t"),
        "CUDBG_EXCEPTION_" + name,
        CUDADEBUGGER_H,
    )


# Scalar field widths the Cudbg*TableEntry structs are built from, and the
# struct-module code that writes each width little-endian. A field of any
# other kind invalidates the layout computation below, which raises rather
# than guess.
_SCALAR_WIDTHS = {"uint64_t": 8, "uint32_t": 4, "uint16_t": 2, "uint8_t": 1}
_WIDTH_CODES = {8: "<Q", 4: "<I", 2: "<H", 1: "<B"}


@functools.lru_cache(maxsize=None)
def cudbg_struct_layout(struct_name):
    """C layout of a Cudbg*TableEntry, read from cudacoredump.h.

    Returns ``(fields, size)`` where ``fields`` maps each field name to its
    ``(offset, width)``. The structs are flat runs of fixed-width scalars with
    explicit padding fields, so the layout follows from the declaration order
    alone: each field sits at the next offset aligned to its own width, and
    the total rounds up to the widest field. That reproduces sizeof exactly
    for every struct this module emits.
    """
    body = re.search(
        r"typedef struct\s*\{([^{}]*)\}\s*%s\s*;" % re.escape(struct_name),
        _header_text(CUDACOREDUMP_H),
        re.S,
    )
    if body is None:
        raise AssertionError(
            f"no {struct_name} struct found in {CUDACOREDUMP_H}"
        )
    fields = {}
    offset = 0
    alignment = 1
    for field_type, field_name, array in re.findall(
        r"(\w+)\s+(\w+)\s*(\[[^\]]*\])?\s*;", body.group(1)
    ):
        width = _SCALAR_WIDTHS.get(field_type)
        if width is None or array:
            raise AssertionError(
                f"{struct_name}.{field_name} is a '{field_type}{array}', "
                "which cudbg_struct_layout cannot lay out; teach it that "
                "field kind before relying on this layout"
            )
        offset = (offset + width - 1) // width * width
        fields[field_name] = (offset, width)
        offset += width
        alignment = max(alignment, width)
    return fields, (offset + alignment - 1) // alignment * alignment


def cudbg_row_size(struct_name):
    """sizeof(struct_name), i.e. the row stride of the table that holds it.

    Deriving this rather than hardcoding it keeps generated cores using the
    same stride as a real dump. A stale stride does not fail a test -- the
    reader bounds each row decode by the section's own EntSize -- so a struct
    that grew a field would otherwise leave the synthetic cores quietly
    unrepresentative.
    """
    return cudbg_struct_layout(struct_name)[1]


def _cudbg_field(struct_name, field_name):
    """The ``(offset, width)`` of one field, or KeyError naming the choices."""
    fields, _ = cudbg_struct_layout(struct_name)
    return _cudbg_constant(fields, field_name, struct_name)


def cudbg_field_offset(struct_name, field_name):
    """Byte offset of one field within its Cudbg*TableEntry."""
    return _cudbg_field(struct_name, field_name)[0]


def pack_row(struct_name, **values):
    """Pack one Cudbg*TableEntry from named fields, zeroing the rest.

    Each value lands at the offset the header gives its field, so a struct
    that reorders or grows cannot silently shift a value into a neighbouring
    slot the way a positional format string would, and fields a test does not
    care about need not be spelled out at all.
    """
    row = bytearray(cudbg_row_size(struct_name))
    for field_name, value in values.items():
        offset, width = _cudbg_field(struct_name, field_name)
        struct.pack_into(_WIDTH_CODES[width], row, offset, value)
    return bytes(row)


# CUDA coredump section types, read from the header's CudbgSectionHeaderTypes.
# Written numerically in YAML because yaml2obj has no names for them.
_SECTION_TYPES = _cudbg_enum(CUDACOREDUMP_H, "CudbgSectionHeaderTypes")


def cudbg_section_type(name):
    """Value of CUDBG_SHT_<name> from the vendored cudacoredump.h."""
    return _cudbg_constant(
        _SECTION_TYPES, "CUDBG_SHT_" + name, CUDACOREDUMP_H
    )


CUDBG_SHT_MANAGED_MEM = cudbg_section_type("MANAGED_MEM")
CUDBG_SHT_GLOBAL_MEM  = cudbg_section_type("GLOBAL_MEM")
CUDBG_SHT_LOCAL_MEM   = cudbg_section_type("LOCAL_MEM")
CUDBG_SHT_SHARED_MEM  = cudbg_section_type("SHARED_MEM")
CUDBG_SHT_DEV_REGS    = cudbg_section_type("DEV_REGS")
CUDBG_SHT_ELF_IMG     = cudbg_section_type("ELF_IMG")
CUDBG_SHT_RELF_IMG    = cudbg_section_type("RELF_IMG")
CUDBG_SHT_BT          = cudbg_section_type("BT")
CUDBG_SHT_DEV_TABLE   = cudbg_section_type("DEV_TABLE")
CUDBG_SHT_CTX_TABLE   = cudbg_section_type("CTX_TABLE")
CUDBG_SHT_SM_TABLE    = cudbg_section_type("SM_TABLE")
CUDBG_SHT_GRID_TABLE  = cudbg_section_type("GRID_TABLE")
CUDBG_SHT_CTA_TABLE   = cudbg_section_type("CTA_TABLE")
CUDBG_SHT_WP_TABLE    = cudbg_section_type("WP_TABLE")
CUDBG_SHT_LN_TABLE    = cudbg_section_type("LN_TABLE")
CUDBG_SHT_MOD_TABLE   = cudbg_section_type("MOD_TABLE")
CUDBG_SHT_DEV_PRED    = cudbg_section_type("DEV_PRED")
CUDBG_SHT_PARAM_MEM   = cudbg_section_type("PARAM_MEM")
CUDBG_SHT_DEV_UREGS   = cudbg_section_type("DEV_UREGS")
CUDBG_SHT_DEV_UPRED   = cudbg_section_type("DEV_UPRED")
CUDBG_SHT_CB_TABLE    = cudbg_section_type("CB_TABLE")
CUDBG_SHT_META_DATA   = cudbg_section_type("META_DATA")
CUDBG_SHT_CBU_BAR     = cudbg_section_type("CBU_BAR")

# Row sizes (bytes), one per Cudbg*TableEntry; also the section EntSize (row
# stride). Full entries are emitted so the cores read identically across tools.
# Derived from the vendored header rather than hardcoded, so a struct that
# grows a field cannot leave these behind (see cudbg_row_size).
DEVICE_ROW_SIZE    = cudbg_row_size("CudbgDeviceTableEntry")
SM_ROW_SIZE        = cudbg_row_size("CudbgSmTableEntry")
CTA_ROW_SIZE       = cudbg_row_size("CudbgCTATableEntry")
WARP_ROW_SIZE      = cudbg_row_size("CudbgWarpTableEntry")
LANE_ROW_SIZE      = cudbg_row_size("CudbgThreadTableEntry")
CONSTBANK_ROW_SIZE = cudbg_row_size("CudbgConstBankTableEntry")
GRID_ROW_SIZE      = cudbg_row_size("CudbgGridTableEntry")
CONTEXT_ROW_SIZE   = cudbg_row_size("CudbgContextTableEntry")
MODULE_ROW_SIZE    = cudbg_row_size("CudbgModuleTableEntry")
BT_ROW_SIZE        = cudbg_row_size("CudbgBacktraceTableEntry")
META_ROW_SIZE      = cudbg_row_size("CudbgMetaDataEntry")

# Offset of callDepth within CudbgThreadTableEntry, patched into a lane row
# after the fact when a backtrace is attached.
LANE_CALL_DEPTH_OFFSET = cudbg_field_offset(
    "CudbgThreadTableEntry", "callDepth"
)

# CUDBGGridStatus value for a grid running on the hardware.
CUDBG_GRID_STATUS_ACTIVE = _cudbg_constant(
    _cudbg_enum(CUDADEBUGGER_H, "CUDBGGridStatus"),
    "CUDBG_GRID_STATUS_ACTIVE",
    CUDADEBUGGER_H,
)

# CUDA coredump ELF identification, set alongside EM_CUDA + ET_CORE on a real
# core. cudacoredump.h states these in its overview prose rather than
# declaring them, and yaml2obj wants a number for OSABI regardless, so they
# are the one pair of SDK values this module still spells out. Neither affects
# how the reader identifies a core -- it keys off e_type and e_machine.
ELFOSABI_CUDA       = 0x33
CUDA_ELF_ABIVERSION = 7

# Version stamped into the metadata section of generated cores. The CUDA
# version tracks the header the plugin is built against, because
# ProcessNVGPUCore warns about a core from a different CUDA major release;
# taking it from the header keeps generated cores from tripping that warning
# the moment the SDK moves on. The driver branch is not checked by anything,
# so it stays an arbitrary plausible value.
DEFAULT_DRIVER_BRANCH = 580
DEFAULT_CUDA_MAJOR    = _header_define(
    CUDADEBUGGER_H, "CUDBG_API_VERSION_MAJOR"
)
DEFAULT_CUDA_MINOR    = _header_define(
    CUDADEBUGGER_H, "CUDBG_API_VERSION_MINOR"
)


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
    return pack_row(
        "CudbgMetaDataEntry",
        driverVersionMajor=driver_branch,
        cudaDriverVersionMajor=cuda_major,
        cudaDriverVersionMinor=cuda_minor,
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
    contexts: list = field(default_factory=list)

    @property
    def tag(self):
        return f"dev{self.idx}"


@dataclass
class _Context:
    device: _Device
    row_index: int
    context_id: int
    row_bytes: bytes
    modules: list = field(default_factory=list)

    @property
    def tag(self):
        return f"{self.device.tag}.ctx{self.row_index}"


@dataclass
class _Module:
    context: _Context
    row_index: int
    module_handle: int


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
    backtrace: Optional[bytes] = None  # packed CudbgBacktraceTableEntry rows


@dataclass
class _Grid:
    row_index: int
    row_bytes: bytes
    constbanks: list = field(default_factory=list)  # list of row_bytes
    params: Optional[bytes] = None


class NVGPUCoreBuilder:
    """Builds an artificial NVGPU core file. See module docstring."""

    def __init__(self):
        self.devices = []
        self.global_mem = []  # list of (address, data, name)
        self.managed_mem = []
        self.relocated_cubins = []  # list of (bytes, name, module)
        self.unrelocated_cubins = []
        self.metadata = pack_metadata_row()  # raw bytes, or None to omit
        self.raw_sections = []  # list of _Section escape-hatch entries
        # ELF string table (.strtab); index 0 is the empty string. Emitted only
        # when a device interns a name into it (see add_device's string args).
        self.strtab = bytearray(b"\x00")

    def _intern_string(self, text):
        """Append a NUL-terminated string to .strtab, returning its offset
        (0, the empty string, for None)."""
        if text is None:
            return 0
        offset = len(self.strtab)
        self.strtab += text.encode("ascii") + b"\x00"
        return offset

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
        sm_type=None,
        dev_name=None,
        dev_type=None,
    ):
        # devName/devType/smType are .strtab indices; smType must name a real
        # arch (e.g. "sm_80"). Passing sm_type emits the .strtab section.
        dev_name_off = self._intern_string(dev_name)
        dev_type_off = self._intern_string(dev_type)
        sm_type_off = self._intern_string(sm_type)
        dev_id = len(self.devices)
        row = pack_row(
            "CudbgDeviceTableEntry",
            devName=dev_name_off,  # string-table index
            devType=dev_type_off,  # string-table index
            smType=sm_type_off,  # string-table index
            devId=dev_id,
            numSMs=num_sms,
            numWarpsPerSM=num_warps_per_sm,
            numLanesPerWarp=num_lanes_per_warp,
            numRegsPerLane=num_regs_per_lane,
            numPredicatesPrLane=num_predicates_per_lane,
            smMajor=sm_major,
            smMinor=sm_minor,
            instructionSize=instruction_size,
            numUniformRegsPrWarp=num_uniform_regs_per_warp,
            numUniformPredicatesPrWarp=num_uniform_predicates_per_warp,
        )
        # Remember the advertised register count so warp rows can default
        # numRegs to it.
        dev = _Device(dev_id, row, num_regs_per_lane=num_regs_per_lane)
        self.devices.append(dev)
        return dev

    def add_sm(self, device, *, sm_id=0, exception=0, error_pc=None):
        row = pack_row(
            "CudbgSmTableEntry",
            smId=sm_id,
            exception=exception,
            errorPCValid=0 if error_pc is None else 1,
            errorPC=0 if error_pc is None else error_pc,
        )
        sm = _SM(device, len(device.sms), row)
        device.sms.append(sm)
        return sm

    def add_grid(
        self,
        device,
        *,
        grid_id=1,
        context=None,
        module_handle=0,
        function=0,
        function_entry=0,
        params_offset=0,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
        grid_status=None,
        kernel_type=0,
    ):
        """Add a full CudbgGridTableEntry. ``context`` ties the grid to its
        context and module so a thread's grid resolves to its kernel; the
        grid/block dimensions and ACTIVE status describe the launch."""
        context_id = context.context_id if context is not None else 0
        status = (
            CUDBG_GRID_STATUS_ACTIVE if grid_status is None else grid_status
        )
        row = pack_row(
            "CudbgGridTableEntry",
            gridId64=grid_id,
            contextId=context_id,
            function=function,
            functionEntry=function_entry,
            moduleHandle=module_handle,
            paramsOffset=params_offset,
            kernelType=kernel_type,
            gridStatus=status,
            gridDimX=grid_dim[0],
            gridDimY=grid_dim[1],
            gridDimZ=grid_dim[2],
            blockDimX=block_dim[0],
            blockDimY=block_dim[1],
            blockDimZ=block_dim[2],
        )
        grid = _Grid(len(device.grids), row)
        device.grids.append(grid)
        return grid

    def add_context(
        self,
        device,
        *,
        context_id,
        tid=0,
        shared_window_base=0,
        local_window_base=0,
        global_window_base=0,
    ):
        """Add a CUDA context to a device (CudbgContextTableEntry). A grid's
        ``contextId`` and a module's owning context resolve through this row."""
        row = pack_row(
            "CudbgContextTableEntry",
            contextId=context_id,
            sharedWindowBase=shared_window_base,
            localWindowBase=local_window_base,
            globalWindowBase=global_window_base,
            deviceIdx=device.idx,
            tid=tid,
        )
        ctx = _Context(device, len(device.contexts), context_id, row)
        device.contexts.append(ctx)
        return ctx

    def add_module(self, context, *, module_handle):
        """Add a loaded module to a context (CudbgModuleTableEntry). Cubins
        passed ``module=`` are linked to it, mapping a grid's ``moduleHandle``
        to its cubin image."""
        mod = _Module(context, len(context.modules), module_handle)
        context.modules.append(mod)
        return mod

    def add_cta(self, sm, *, grid_id=1, block_idx=(0, 0, 0)):
        row = pack_row(
            "CudbgCTATableEntry",
            gridId64=grid_id,
            blockIdxX=block_idx[0],
            blockIdxY=block_idx[1],
            blockIdxZ=block_idx[2],
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
        row = pack_row(
            "CudbgWarpTableEntry",
            errorPC=0 if error_pc is None else error_pc,
            warpId=warp_id,
            validLanesMask=valid_lanes_mask,
            activeLanesMask=active_lanes_mask,
            isWarpBroken=1 if is_warp_broken else 0,
            errorPCValid=0 if error_pc is None else 1,
            numRegs=num_regs,
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
        row = pack_row(
            "CudbgThreadTableEntry",
            virtualPC=pc,
            physPC=pc,
            ln=lane_id,
            threadIdxX=thread_idx[0],
            threadIdxY=thread_idx[1],
            threadIdxZ=thread_idx[2],
            exception=exception,
            callDepth=call_depth,
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

    def set_lane_backtrace(self, lane, frames):
        """Attach a thread call stack (CudbgBacktraceTableEntry rows) to a lane.

        ``frames`` is a list of (return_address, virtual_return_address, level)
        tuples; the outermost frame uses a virtual_return_address of 0 to
        terminate unwinding."""
        frames = tuple(frames)
        backtrace = b"".join(
            pack_row(
                "CudbgBacktraceTableEntry",
                returnAddress=ra,
                virtualReturnAddress=vra,
                level=level,
            )
            for ra, vra, level in frames
        )
        row = bytearray(lane.row_bytes)
        struct.pack_into("<I", row, LANE_CALL_DEPTH_OFFSET, len(frames))
        lane.row_bytes = bytes(row)
        lane.backtrace = backtrace

    # -- memory ------------------------------------------------------------

    def add_global_memory(self, addr, data, name=None):
        self.global_mem.append((addr, bytes(data), name))

    def add_managed_memory(self, addr, data, name=None):
        self.managed_mem.append((addr, bytes(data), name))

    def add_local_memory(self, lane, addr, data):
        lane.local = (addr, bytes(data))

    def add_shared_memory(self, cta, addr, data):
        cta.shared = (addr, bytes(data))

    def add_parameter_memory(self, grid, data):
        grid.params = bytes(data)

    # -- grid constant banks ----------------------------------------------

    def add_constbank(self, grid, *, addr, size, bank_id=0):
        grid.constbanks.append(
            pack_row(
                "CudbgConstBankTableEntry",
                addr=addr,
                size=size,
                bankId=bank_id,
            )
        )

    # -- images ------------------------------------------------------------

    def add_relocated_cubin(self, cubin_bytes, name=None, module=None):
        """Embed a relocated cubin. With ``module`` (from ``add_module``) the
        image is linked under that module's table, associating it with a grid;
        without it the image is emitted at the root (sh_link == 0)."""
        self.relocated_cubins.append((bytes(cubin_bytes), name, module))

    def add_unrelocated_cubin(self, cubin_bytes, name=None, module=None):
        self.unrelocated_cubins.append((bytes(cubin_bytes), name, module))

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

        # Resolved during device emission: maps a module object id to the
        # (modtbl section name, module row index) a cubin links to.
        self._module_links = {}

        # The ELF string table device/sm-type names resolve through; found by
        # the name ".strtab". Emitted only when a name was interned.
        if len(self.strtab) > 1:
            emit(".strtab", 3, bytes(self.strtab))  # SHT_STRTAB

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

        # Root-level memory leaves (sh_link == 0).
        for i, (addr, data, name) in enumerate(self.global_mem):
            emit(name or f".cudbg.global.{i}", CUDBG_SHT_GLOBAL_MEM, data,
                 address=addr)
        for i, (addr, data, name) in enumerate(self.managed_mem):
            emit(name or f".cudbg.managed.{i}", CUDBG_SHT_MANAGED_MEM, data,
                 address=addr)
        # Cubin images: linked under their module table when a module was
        # given (so a grid's moduleHandle resolves to it), else at the root.
        for i, (data, name, module) in enumerate(self.relocated_cubins):
            link, info = self._cubin_link(module)
            emit(name or f".cudbg.relfimg.{i}", CUDBG_SHT_RELF_IMG, data,
                 link=link, info=info)
        for i, (data, name, module) in enumerate(self.unrelocated_cubins):
            link, info = self._cubin_link(module)
            emit(name or f".cudbg.elfimg.{i}", CUDBG_SHT_ELF_IMG, data,
                 link=link, info=info)

        sections.extend(self.raw_sections)
        return sections

    def _cubin_link(self, module):
        """Resolve a module to the (modtbl section name, row index) a cubin's
        sh_link / sh_info should reference, or (None, 0) for a root cubin."""
        if module is None:
            return None, 0
        return self._module_links[id(module)]

    def _emit_device(self, emit, dev, devtbl):
        """Emit the context, grid, and SM subtrees for one device."""
        if dev.contexts:
            self._emit_contexts(emit, dev, devtbl)

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

    def _emit_contexts(self, emit, dev, devtbl):
        """Emit a device's context table and, per context, its module table.
        Records where each module's cubin should link (see _cubin_link)."""
        ctxtbl = f".cudbg.ctxtbl.{dev.tag}"
        emit(ctxtbl, CUDBG_SHT_CTX_TABLE,
             _table((c.row_bytes for c in dev.contexts), CONTEXT_ROW_SIZE),
             link=devtbl, info=dev.idx, entsize=CONTEXT_ROW_SIZE)
        for ctx in dev.contexts:
            if not ctx.modules:
                continue
            modtbl = f".cudbg.modtbl.{ctx.tag}"
            emit(modtbl, CUDBG_SHT_MOD_TABLE,
                 _table((pack_row("CudbgModuleTableEntry",
                                  moduleHandle=m.module_handle)
                         for m in ctx.modules), MODULE_ROW_SIZE),
                 link=ctxtbl, info=ctx.row_index, entsize=MODULE_ROW_SIZE)
            for mod in ctx.modules:
                self._module_links[id(mod)] = (modtbl, mod.row_index)

    def _emit_grids(self, emit, dev, devtbl):
        """Emit a device's grid table (sibling of the SM subtree) and the
        parameter-memory and constant-bank leaves hanging off it."""
        gridtbl = f".cudbg.gridtbl.{dev.tag}"
        emit(gridtbl, CUDBG_SHT_GRID_TABLE,
             _table((g.row_bytes for g in dev.grids), GRID_ROW_SIZE),
             link=devtbl, info=dev.idx, entsize=GRID_ROW_SIZE)
        for grid in dev.grids:
            emit(
                f".cudbg.param.{dev.tag}.grid{grid.row_index}",
                CUDBG_SHT_PARAM_MEM,
                grid.params,
                link=gridtbl,
                info=grid.row_index,
            )
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
        emit(".cudbg.bt" + suffix, CUDBG_SHT_BT, lane.backtrace,
             link=lntbl, info=lane_id, entsize=BT_ROW_SIZE)

    def write_yaml(self, path):
        sections = self._build_sections()
        lines = [
            "--- !ELF",
            "FileHeader:",
            "  Class:   ELFCLASS64",
            "  Data:    ELFDATA2LSB",
            "  Type:    ET_CORE",
            "  Machine: EM_CUDA",
            f"  OSABI:   {ELFOSABI_CUDA:#x}",
            f"  ABIVersion: {CUDA_ELF_ABIVERSION:#x}",
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
