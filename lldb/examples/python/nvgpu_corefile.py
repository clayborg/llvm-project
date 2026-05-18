#!/usr/bin/env python
"""
LLDB script helpers for inspecting the NVGPU corefile section hierarchy
built by `ObjectFileELF::BuildNVGPUSectionList` for EM_CUDA + ET_CORE
input.

Tree shape produced by the corefile plugin:

    nvgpucore
      global, managed, cubin, ucubin   (root-level leaves)
      devN
        smN
          ctaN
            shared                     (per-CTA leaf)
            warpN
              uregs, upreds, cbarrier  (per-warp leaves)
              laneN
                regs, preds, local     (per-lane leaves)

Load with:
    (lldb) command script import /path/to/nvgpu_corefile.py

Commands registered under the `nvgpu core` container:
    (lldb) nvgpu core tree         [--depth N] [--filter substr] [--module-idx N]
    (lldb) nvgpu core siblings     <dotted.path> [--filter substr]
    (lldb) nvgpu core stats        [--module-idx N]
    (lldb) nvgpu core thread info  [--tid N]
    (lldb) nvgpu core lookup       [--path PATH | --dev D --sm S --cta C
                                    --warp W --lane L] [--module-idx N]

The `nvgpu core ...` namespace is reserved for commands that operate on
a loaded NVGPU corefile. Future categories (e.g. `nvgpu cubin ...`,
`nvgpu live ...`) can live alongside it at the top level without
colliding.
"""

# [NVIDIA] Example helpers for the nvgpu-core process plugin.

import argparse
import re
import shlex
import struct
import lldb


# SBSection.GetSectionType() returns an integer enum value; the public Python
# API doesn't expose a name lookup. Build one from the lldb module constants.
def _build_type_names():
    spec = [
        # NVGPU-specific
        ("eSectionTypeNVGPURoot", "nvgpu-root"),
        ("eSectionTypeNVGPUDevice", "nvgpu-device"),
        ("eSectionTypeNVGPUSm", "nvgpu-sm"),
        ("eSectionTypeNVGPUCta", "nvgpu-cta"),
        ("eSectionTypeNVGPUWarp", "nvgpu-warp"),
        ("eSectionTypeNVGPULane", "nvgpu-lane"),
        ("eSectionTypeNVGPUGlobalMemory", "nvgpu-global-memory"),
        ("eSectionTypeNVGPUManagedMemory", "nvgpu-managed-memory"),
        ("eSectionTypeNVGPULocalMemory", "nvgpu-local-memory"),
        ("eSectionTypeNVGPUSharedMemory", "nvgpu-shared-memory"),
        ("eSectionTypeNVGPUParamMemory", "nvgpu-param-memory"),
        ("eSectionTypeNVGPURegisters", "nvgpu-registers"),
        ("eSectionTypeNVGPUPredicates", "nvgpu-predicates"),
        ("eSectionTypeNVGPUUniformRegisters", "nvgpu-uniform-registers"),
        ("eSectionTypeNVGPUUniformPredicates", "nvgpu-uniform-predicates"),
        ("eSectionTypeNVGPURelocatedImage", "nvgpu-relocated-image"),
        ("eSectionTypeNVGPUUnrelocatedImage", "nvgpu-unrelocated-image"),
        ("eSectionTypeNVGPUBacktrace", "nvgpu-backtrace"),
        ("eSectionTypeNVGPUDeviceTable", "nvgpu-device-table"),
        ("eSectionTypeNVGPUContextTable", "nvgpu-context-table"),
        ("eSectionTypeNVGPUSmTable", "nvgpu-sm-table"),
        ("eSectionTypeNVGPUGridTable", "nvgpu-grid-table"),
        ("eSectionTypeNVGPUCtaTable", "nvgpu-cta-table"),
        ("eSectionTypeNVGPUWarpTable", "nvgpu-warp-table"),
        ("eSectionTypeNVGPULaneTable", "nvgpu-lane-table"),
        ("eSectionTypeNVGPUModuleTable", "nvgpu-module-table"),
        ("eSectionTypeNVGPUConstBankTable", "nvgpu-constbank-table"),
        ("eSectionTypeNVGPUMetadata", "nvgpu-metadata"),
        ("eSectionTypeNVGPUConvergenceBarrier", "nvgpu-convergence-barrier"),
        # Common ELF / generic types you may run into in cubin modules
        ("eSectionTypeContainer", "container"),
        ("eSectionTypeCode", "code"),
        ("eSectionTypeData", "data"),
        ("eSectionTypeZeroFill", "zero-fill"),
        ("eSectionTypeELFSymbolTable", "elf-symbol-table"),
        ("eSectionTypeELFDynamicSymbols", "elf-dynamic-symbols"),
        ("eSectionTypeELFRelocationEntries", "elf-relocation-entries"),
        ("eSectionTypeELFDynamicLinkInfo", "elf-dynamic-link-info"),
        ("eSectionTypeOther", "regular"),
        ("eSectionTypeInvalid", "invalid"),
    ]
    out = {}
    for attr, name in spec:
        val = getattr(lldb, attr, None)
        if val is not None:
            out[val] = name
    return out


_TYPE_NAMES = _build_type_names()
_NVGPU_ROOT_TYPE = getattr(lldb, "eSectionTypeNVGPURoot", None)


def _section_type_name(section):
    sect_type = section.GetSectionType()
    return _TYPE_NAMES.get(sect_type, f"type={sect_type}")


def _find_nvgpu_module(target):
    """Find the module whose top-level section list contains an nvgpu-root.

    That's the NVGPU corefile module produced by ObjectFileELF for an
    EM_CUDA + ET_CORE input. Returns None if no such module is loaded."""
    if _NVGPU_ROOT_TYPE is None:
        return None
    if not target.IsValid():
        return None
    for i in range(target.GetNumModules()):
        mod = target.GetModuleAtIndex(i)
        for j in range(mod.GetNumSections()):
            sect = mod.GetSectionAtIndex(j)
            if sect.GetSectionType() == _NVGPU_ROOT_TYPE:
                return mod
    return None


def _resolve_module(target, module_idx):
    """Pick the module to inspect. module_idx=None auto-finds the
    corefile (nvgpucore) module; otherwise returns
    target.GetModuleAtIndex(module_idx)."""
    if not target.IsValid() or target.GetNumModules() == 0:
        return None
    if module_idx is None:
        mod = _find_nvgpu_module(target)
        if mod is not None:
            return mod
        return target.GetModuleAtIndex(0)
    if module_idx >= target.GetNumModules():
        return None
    return target.GetModuleAtIndex(module_idx)


def _walk(section, indent, depth_remaining, out, type_filter):
    name = section.GetName()
    type_name = _section_type_name(section)
    file_addr = section.GetFileAddress()
    byte_size = section.GetByteSize()
    file_off = section.GetFileOffset()
    file_size = section.GetFileByteSize()

    line = (
        f"{'  ' * indent}{name:<14} "
        f"type={type_name:<26} "
        f"vmaddr=0x{file_addr:016x} size=0x{byte_size:08x} "
        f"foff=0x{file_off:08x} fsz=0x{file_size:08x}"
    )
    if not type_filter or type_filter in type_name:
        out.append(line)

    if depth_remaining == 0:
        return
    next_depth = depth_remaining - 1 if depth_remaining > 0 else -1
    for i in range(section.GetNumSubSections()):
        _walk(section.GetSubSectionAtIndex(i), indent + 1, next_depth,
              out, type_filter)


def _resolve_path(mod, path):
    """Resolve a dotted path like 'nvgpucore.dev0.sm0.cta0.warp0.lane8'
    against the given module."""
    if mod is None:
        return None
    parts = path.split(".")
    cur = None
    for i in range(mod.GetNumSections()):
        s = mod.GetSectionAtIndex(i)
        if s.GetName() == parts[0]:
            cur = s
            break
    if cur is None:
        return None
    for part in parts[1:]:
        nxt = None
        for i in range(cur.GetNumSubSections()):
            child = cur.GetSubSectionAtIndex(i)
            if child.GetName() == part:
                nxt = child
                break
        if nxt is None:
            return None
        cur = nxt
    return cur


def nvgpu_tree(debugger, command, exe_ctx, result, _):
    parser = argparse.ArgumentParser(prog="nvgpu core tree")
    parser.add_argument("--depth", type=int, default=-1,
                        help="max depth (-1 = unlimited)")
    parser.add_argument("--filter", default="",
                        help="substring match against section type name")
    parser.add_argument("--module-idx", type=int, default=None,
                        help="module index to inspect (default: auto-find "
                             "the nvgpucore module)")
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        return

    target = exe_ctx.target if exe_ctx and exe_ctx.target \
             else debugger.GetSelectedTarget()
    mod = _resolve_module(target, args.module_idx)
    if mod is None:
        result.SetError("no target/module loaded "
                        "(or no nvgpucore module; pass --module-idx N)")
        return

    out = [f"# module: {mod.GetFileSpec().GetFilename()}"]
    for i in range(mod.GetNumSections()):
        _walk(mod.GetSectionAtIndex(i), 0, args.depth, out, args.filter)
    result.AppendMessage("\n".join(out))


def nvgpu_siblings(debugger, command, exe_ctx, result, _):
    parser = argparse.ArgumentParser(prog="nvgpu core siblings")
    parser.add_argument("path",
                        help="dotted path; lists this section's direct children")
    parser.add_argument("--filter", default="",
                        help="only show children whose section type contains this")
    parser.add_argument("--module-idx", type=int, default=None)
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        return

    target = exe_ctx.target if exe_ctx and exe_ctx.target \
             else debugger.GetSelectedTarget()
    mod = _resolve_module(target, args.module_idx)
    if mod is None:
        result.SetError("no target/module loaded "
                        "(or no nvgpucore module; pass --module-idx N)")
        return
    sect = _resolve_path(mod, args.path)
    if not sect:
        result.SetError(f"path not found: {args.path}")
        return

    n = sect.GetNumSubSections()
    out = [f"{args.path} has {n} children:"]
    type_counts = {}
    for i in range(n):
        c = sect.GetSubSectionAtIndex(i)
        type_str = _section_type_name(c)
        type_counts[type_str] = type_counts.get(type_str, 0) + 1
        if args.filter and args.filter not in type_str:
            continue
        out.append(
            f"  [{i:3}] {c.GetName():<14} type={type_str:<26}"
            f"  vmaddr=0x{c.GetFileAddress():016x}"
            f"  foff=0x{c.GetFileOffset():08x}"
            f"  fsz=0x{c.GetFileByteSize():08x}"
        )
    out.append("")
    out.append("Children by type:")
    for t, count in sorted(type_counts.items()):
        out.append(f"  {t:<28} {count}")
    result.AppendMessage("\n".join(out))


def _count_subtree(section, type_counts):
    name = _section_type_name(section)
    type_counts[name] = type_counts.get(name, 0) + 1
    for i in range(section.GetNumSubSections()):
        _count_subtree(section.GetSubSectionAtIndex(i), type_counts)


def nvgpu_stats(debugger, command, exe_ctx, result, _):
    parser = argparse.ArgumentParser(prog="nvgpu core stats")
    parser.add_argument("--module-idx", type=int, default=None,
                        help="module index to inspect (default: auto-find "
                             "the nvgpucore module)")
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        return

    target = exe_ctx.target if exe_ctx and exe_ctx.target \
             else debugger.GetSelectedTarget()
    mod = _resolve_module(target, args.module_idx)
    if mod is None:
        result.SetError("no target/module loaded "
                        "(or no nvgpucore module; pass --module-idx N)")
        return

    counts = {}
    for i in range(mod.GetNumSections()):
        _count_subtree(mod.GetSectionAtIndex(i), counts)

    interesting_order = [
        "nvgpu-root", "nvgpu-device", "nvgpu-sm", "nvgpu-cta", "nvgpu-warp",
        "nvgpu-lane", "nvgpu-registers", "nvgpu-predicates",
        "nvgpu-local-memory", "nvgpu-uniform-registers",
        "nvgpu-uniform-predicates", "nvgpu-convergence-barrier",
        "nvgpu-shared-memory", "nvgpu-global-memory", "nvgpu-managed-memory",
        "nvgpu-relocated-image", "nvgpu-unrelocated-image",
    ]
    out = [f"# module: {mod.GetFileSpec().GetFilename()}",
           "Section type counts:"]
    for t in interesting_order:
        if t in counts:
            out.append(f"  {t:<28} {counts[t]}")
    other = sorted(set(counts) - set(interesting_order))
    if other:
        out.append("")
        out.append("Other types in this module:")
        for t in other:
            out.append(f"  {t:<28} {counts[t]}")
    result.AppendMessage("\n".join(out))


# =============================================================================
# Row decoders. These mirror the layout of the corresponding
# CudbgXxxTableEntry structs from cudacoredump.h that the corefile dumper
# writes. struct.unpack_from ignores trailing bytes, so newer corefile
# versions that add fields stay readable -- we just won't surface the new
# fields here.
# =============================================================================


def _read_section_bytes(section):
    """Read all bytes of an SBSection's data window, or b'' on failure."""
    if not section or not section.IsValid():
        return b""
    err = lldb.SBError()
    data = section.GetSectionData()
    if not data or not data.IsValid():
        return b""
    size = data.GetByteSize()
    if size == 0:
        return b""
    if hasattr(data, "uint8s"):
        try:
            return bytes(data.uint8s)
        except (TypeError, ValueError):
            pass
    return bytes(data.GetUnsignedInt8(err, i) for i in range(size))


# CudbgThreadTableEntry: virtualPC(Q) physPC(Q) ln(I) threadIdxX/Y/Z(I)
# exception(I) callDepth(I) syscallCallDepth(I) ccRegister(I) cbuThreadState(I).
_LANE_FMT = "<QQIIIIIIIII"
_LANE_SIZE = struct.calcsize(_LANE_FMT)


def _decode_lane(data_bytes):
    if len(data_bytes) < _LANE_SIZE:
        return None
    f = struct.unpack_from(_LANE_FMT, data_bytes)
    return {
        "virtualPC": f[0],
        "physPC": f[1],
        "ln": f[2],
        "threadIdx": (f[3], f[4], f[5]),
        "exception": f[6],
        "callDepth": f[7],
        "syscallCallDepth": f[8],
        "ccRegister": f[9],
        "cbuThreadState": f[10],
    }


# CudbgCTATableEntry: gridId64(Q) blockIdxX/Y/Z(I) pad(I) clusterIdxX/Y/Z(I)
# pad(I) clusterDimX/Y/Z(I).
_CTA_FMT = "<QIIIIIIIIIII"
_CTA_SIZE = struct.calcsize(_CTA_FMT)


def _decode_cta(data_bytes):
    if len(data_bytes) < _CTA_SIZE:
        return None
    f = struct.unpack_from(_CTA_FMT, data_bytes)
    return {
        "gridId": f[0],
        "blockIdx": (f[1], f[2], f[3]),
        "clusterIdx": (f[5], f[6], f[7]),
        "clusterDim": (f[9], f[10], f[11]),
    }


# CudbgWarpTableEntry: errorPC(Q) warpId(I) validLanesMask(I) activeLanesMask(I)
# isWarpBroken(I) errorPCValid(I) pad(I) numRegs(I) pad(I) sharedMemSize(I)
# pad(I) inSyscallLanesMask(I) cbuActive(I) cbuExited(I) cbuCollective(I)
# barrierScope(I) pad(I) additionalBarrierInfo(Q).
_WARP_FMT = "<QIIIIIIIIIIIIIIIIQ"
_WARP_SIZE = struct.calcsize(_WARP_FMT)


def _decode_warp(data_bytes):
    if len(data_bytes) < _WARP_SIZE:
        return None
    f = struct.unpack_from(_WARP_FMT, data_bytes)
    return {
        "errorPC": f[0],
        "warpId": f[1],
        "validLanesMask": f[2],
        "activeLanesMask": f[3],
        "isWarpBroken": f[4],
        "errorPCValid": f[5],
        "numRegs": f[7],
        "sharedMemSize": f[9],
        "inSyscallLanesMask": f[11],
        "cbuActiveLanesMask": f[12],
        "cbuExitedLanesMask": f[13],
        "cbuCollectiveLanesMask": f[14],
        "barrierScope": f[15],
        "additionalBarrierInfo": f[17],
    }


# CudbgSmTableEntry: smId(I) pad(I) exception(I) errorPCValid(I) errorPC(Q)
# clusterExcTargetBlockIdxValid(I) clusterExcTargetBlockIdxX/Y/Z(I)
# exceptionString(Q).
_SM_FMT = "<IIIIQIIIIQ"
_SM_SIZE = struct.calcsize(_SM_FMT)


def _decode_sm(data_bytes):
    if len(data_bytes) < _SM_SIZE:
        return None
    f = struct.unpack_from(_SM_FMT, data_bytes)
    return {
        "smId": f[0],
        "exception": f[2],
        "errorPCValid": f[3],
        "errorPC": f[4],
        "clusterExceptionTargetBlockIdxValid": f[5],
        "clusterExceptionTargetBlockIdx": (f[6], f[7], f[8]),
    }


# CudbgDeviceTableEntry (partial): devName(Q) devType(Q) smType(Q) devId(I)
# pciBusId(I) pciDevId(I) numSMs(I) numWarpsPerSM(I) numLanesPerWarp(I)
# numRegsPerLane(I) numPredicatesPrLane(I) smMajor(I) smMinor(I)
# instructionSize(I) status(I) numUniformRegsPrWarp(I)
# numUniformPredicatesPrWarp(I) numConvergenceBarriersPrWarp(I).
_DEV_FMT = "<QQQIIIIIIIIIIIIIII"
_DEV_SIZE = struct.calcsize(_DEV_FMT)


def _decode_device(data_bytes):
    if len(data_bytes) < _DEV_SIZE:
        return None
    f = struct.unpack_from(_DEV_FMT, data_bytes)
    return {
        "devId": f[3],
        "pciBusId": f[4],
        "pciDevId": f[5],
        "numSMs": f[6],
        "numWarpsPerSM": f[7],
        "numLanesPerWarp": f[8],
        "numRegsPerLane": f[9],
        "numPredicatesPrLane": f[10],
        "smMajor": f[11],
        "smMinor": f[12],
        "numUniformRegsPrWarp": f[15],
        "numUniformPredicatesPrWarp": f[16],
    }


# =============================================================================
# Commands
# =============================================================================


def _yesno(b):
    return "yes" if b else "no"


def _stop_reason_str(thread):
    desc = thread.GetStopDescription(256)
    return desc if desc else "<none>"


def nvgpu_thread_info(debugger, command, exe_ctx, result, _):
    """Show LLDB-side info (name, stop reason, frames) for an NVGPU thread."""
    parser = argparse.ArgumentParser(prog="nvgpu core thread info")
    parser.add_argument("--tid", type=int, default=None,
                        help="thread ID (default: currently selected thread)")
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        return

    target = exe_ctx.target if exe_ctx and exe_ctx.target \
             else debugger.GetSelectedTarget()
    process = target.GetProcess() if target.IsValid() else None
    if not process or not process.IsValid():
        result.SetError("no process loaded")
        return

    if args.tid is not None:
        thread = process.GetThreadByID(args.tid)
        if not thread.IsValid():
            result.SetError(f"no thread with tid={args.tid}")
            return
    else:
        thread = process.GetSelectedThread()
        if not thread.IsValid():
            result.SetError("no thread selected; pass --tid N")
            return

    out = [f"Thread #{thread.GetIndexID()} (tid={thread.GetThreadID()})",
           f"  Name:        {thread.GetName() or '<unnamed>'}",
           f"  Stop reason: {_stop_reason_str(thread)}"]

    frame0 = thread.GetFrameAtIndex(0)
    if frame0.IsValid():
        out.append(f"  PC:          0x{frame0.GetPC():016x}")
        err_pc = frame0.FindRegister("errorPC")
        if err_pc.IsValid():
            val = err_pc.GetValueAsUnsigned(0)
            if val:
                out.append(f"  errorPC:     0x{val:016x}")

    nframes = thread.GetNumFrames()
    if nframes > 0:
        out.append(f"  Frames:      {nframes}")
        for i in range(min(nframes, 16)):
            f = thread.GetFrameAtIndex(i)
            if f.IsValid():
                fn = f.GetFunctionName() or "<unknown>"
                out.append(f"    #{i:<2} 0x{f.GetPC():016x} {fn}")
        if nframes > 16:
            out.append(f"    ... ({nframes - 16} more frames not shown)")

    result.AppendMessage("\n".join(out))


def nvgpu_lookup(debugger, command, exe_ctx, result, _):
    """Decode the lane/warp/CTA/SM/device rows at a hardware position."""
    parser = argparse.ArgumentParser(prog="nvgpu core lookup")
    parser.add_argument("--path", default=None,
                        help="explicit section path "
                             "(e.g. nvgpucore.dev0.sm0.cta0.warp0.lane0); "
                             "overrides the --dev/--sm/--cta/--warp/--lane "
                             "form when both are given")
    parser.add_argument("--dev", type=int, default=0,
                        help="device index (default 0)")
    parser.add_argument("--sm", type=int, default=0,
                        help="SM index within the device (default 0)")
    parser.add_argument("--cta", type=int, default=0,
                        help="CTA index within the SM (default 0)")
    parser.add_argument("--warp", type=int, default=0,
                        help="warp index within the CTA (default 0)")
    parser.add_argument("--lane", type=int, default=0,
                        help="lane index within the warp 0..31 (default 0)")
    parser.add_argument("--module-idx", type=int, default=None,
                        help="module index to inspect (default: auto-find "
                             "the nvgpucore module)")
    try:
        args = parser.parse_args(shlex.split(command))
    except SystemExit:
        return

    target = exe_ctx.target if exe_ctx and exe_ctx.target \
             else debugger.GetSelectedTarget()
    mod = _resolve_module(target, args.module_idx)
    if mod is None:
        result.SetError("no target/module loaded "
                        "(or no nvgpucore module; pass --module-idx N)")
        return

    if args.path:
        path = args.path
    else:
        path = (f"nvgpucore.dev{args.dev}.sm{args.sm}.cta{args.cta}"
                f".warp{args.warp}.lane{args.lane}")
    lane_sec = _resolve_path(mod, path)
    if not lane_sec or not lane_sec.IsValid():
        result.SetError(f"path not found: {path}")
        return

    # Walk up the synthetic parent chain. Each ancestor's data window is
    # its row in the corresponding table.
    warp_sec = lane_sec.GetParent()
    cta_sec = warp_sec.GetParent() if warp_sec.IsValid() else lldb.SBSection()
    sm_sec = cta_sec.GetParent() if cta_sec.IsValid() else lldb.SBSection()
    dev_sec = sm_sec.GetParent() if sm_sec.IsValid() else lldb.SBSection()

    lane = _decode_lane(_read_section_bytes(lane_sec))
    warp = _decode_warp(_read_section_bytes(warp_sec))
    cta = _decode_cta(_read_section_bytes(cta_sec))
    sm = _decode_sm(_read_section_bytes(sm_sec))
    dev = _decode_device(_read_section_bytes(dev_sec))

    # The lane index is encoded in the section's name ("lane0", "lane1", ...)
    # by `BuildNVGPUSectionList`. We recover it by parsing the name, which
    # works for both the --dev/--sm/--cta/--warp/--lane form and an
    # explicit --path form.
    lane_name = lane_sec.GetName() or ""
    name_match = re.match(r"lane(\d+)$", lane_name)
    lane_idx = int(name_match.group(1)) if name_match else args.lane

    out = [f"Lane at {path}:"]
    if lane:
        valid = bool(warp and (warp["validLanesMask"] & (1 << lane_idx)))
        active = bool(warp and warp["errorPCValid"] and
                      (warp["activeLanesMask"] & (1 << lane_idx)))
        out += [
            "  Lane:",
            f"    lane index:  {lane_idx}",
            f"    valid:       {_yesno(valid)}",
            f"    active:      {_yesno(active)}",
            f"    virtualPC:   0x{lane['virtualPC']:016x}",
            f"    threadIdx:   (x={lane['threadIdx'][0]} "
            f"y={lane['threadIdx'][1]} z={lane['threadIdx'][2]})",
            f"    exception:   {lane['exception']}",
            f"    callDepth:   {lane['callDepth']}",
        ]
    else:
        out.append("  Lane:        <unable to decode lane row>")

    if cta:
        out += [
            "  CTA:",
            f"    gridId:      {cta['gridId']}",
            f"    blockIdx:    (x={cta['blockIdx'][0]} "
            f"y={cta['blockIdx'][1]} z={cta['blockIdx'][2]})",
            f"    clusterIdx:  (x={cta['clusterIdx'][0]} "
            f"y={cta['clusterIdx'][1]} z={cta['clusterIdx'][2]})",
            f"    clusterDim:  (x={cta['clusterDim'][0]} "
            f"y={cta['clusterDim'][1]} z={cta['clusterDim'][2]})",
        ]

    if warp:
        out += [
            "  Warp:",
            f"    warpId:      {warp['warpId']}",
            f"    validLanes:  0x{warp['validLanesMask']:08x}",
            f"    activeLanes: 0x{warp['activeLanesMask']:08x}",
            f"    errorPC:     0x{warp['errorPC']:016x} "
            f"(valid={_yesno(warp['errorPCValid'])})",
            f"    isWarpBroken: {warp['isWarpBroken']}",
            f"    numRegs:     {warp['numRegs']}",
            f"    sharedMemSize: {warp['sharedMemSize']}",
        ]

    if sm:
        out += [
            "  SM:",
            f"    smId:        {sm['smId']}",
            f"    exception:   {sm['exception']}",
            f"    errorPC:     0x{sm['errorPC']:016x} "
            f"(valid={_yesno(sm['errorPCValid'])})",
        ]

    if dev:
        out += [
            "  Device:",
            f"    devId:       {dev['devId']}",
            f"    pci:         {dev['pciBusId']:02x}:{dev['pciDevId']:02x}",
            f"    sm_arch:     {dev['smMajor']}.{dev['smMinor']}",
            f"    counts:      {dev['numSMs']} SMs, "
            f"{dev['numWarpsPerSM']} warps/SM, "
            f"{dev['numLanesPerWarp']} lanes/warp",
            f"    regs/lane:   {dev['numRegsPerLane']} R, "
            f"{dev['numPredicatesPrLane']} P",
            f"    uregs/warp:  {dev['numUniformRegsPrWarp']} UR, "
            f"{dev['numUniformPredicatesPrWarp']} UP",
        ]

    result.AppendMessage("\n".join(out))


def __lldb_init_module(debugger, _):
    # Use __name__ so the registrations work regardless of how the file
    # is loaded (filename inside lldb's examples tree, or a user copy).
    mod = __name__
    # Top-level container reserved for any NVGPU-related Python tooling
    # (corefile inspection lives under `nvgpu core`; future helpers like
    # `nvgpu cubin` or `nvgpu live` can sit alongside).
    debugger.HandleCommand(
        'command container add -h '
        '"NVIDIA GPU debugging helpers" nvgpu')
    debugger.HandleCommand(
        'command container add -h '
        '"Inspect a loaded NVGPU corefile (section tree + per-thread state)" '
        'nvgpu core')
    debugger.HandleCommand(
        f'command script add -f {mod}.nvgpu_tree '
        '-h "Walk the synthetic NVGPU section tree." nvgpu core tree')
    debugger.HandleCommand(
        f'command script add -f {mod}.nvgpu_siblings '
        '-h "List a section\'s direct children by dotted path." '
        'nvgpu core siblings')
    debugger.HandleCommand(
        f'command script add -f {mod}.nvgpu_stats '
        '-h "Count sections by type in the corefile module." '
        'nvgpu core stats')
    debugger.HandleCommand(
        f'command script add -f {mod}.nvgpu_lookup '
        '-h "Decode the lane/warp/CTA/SM/device rows at a hardware position." '
        'nvgpu core lookup')
    debugger.HandleCommand(
        'command container add -h '
        '"Per-thread inspection commands for NVGPU corefile threads" '
        'nvgpu core thread')
    debugger.HandleCommand(
        f'command script add -f {mod}.nvgpu_thread_info '
        '-h "Show LLDB-side info for an NVGPU thread." '
        'nvgpu core thread info')
    print('"nvgpu core" command group registered: tree, siblings, stats, '
          'lookup, thread info')
