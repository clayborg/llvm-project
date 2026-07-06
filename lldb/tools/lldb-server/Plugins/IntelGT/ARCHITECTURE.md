# IntelGT lldb-server Plugin — Architecture

## Overview

`LLDBServerPluginIntelGT` is a server-side plugin for `lldb-server` that adds
live Intel GPU EU-thread debugging to the existing CPU+GPU hybrid debugging
architecture introduced by the AMD GPU plugin.

The plugin re-uses the `LLDBServerPlugin` infrastructure that was designed to
be GPU-vendor-agnostic.  A single `lldb-server` process handles both the CPU
inferior (via the standard `NativeProcessLinux`/ptrace path) and all attached
Intel GPU devices (via the Level Zero EU debugger API).

---

## High-level flow

```
┌─────────────────────────────────────────────────────┐
│  lldb-server process                                │
│                                                     │
│  NativeProcessLinux  ←── ptrace ──→  CPU inferior  │
│         │                                           │
│  LLDBServerPluginIntelGT                           │
│         │                                           │
│         ├─ zetDebugAttach() ──→  GPU device 0      │
│         ├─ zetDebugAttach() ──→  GPU device 1  … │
│         │                                           │
│  ProcessIntelGT  (fake NativeProcessProtocol)      │
│         │                                           │
│  GDBRemoteCommunicationServerLLGS  (GPU side)      │
│         │                                           │
│         └── reverse TCP ──→  lldb client           │
└─────────────────────────────────────────────────────┘
```

The LLDB client sees two GDB-remote connections:

1. The primary CPU connection (standard lldb-server).
2. A secondary GPU connection opened by the plugin after `zeModuleCreate`
   fires.  The client uses `DynamicLoaderGDBRemoteGPU` and the `intelgt`
   GPU plugin to drive this connection.

---

## Class hierarchy

```
LLDBServerPlugin  (base, lldb/source/Plugins/Process/gdb-remote/)
└── LLDBServerPluginIntelGT         — plugin entry point

NativeProcessProtocol  (base)
└── ProcessIntelGT                  — GPU-side process object

NativeThreadProtocol  (base)
└── ThreadIntelGT                   — one per SIMD lane (grouped under EUThread)

NativeRegisterContext  (base)
└── RegisterContextIntelGT          — per-lane register access via EUThread cache

EUThreadIntelGT (enable_shared_from_this)
                                    — one per ze_device_thread_t (hardware EU thread)
                                    — owns shared register data cache
                                    — owns stop reason (shared by all lanes)
                                    — creates N lane ThreadIntelGT objects
GpuModuleManager                    — tracks code-object load/unload deltas
```

---

## Plugin lifecycle

### State machine

```
Uninitialized
    │  lldb-server starts; GetInitializeActions() installs a
    │  breakpoint on zeModuleCreate in libze_loader.so
    ▼
Initialized
    │  BreakpointWasHit() fires; zeInit() succeeds
    ▼
Attached
    │  AttachToDevices() calls zetDebugAttach() on every Intel GPU;
    │  CreateGpuProcess() creates a fake ProcessIntelGT;
    │  reverse TCP listener is opened for the LLDB client
    ▼
RuntimeLoaded
    │  ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY received and ACKed;
    │  ZE MODULE_LOAD / THREAD_STOPPED events flow normally
    ▼
Detached
    │  NativeProcessDidExit() or ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT;
    │  zetDebugDetach() called on all sessions
    ▼
  (done)
```

### zeModuleCreate breakpoint

The plugin intercepts `zeModuleCreate` in `libze_loader.so` as the trigger
for GPU attachment.  This is a one-shot breakpoint; after it fires,
`BreakpointWasHit()` sets `disable_bp = true`.  Subsequent GPU module
load/unload events arrive through `ZET_DEBUG_EVENT_TYPE_MODULE_LOAD` and
`ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD` from `zetDebugReadEvent()`.

### Device enumeration and sub-device fallback

`AttachToDevices()` mirrors GDB `ze-low.cc attach_to_devices()`:

1. Call `zeDriverGet()` / `zeDeviceGet()` to enumerate all drivers and devices.
2. For each device whose `vendorId == 0x8086` and `type == ZE_DEVICE_TYPE_GPU`:
   a. Try sub-devices first via `zeDeviceGetSubDevices()` + `zetDebugAttach()`.
   b. `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` → skip, fall back to parent device.
   c. `ZE_RESULT_ERROR_NOT_AVAILABLE` → another debugger is attached, log error.
3. If no sub-device attached, call `zetDebugAttach()` on the parent device.

Each successfully attached device produces a `DeviceSession` record.

---

## ZE event loop

Level Zero does not provide a notification file descriptor.  Events are
polled with `zetDebugReadEvent(session, timeout_ms=0, &event)`.

The plugin creates a `pipe(2)` pair (`m_notifier_fd[2]`).  The read end is
registered with the `MainLoop` as a `GPUIOObject`.  After every event drain,
the write end is written to so the main loop fires again on the next CPU stop.

`NativeProcessIsStopping()` is called each time the CPU process stops.  It
calls `DrainZeEvents()`, which loops over all device sessions calling
`zetDebugReadEvent(timeout=0)` until `ZE_RESULT_NOT_READY`.

Every event with `ZET_DEBUG_EVENT_FLAG_NEED_ACK` set must be followed by
`zetDebugAcknowledgeEvent()` — missing a single ACK silently blocks all
subsequent events on that session.

### Event handling summary

| ZE event type | Action |
|---|---|
| `PROCESS_ENTRY` | ACK; transition state → `RuntimeLoaded` |
| `PROCESS_EXIT` | ACK; call `HandleNativeProcessExit()`; state → `Detached` |
| `MODULE_LOAD` | `ProcessIntelGT::HandleModuleLoad()`; set `load_libraries=true` |
| `MODULE_UNLOAD` | `ProcessIntelGT::HandleModuleUnload()`; set `load_libraries=true` |
| `THREAD_STOPPED` | `ProcessIntelGT::HandleZeThreadStopped()`; record first TID |
| `THREAD_UNAVAILABLE` | `ProcessIntelGT::HandleZeThreadUnavailable()` |
| `PAGE_FAULT` | `ProcessIntelGT::HandleZePageFault()` |
| `DETACHED` | state → `Detached` |

---

## Thread model — SIMD lane granularity

The plugin follows AMD's per-lane thread model:

- **One `EUThreadIntelGT`** per hardware EU thread (`ze_device_thread_t`).
  Analogous to `WaveAMDGPU`.
- **Multiple `ThreadIntelGT`** per EU thread — one per active SIMD lane.
  Analogous to `ThreadAMDGPU`.

When `ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED` is received:

1. An `EUThreadIntelGT` (shared_ptr) is created for the `ze_device_thread_t`.
2. The execution mask is read (CE register & SR0 dispatch mask) to determine
   which SIMD lanes are active.
3. One `ThreadIntelGT` is created per active lane.  All lane threads share
   the same `EUThreadIntelGT` parent and its stop reason and register data.
4. Only the first lane of the first stopped EU thread gets
   `eStopReasonBreakpoint`; other lanes get `eStopReasonNone`.

### Resume logic — x86-like all-stop

All stopped EU threads are visible in `thread list` (like x86 threads).
Only the current thread is shown at the CLI prompt.

When a breakpoint fires, the GPU hardware halts all active EU threads in the
kernel.  Some are at the breakpoint instruction (CR0.1 bit 31 =
`breakpoint_status`); divergent threads are merely halted (CR0.1 bit 30 =
`external_halt_status`).  `HandleZeThreadStopped()` reads CR0.1 via
`ReadStopReason()` to determine the actual stop reason per EU thread.

Lane threads are created for ALL stopped EU threads.  Lane 0 of each EU
thread inherits the EU thread's actual stop reason; other lanes get
`eStopReasonNone`.

When the user types `continue`, ALL stopped EU threads are resumed at once
(x86 all-stop semantics).  Pending MODULE_LOAD ACKs are sent after resume.
The user can inspect any thread before continuing via `thread select`.

### Shadow thread

TID 1 is a permanent "shadow thread" that LLDB requires a process to have at
all times.  It carries no real ZE state; register reads on it return an error
immediately.

### TID encoding

TID allocation uses a monotonically increasing static counter (same pattern
as AMD's `ReserveTidsForWave`).  Each EU thread's lanes get a contiguous
block of TIDs:

```
tid_base = <next available from static counter>
lane 0:  tid = tid_base
lane 1:  tid = tid_base + 1
...
lane N:  tid = tid_base + N
```

TID 1 is reserved for the shadow thread.  The static counter starts at 2.

### Execution mask

Active SIMD lanes are determined from two hardware registers:

- **CE (Channel Enable)** — regset type 4, one DWORD bitmask.
- **SR0 (Status Register 0)** — regset type 5, DWORD 0 = dispatch mask.
- **Active lanes** = CE & dispatch_mask, clamped to `simd_width` bits.

If the execution mask registers cannot be read, all lanes are assumed active.
The default SIMD width is 16 (`.ze_info` parsing is a follow-up).

### Wildcard THREAD_STOPPED (known limitation)

`ze_device_thread_t{UINT32_MAX,...}` means all threads stopped.  The POC
converts wildcard to `{0,0,0,0}` as a representative.  Full topology
expansion is a follow-up task.

---

## Register context — lane-aware

`RegisterContextIntelGT` uses `zetDebugGetRegisterSetProperties()` on
first access to discover the register sets available.  The actual register
data is read through the EU thread's shared cache — all lanes of the same
EU thread share the same raw hardware data.

### SIMD-wide vs shared register sets

Register sets are classified as SIMD-wide or shared:

| Type | Register sets | Lane behavior |
|---|---|---|
| **SIMD-wide** | GRF, ADDR, ACC, MME, SP | Each lane sees a different slice: `offset = lane_id * (byteSize / simd_width)` |
| **Shared** | CR, CE, SR, SBA, FLAG, TDR, DBG, FC | All lanes see the full value |

For SIMD-wide registers, `RegisterInfo.byte_size` reports the per-lane
element size (e.g., 4 bytes for a 64-byte GRF with SIMD16).

### Shared register data cache

Register data is read from hardware once per EU thread stop and cached in
`EUThreadIntelGT`.  All lane `RegisterContextIntelGT` objects index into
this shared cache.  This avoids N redundant `zetDebugReadRegisters` calls
for N lanes.

### PC computation (heapless mode only)

```
pc = isabase + (uint64_t)cr0_dword2
```

Where `cr0_dword2` is bytes 8-11 of the CR0 register, and `isabase` is
SBA register index 4.  The PC is the same for all lanes (it is a shared
register).

Heapful mode (`isabase + IP`) is not supported.

### Expedited registers

The `pc` pseudo-register is marked expedited so LLDB does not need a
round-trip to display the PC after every stop.

---

## Breakpoints

Intel GPU breakpoints work by patching a specific "breakpoint enable" bit in
the existing instruction at the target address, rather than inserting a fixed
opcode.

`ProcessIntelGT::SetBreakpoint()`:

1. Read 16 bytes at `addr` via `zetDebugReadMemory()` using the wildcard
   thread (ISA memory does not require a specific stopped thread).
2. Detect instruction format: bit 29 of DWORD[0] == 1 → compact (8 bytes),
   else full (16 bytes).
3. Call `intelgt::breakpoint_bit_offset(inst, device_id)` to find the correct
   bit position for the device's Xe generation.
4. Save original bytes in `m_bp_saved_opcodes[addr]`.
5. Call `intelgt::set_inst_bit()` and write the patched instruction back.

`ProcessIntelGT::RemoveBreakpoint()` restores from `m_bp_saved_opcodes`.

`GetSoftwareBreakpointPCOffset()` returns 0 — the IntelGT PC points directly
at the breakpoint instruction after a stop, unlike x86 where the PC is
advanced past the `int3`.

### Xe version and breakpoint bit position

`intelgt::get_xe_version(device_id)` maps PCI device IDs to:

| Generation | Breakpoint bit |
|---|---|
| Xe_HP, Xe_HPG, Xe_HPC | bit 30 of DWORD[0] |
| Xe2 (Battlemage) | bit 7 of DWORD[0] |

---

## Module loading

ZE does not provide file paths for GPU ELF modules.  Each
`ZET_DEBUG_EVENT_TYPE_MODULE_LOAD` event carries:

- `moduleBegin` / `moduleEnd` — the in-memory ELF image address range.
- `load` — the GPU virtual address where the module is loaded for execution.

The plugin synthesises a URI of the form `memory://0x<addr>+0x<size>` and
passes it to `GpuModuleManager`.  The `GPUDynamicLoaderLibraryInfo` carries
`native_memory_address` + `native_memory_size` so the LLDB client can read the
ELF bytes directly from the inferior's address space.

---

## Address spaces

| DWARF address space | ZE memory space |
|---|---|
| `ASPACE_GLOBAL (0)` | `ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT` |
| `ASPACE_SLM (1)` | `ZET_DEBUG_MEMORY_SPACE_TYPE_SLM` |

SLM reads require a specific stopped thread context; global reads use the
wildcard thread `{UINT32_MAX,...}`.

---

## GDB code reuse

The following items are derived from GDB source:

| GDB source | LLDB file | What was taken |
|---|---|---|
| `gdb/gdb/arch/intelgt.h` | `IntelGTArch.h` | Feature name constants, `xe_version`, `address_space`, `breakpoint_kind`, `get/set/clear_inst_bit()`, `breakpoint_bit_offset()` |
| `gdb/gdb/arch/intelgt.c` | `IntelGTArch.cpp` | `get_xe_version()` device-ID table |
| `gdb/gdbserver/ze-low.h` | `IntelGTArch.h` | `ze_regset_info`, `ze_node_level_t`, `ze_node_state_t` |
| `gdb/gdbserver/intelgt-ze-low.cc` | `IntelGTArch.cpp`, `LevelZeroHelpers.h` | CR0 bit position constants, `ZeAckEvent()` |
| `gdb/gdbserver/ze-low.cc` | `LevelZeroHelpers.h` | TID encoding/decoding formulas |
| `AMDGPU/GpuModuleManager.h` | `GpuModuleManager.h` | Verbatim copy |

---

## Single-stepping via CR0

Single-stepping (`next`/`step`) is implemented by manipulating the CR0 control
register before calling `zetDebugResume()`.  This is the same mechanism GDB
uses (`gdbserver/intelgt-ze-low.cc`).

### CR0 bits

| DWORD | Bit | Name | Purpose |
|-------|-----|------|---------|
| CR0.0 | 15 | `breakpoint_suppress` | Suppress breakpoint on the current instruction |
| CR0.1 | 31 | `breakpoint_status` | Trigger breakpoint exception on the next instruction |

### Flow

1. LLDB sends `vCont;s` for the stepping thread.
2. `ProcessIntelGT::Resume()` detects `eStateStepping` in the action list.
3. `EUThreadIntelGT::PrepareStep()` reads CR0, sets bit 15 in DWORD[0] and
   bit 31 in DWORD[1], writes CR0 back.
4. `zetDebugResume()` is called.  The EU executes one instruction and fires a
   breakpoint exception.
5. `ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED` arrives.
6. `HandleZeThreadStopped()` checks `m_stepping_eu_threads`: if the thread was
   stepping, calls `ClearStepBits()` and sets `eStopReasonTrace` (instead of
   `eStopReasonBreakpoint`).
7. LLDB receives the stop and shows the next source line.

### Tracking stepping threads

`m_stepping_eu_threads` (`ZeThreadMap<bool>`) tracks which EU threads were
armed for single-step.  Entries are added in `Resume()` when stepping and
removed in `HandleZeThreadStopped()` when the step completes.

---

## DWARF extensions

### DW_OP_INTEL_push_simd_lane (0xed)

Pushes the current SIMD lane index onto the DWARF expression stack.  This
opcode collides with `DW_OP_WASM_location` (also 0xed) and cannot be registered
in `Dwarf.def`.  The named constant and handler live in `SymbolFileIntelGT`
(`DW_OP_INTEL_push_simd_lane = 0xed`, no inline operands).  The correct handler
is selected at object-file open time: `SymbolFileDWARF::CreateInstance` returns a
`SymbolFileIntelGT` for spirv64 triples (Intel GT GPU, `isSPIRV()`) and a
`SymbolFileWasm` for wasm triples, so `ParseVendorDWARFOpcode` dispatches to the right implementation
without any target check inside the expression evaluator itself.

### DW_OP_INTEL_regval_bits (0xfe)

1-byte inline operand: bit count for bit-level register extraction.  Named
constant and handler in `SymbolFileIntelGT` (`DW_OP_INTEL_regval_bits = 0xfe`).

---

## Known limitations (POC scope)

- `llvm::Triple::intelgt` is not registered; LLDB creates the GPU target with
  an unknown architecture.  ELF/DWARF loading still works via generic handlers.
- Wildcard `THREAD_STOPPED` events are converted to `{0,0,0,0}` as a
  representative (full topology expansion not implemented).
- SIMD width defaults to 16; `.ze_info` section parsing for per-kernel SIMD
  width detection is a follow-up.
- Heapful mode PC computation (`isabase + IP`) is not supported.
- No hardware breakpoints or watchpoints.
- No range stepping.
- No core dump support.
- No client-side `PlatformIntelGT` plugin.
- Scheduler-locking (per-EU-thread suspend) is not yet implemented; depends on
  single-stepping working first.
