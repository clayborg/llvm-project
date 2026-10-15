# Plan: Convert GDB SYCL Tests to LLDB IntelGT Tests

## Status: READY — all design questions resolved, implementation can start

---

## 1.  Source material

GDB SYCL tests live in `/home/kgerlich/dev/gdb/gdb/testsuite/gdb.sycl/`.

### 1.1  Test inventory

| GDB test file | Source program | What it tests |
|---|---|---|
| `break.exp` | `single-task.cpp` | Set 4 kernel + 1 host BP before run; verify all hit in order |
| `break2.exp` | `single-task.cpp` | Set kernel BPs *while inside* the kernel; verify hit |
| `step.exp` | `single-task.cpp` | `next` × 3 from `kernel-line-1` |
| `step-into-function.exp` | `call-stack.cpp` | `step` into `first()` from the call site |
| `info-locals-and-args.exp` | `call-stack.cpp` | `info locals` / `info args` inside GPU kernel |
| `parallel-for-1D.exp` | `parallel-for-1D.cpp` | BP at `kernel-last-line`, hit 5 times |
| `parallel-for-2D.exp` | `parallel-for-2D.cpp` | Same but 2-D range |
| `step-parallel-for.exp` | `parallel-for-1D.cpp` | `next` × 3 inside `parallel_for`, 5 trips |
| `scheduler-locking.exp` | `parallel-for-1D.cpp` | `scheduler-locking on` → only current thread advances |
| `call-stack.exp` | `call-stack.cpp` | Backtrace at nested + inlined calls; `stepi` loop through prologue |
| `step-canceled.exp` | `parallel-for-1D.cpp` | Racy: step canceled when another thread hits BP first |

### 1.2  Source programs

| Source file | Description |
|---|---|
| `single-task.cpp` | `single_task` kernel, 3-element buffer, simple arithmetic (lines tagged `kernel-line-{1-4}`) |
| `call-stack.cpp` | Deep ordinary call chain (`first`→`second`→`third`→`fourth`) plus inlined calls inside a `single_task` kernel |
| `parallel-for-1D.cpp` | `parallel_for` over 1024 elements; callee `get_dim`; lines tagged `kernel-first-line` / `kernel-last-line` |
| `parallel-for-2D.cpp` | `parallel_for` over 128×64 2-D range; same tags |

All source files `#include "../lib/sycl-util.cpp"` which provides `get_sycl_queue(argc,
argv)` for device selection.  `sycl-util.cpp` is a **GDB-internal** test helper
(`gdb/testsuite/lib/sycl-util.cpp`) — not part of any SYCL SDK or system include
path.  It parses three CLI arguments (device type, name substring, backend) to pick a
`sycl::queue`.  In the LLDB ports this file is **dropped entirely**: each adapted
source file uses `sycl::queue{sycl::gpu_selector_v}` and relies on the env var
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` being set in the test environment.

---

## 2.  Existing LLDB IntelGT infrastructure

| Path | State |
|---|---|
| `lldb/test/API/gpu/intelgt/basic/` | Directory exists, Python source deleted (only `.pyc` remnant) |
| `lldb/packages/Python/lldbsuite/test/tools/gpu/intelgt_testcase.py` | Deleted (only `.pyc` remnant) |
| `lldb/packages/Python/lldbsuite/test/tools/gpu/amdgpu_testcase.py` | Present, reference implementation |
| `lldb/packages/Python/lldbsuite/test/tools/gpu/gpu_testcase.py` | Present, base class |

The deleted `intelgt_testcase.py` was modelled exactly after `amdgpu_testcase.py`:
two-target model, `run_to_gpu_breakpoint`, `set_gpu_source_breakpoint`,
`continue_to_gpu_breakpoint`, `continue_to_gpu_source_breakpoint`.

---

## 3.  Proposed LLDB test tree

Tests are organised by **debugger feature** being tested.  All C++ source programs
live in a single shared `src/` directory; each feature subdirectory contains only
Python test files and a Makefile that builds what it needs from `../src/`.

**Source adaptation rules (applied to every `.cpp` file in `src/`):**
- Remove `#include "../lib/sycl-util.cpp"` and drop `argc`/`argv` from `main`.
- Replace `get_sycl_queue(argc, argv)` with `sycl::queue{sycl::gpu_selector_v}`.
- Add a `// CPU BREAKPOINT` comment after `deviceQueue.submit(...)` as a
  natural host-side stop; GPU BPs set before the GPU target exists are held as
  *pending* by LLDB and auto-resolve when the module loads.

**Build:** every Makefile uses `icpx -fsycl -g` and references sources as
`../src/<file>.cpp`.

```
lldb/packages/Python/lldbsuite/test/tools/gpu/
    intelgt_testcase.py               Phase 1 — restore base class
                                      (add get_current_gpu_thread() method)

lldb/test/API/gpu/intelgt/
    lit.local.cfg                     Phase 1 — guard: requires lldb-intelgt

    src/                              shared C++ sources (no tests here)
        single-task.cpp               adapted from GDB
        call-stack.cpp                adapted from GDB
        parallel-for-1D.cpp           adapted from GDB
        parallel-for-2D.cpp           adapted from GDB

    break/                            Phase 1 — breakpoint tests
        Makefile                      builds single-task, parallel-for-1D/2D
        TestBreakIntelGT.py           from break.exp        (single-task)
        TestBreak2IntelGT.py          from break2.exp       (single-task)
        TestParallelFor1DIntelGT.py   from parallel-for-1D.exp
        TestParallelFor2DIntelGT.py   from parallel-for-2D.exp

    step/                             Phase 1 — stepping tests
        Makefile                      builds single-task, call-stack,
                                      parallel-for-1D
        TestStepIntelGT.py            from step.exp         (single-task)
        TestStepIntoFunctionIntelGT.py from step-into-function.exp (call-stack)
        TestStepParallelForIntelGT.py from step-parallel-for.exp
        TestStepCanceledIntelGT.py    Phase 2 — from step-canceled.exp
                                      (deferred: inherently racy test)

    backtrace/                        Phase 2 — call stack / frame tests
        Makefile                      builds call-stack
        TestCallStackIntelGT.py       from call-stack.exp
                                      (full stepi loop, up to 100 iterations
                                       with per-instruction backtrace check)

    variables/                        Phase 1 — locals / args inspection
        Makefile                      builds call-stack
        TestInfoLocalsAndArgsIntelGT.py from info-locals-and-args.exp

    scheduler-locking/                Phase 2 — thread control
        Makefile                      builds parallel-for-1D
        TestSchedulerLockingIntelGT.py from scheduler-locking.exp
                                       (@skip: no LLDB equivalent yet)
```

---

## 4.  sycl.exp vs. intelgt_testcase.py analysis

`sycl.exp` (403 lines) provides the following.  The table shows whether each
piece needs porting, is already covered, or is simply not needed.

| sycl.exp function | What it does | LLDB disposition |
|---|---|---|
| `init_sycl_devices_list` | Probe all SYCL devices by running `sycl-devices.cpp` | **Not needed** — IntelGT only targets Intel GPU via Level Zero; no multi-device enumeration |
| `get_sycl_supported_devices` | Compile + run `sycl-hello.cpp` to verify SYCL works per device | **Not needed** — covered by `lit.local.cfg` guard + `lldb-intelgt` plugin check |
| `sycl_start` (Intel GPU path) | `target-non-stop on`, pass device args, run to main, continue until auto-attach fires, `schedule-multi on` | **Already covered** by `run_to_gpu_breakpoint` in `intelgt_testcase.py` |
| `sycl_with_intelgt_lock` | Wrap body in a file lock (GPU is exclusive), set test prefix, call `gdb_exit` | **Not needed** — pytest runs tests in isolation; no file lock required |
| `require_sycl_device` | Filter: does device match type/name? | **Not needed** — device selection uses env var `ZET_ENABLE_PROGRAM_DEBUGGING=1` / `ONEAPI_DEVICE_SELECTOR`, not argv |
| `is_sycl_device_filtered` / `OFFLOAD_DEVICE_GROUP` | Filter by backend and device type | **Not needed** |
| `sycl_get_device_args` / `_prefix` / `_count` | Parse the `;`-separated device descriptor string | **Not needed** |
| `get_current_thread` | Return current thread ID, handles `N` and `N.M` formats | **Gap** — add `get_current_gpu_thread()` to `intelgt_testcase.py` (maps to `process.GetSelectedThread().GetIndexID()`) |
| `get_sycl_header_version` | Return SYCL header namespace version based on `icpx` version | **Defer** — only needed if a test has version-gated `#ifdef` behavior |
| `spawn_sycl_proc` | Low-level process spawn with device args | **Not needed** |

**Bottom line:** No separate `sycl_testcase.py` is needed.  The device-enumeration
and multi-device-selector machinery in `sycl.exp` is a GDB-ism with no LLDB
equivalent.  The only genuine gap is `get_current_thread`, which adds one small
method to `intelgt_testcase.py`.

---

## 5.  GDB → LLDB translation map

| GDB concept | LLDB equivalent |
|---|---|
| `sycl_start $device` | `lldbutil.run_to_source_breakpoint(self, "// CPU BREAKPOINT", source_spec)` |
| `sycl_with_intelgt_lock` | `self.setAsync(True)` + listener-based state tracking |
| `gdb_breakpoint "$srcfile:$line"` | `run_break_set_by_file_and_line(...)` on GPU target |
| `gdb_continue_to_breakpoint` | `self.continue_to_gpu_breakpoint(bkpt_id)` from base class |
| `gdb_test "next"` | `self.runCmd("next")` on GPU thread |
| `gdb_test "step"` | `self.runCmd("step")` on GPU thread |
| `gdb_test "backtrace N"` | `self.runCmd("bt N")` + `self.expect(...)` |
| `gdb_test "info locals"` | `self.runCmd("frame variable")` |
| `gdb_test "info args"` | SB API `frame.GetArguments()` |
| `{sycl debug}` build flags | `icpx -fsycl -g` in Makefile |
| `set scheduler-locking on` | No direct LLDB equivalent; affected tests are `@skip` (Phase 2) |

---

## 6.  How the tests run — connection model

### 6.1  The fundamental issue: auto-launch vs. manual connect

LLDB always uses `lldb-server` under the hood, even for local debugging.
For CPU programs `ProcessGDBRemote::LaunchAndConnectToDebugserver()` spawns
`lldb-server` automatically.  For IntelGT GPU debugging today this does **not**
work end-to-end because:

1. `lldb-server` must be started with `ZET_ENABLE_PROGRAM_DEBUGGING=1` and
   the Level Zero library on `LD_LIBRARY_PATH` **before** the inferior is
   launched — the auto-launch path does not inject these.
2. There is no `PlatformIntelGT` plugin that knows how to set up the
   environment and hand off to the auto-launch machinery.

### 6.2  Option C: environment-variable workaround (available today)

The auto-launch path **does** work if the parent shell has the right
environment set before starting `lldb`, because the spawned `lldb-server`
inherits the parent environment:

```bash
# Once per shell (or in .bashrc):
export LLDB_DEBUGSERVER_PATH=/path/to/intelgt-enabled/lldb-server
export ZET_ENABLE_PROGRAM_DEBUGGING=1
export LD_LIBRARY_PATH=/path/to/level-zero/lib:$LD_LIBRARY_PATH
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Then just run the tests normally:
python3 -m pytest lldb/test/API/gpu/intelgt/break/ -v
```

`lldb` spawns `lldb-server` with the inherited environment; the IntelGT
plugin initialises, intercepts `zeModuleCreate`, and the two-target model
works.  This is **Option C** from the auto-launch design doc and is the
recommended approach for running the tests today.

### 6.3  Manual two-terminal model (fallback / debugging sessions)

When something goes wrong or while developing a new test, the manual
model is more transparent:

```bash
# Terminal 1 — start lldb-server:
ZET_ENABLE_PROGRAM_DEBUGGING=1 \
LD_LIBRARY_PATH=/path/to/level-zero/lib \
  /path/to/lldb-server gdbserver localhost:1234 -- ./my_sycl_test

# Terminal 2 — connect and run:
/path/to/lldb
(lldb) gdb-remote localhost:1234
(lldb) continue
```

LLDB API tests cannot drive this model directly (they do not manage an
external `lldb-server` process), so automated testing requires Option C.

### 6.4  Should we implement auto-launch first?

**Not a prerequisite.**  Option C already makes automated tests possible
with no code changes to LLDB.  A future `PlatformIntelGT` would make the
experience cleaner (no need to pre-set env vars) but is independent of the
test migration work.  The two efforts can proceed in parallel:

| Track | Depends on | Blocks tests? |
|---|---|---|
| SYCL test migration (this plan) | Option C env vars | No |
| `PlatformIntelGT` implementation | — | No |

The test suite itself will not need to change when auto-launch is
implemented — `intelgt_testcase.py` wraps the connection mechanics
and the test methods call only high-level helpers.

### 6.5  System prerequisites (one-time, per boot)

```bash
# Enable EU debugger in the kernel driver (needs root):
echo 1 | sudo tee /sys/class/drm/card*/device/enable_eudebug
```

### 6.6  Running the LLDB API tests

```bash
# Set environment (once per shell):
export LLDB_DEBUGSERVER_PATH=~/dev/llvm-project-amd/build-fast/bin/lldb-server
export ZET_ENABLE_PROGRAM_DEBUGGING=1
export LD_LIBRARY_PATH=/path/to/level-zero/lib:$LD_LIBRARY_PATH
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Build the SYCL test binary (inside the feature dir):
cd lldb/test/API/gpu/intelgt/break
make

# Run a single feature directory:
python3 -m pytest lldb/test/API/gpu/intelgt/break/ -v

# Run all IntelGT tests:
python3 -m pytest lldb/test/API/gpu/intelgt/ -v

# Or via lit:
llvm-lit lldb/test/API/gpu/intelgt/
```

---

## 7.  Implementation order

1. Restore `intelgt_testcase.py` base class.
2. Add `lit.local.cfg` guard.
3. Adapt `.cpp` source files into `src/` (drop sycl-util, add CPU breakpoint
   comment, use `sycl::gpu_selector_v`).
4. Write per-feature Makefiles.
5. Implement Phase 1 tests (`break/`, `step/`, `variables/`).
6. Verify with Option C environment on real hardware.
7. Implement Phase 2 tests (`backtrace/`, `scheduler-locking/`,
   `step-canceled/`).
