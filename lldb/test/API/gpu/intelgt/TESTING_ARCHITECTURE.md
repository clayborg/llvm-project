# IntelGT LLDB Testing Architecture

This document describes how IntelGT GPU tests work from top (CMake) to bottom (test execution).

## Overview

IntelGT GPU tests use LLDB's Python API test framework (dotest.py) to verify GPU debugging functionality. Tests are discovered by lit, built with Makefiles, and executed via Python unittest.

---

## Layer 1: CMake Build System

**File**: `lldb/test/API/CMakeLists.txt`

CMake generates ninja targets for running tests:
- `check-lldb-api-gpu-intelgt`: Run all IntelGT GPU tests
- `check-lldb-api-gpu-intelgt-basic`: Run basic subdirectory tests

Configuration:
- `LLDB_TEST_EXECUTABLE`: Path to lldb binary (build/bin/lldb)
- `LLDB_TEST_COMPILER`: Compiler for test binaries (can be overridden to icpx)
- `LLDB_TEST_MAKE`: Make tool for building test executables (/usr/bin/make)
- `LLDB_TEST_ARCH`: Target architecture (x86_64)

The `add_lit_testsuites()` function scans the test tree and creates ninja targets for each subdirectory.

---

## Layer 2: Lit Test Discovery

**File**: `lldb/test/API/lit.cfg.py`

Lit (LLVM Integrated Tester) discovers and runs Python API tests:
- **Test format**: Each `.py` file is a test
- **Test runner**: Invokes `dotest.py` for each test or directory
- **Configuration**: Passes build paths, compiler, environment to dotest.py

**IntelGT-specific guard**: `lldb/test/API/gpu/intelgt/lit.local.cfg`
```python
if not "lldb-intelgt" in config.enabled_plugins:
    config.unsupported = True
```

This ensures tests only run when IntelGT plugin is built into lldb-server.

---

## Layer 3: dotest.py Test Framework

**File**: `lldb/test/API/dotest.py`

Simple wrapper that imports and runs the test suite:
```python
import lldbsuite.test
lldbsuite.test.run_suite()
```

**Invocation** (manual):
```bash
LLVM_AR=/usr/bin/ar ZET_ENABLE_PROGRAM_DEBUGGING=1 \
python3 lldb/test/API/dotest.py \
  --executable build/bin/lldb \
  --build-dir build \
  --cmake-build-type RelWithDebInfo \
  --llvm-tools-dir build/bin \
  --compiler icpx \
  --make /usr/bin/make \
  lldb/test/API/gpu/intelgt/basic/
```

**Parameters**:
- `--executable`: lldb binary to test
- `--build-dir`: Where to build test executables
- `--compiler`: Compiler for test programs (icpx for SYCL)
- `--make`: Make tool for building tests
- Final argument: Test directory to run

**Environment requirements**:
- `ZET_ENABLE_PROGRAM_DEBUGGING=1`: Enable Intel GPU debugging
- `LLVM_AR=/usr/bin/ar`: Archiver tool (needed for some build steps)
- oneAPI environment sourced (for icpx, SYCL runtime, Level Zero)

---

## Layer 4: Test Base Class

**File**: `lldb/packages/Python/lldbsuite/test/tools/gpu/intelgt_testcase.py`

`IntelGtTestCaseBase` provides GPU-specific test infrastructure:

### Key Methods:

**`run_to_gpu_breakpoint(source, gpu_bkpt_pattern, cpu_bkpt_pattern)`**
- Builds test binary via Makefile
- Creates LLDB target and launches process
- Waits for GPU target auto-creation (happens during `zeModuleCreate`)
- Sets GPU breakpoint on GPU target
- Continues to breakpoint and returns stopped threads

**`set_gpu_source_breakpoint(source, gpu_bkpt_pattern)`**
- Sets breakpoint on GPU target by source pattern
- Returns breakpoint ID

**`continue_to_gpu_breakpoint(gpu_bkpt_id)`**
- Continues CPU process (GPU implicitly continues too)
- Waits for GPU breakpoint hit
- Returns GPU threads stopped at breakpoint

### Properties (inherited from `GpuTestCaseBase`):
- `self.cpu_target`: Main executable target (index 0)
- `self.gpu_target`: GPU device target (index 1, spirv64 architecture)
- `self.cpu_process`: Host process
- `self.gpu_process`: GPU device process
- `self.select_cpu()`: Switch to CPU target
- `self.select_gpu()`: Switch to GPU target

---

## Layer 5: Test Implementation

**Example**: `lldb/test/API/gpu/intelgt/basic/TestBreakpointHit.py`

```python
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase

class IntelGtBreakpointHitTestCase(IntelGtTestCaseBase):
    def test_gpu_breakpoint_actually_hits(self):
        self.build()  # Invokes Makefile
        
        # Create target and set GPU breakpoint BEFORE running
        exe = self.getBuildArtifact("simple_kernel")
        target = self.dbg.CreateTarget(exe)
        
        source_spec = lldb.SBFileSpec("simple_kernel.cpp", False)
        gpu_bkpt = target.BreakpointCreateBySourceRegex(
            "// GPU BREAKPOINT", source_spec)
        
        # Launch - GPU target auto-creates, breakpoint hits
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetEnvironmentEntries(env_list, True)
        process = target.Launch(launch_info, error)
        
        # Verify GPU target and breakpoint hit
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        gpu_process = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertEqual(gpu_process.GetState(), lldb.eStateStopped)
        
        # Find thread stopped at breakpoint
        stopped_threads = [t for t in gpu_process.threads
                          if t.GetStopReason() == lldb.eStopReasonBreakpoint]
        self.assertGreater(len(stopped_threads), 0)
```

---

## Layer 6: Build System (Makefile)

**File**: `lldb/test/API/gpu/intelgt/basic/Makefile`

Each test directory has a Makefile that builds test executables:

```makefile
CXX := icpx
CXXFLAGS_EXTRAS := -std=c++17 -fsycl -g -O0 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
LD_EXTRAS := -fsycl
CXX_SOURCES := simple_kernel.cpp
EXE := simple_kernel

include Makefile.rules
```

**Makefile.rules location**: `lldb/packages/Python/lldbsuite/test/make/Makefile.rules`

This is LLDB's standard test Makefile infrastructure that:
- Compiles source files with specified compiler
- Links executables with specified flags
- Places output in build directory (accessible via `getBuildArtifact()`)

**Build invocation**:
- Triggered by `self.build()` in test
- Runs: `make -C <test-dir> -f Makefile EXE=simple_kernel CXX=icpx ...`
- Output: `build/gpu/intelgt/basic/TestBreakpointHit.test_*/simple_kernel`

---

## Layer 7: Test Execution Flow

### Step-by-step:

1. **Ninja invokes lit**:
   ```bash
   ninja check-lldb-api-gpu-intelgt-basic
   ```

2. **Lit discovers tests**:
   - Scans `lldb/test/API/gpu/intelgt/basic/` for `Test*.py` files
   - Checks `lit.local.cfg`: requires `lldb-intelgt` plugin
   - If plugin available, proceeds to run tests

3. **Lit runs dotest.py**:
   ```bash
   python3 dotest.py --executable build/bin/lldb \
     --compiler icpx \
     -p TestBreakpointHit.py \
     lldb/test/API/gpu/intelgt/basic/
   ```

4. **dotest.py discovers test methods**:
   - Imports `TestBreakpointHit.py`
   - Finds `IntelGtBreakpointHitTestCase` class
   - Discovers `test_gpu_breakpoint_actually_hits()` method
   - Each `test_*` method becomes a unittest test case

5. **Test execution**:
   ```
   setUp() → self.build() → Makefile builds simple_kernel
   test_gpu_breakpoint_actually_hits():
     → CreateTarget(simple_kernel)
     → BreakpointCreateBySourceRegex()
     → Launch() → lldb spawns lldb-server
     → lldb-server loads IntelGT plugin
     → IntelGT plugin intercepts zeModuleCreate
     → GPU target auto-created (spirv64)
     → GPU breakpoint resolves
     → Kernel executes, hits breakpoint
     → GPU process stops
     → Test verifies stop reason and location
   tearDown() → cleanup
   ```

6. **lldb-server and IntelGT plugin**:
   - lldb spawns `lldb-server gdbserver` automatically
   - lldb-server loads IntelGT plugin (if built-in)
   - Plugin intercepts Level Zero API calls
   - When `zeModuleCreate()` called:
     - Plugin creates GPU target (spirv64 architecture)
     - Registers GPU process/threads
     - Resolves pending breakpoints
   - When kernel executes:
     - EU thread hits breakpoint
     - Plugin reports stop to lldb
     - lldb notifies test via Python API

7. **Result reporting**:
   ```
   PASS: test_gpu_breakpoint_actually_hits
   SUCCESS: Hit GPU breakpoint at simple_kernel.cpp:18
   Ran 1 test in 5.898s
   OK
   ```

---

## Data Flow Summary

```
CMake (generates targets)
  ↓
Ninja (check-lldb-api-gpu-intelgt-basic)
  ↓
Lit (discovers Test*.py files)
  ↓
dotest.py (runs Python unittest)
  ↓
TestBreakpointHit.py::test_gpu_breakpoint_actually_hits()
  ↓
self.build() → Make (icpx builds simple_kernel.cpp)
  ↓
lldb Python API (CreateTarget, Launch, etc.)
  ↓
lldb client (spawns lldb-server, sends GDB-remote packets)
  ↓
lldb-server (IntelGT plugin intercepts zeModuleCreate)
  ↓
Level Zero Runtime (loads SPIR-V module, launches kernel)
  ↓
Intel GPU (kernel executes, hits breakpoint)
  ↓
EU Debugger (reports thread stopped)
  ↓
IntelGT plugin (sends stop packet to lldb)
  ↓
lldb Python API (process.GetState() → eStateStopped)
  ↓
Test assertions (verify breakpoint hit)
  ↓
unittest result (PASS/FAIL)
```

---

## Key Files Reference

```
lldb/test/API/
├── CMakeLists.txt                          # Ninja target generation
├── lit.cfg.py                              # Lit configuration
├── dotest.py                               # Test runner entry point
└── gpu/intelgt/
    ├── lit.local.cfg                       # Plugin guard
    ├── README                              # How to run tests
    ├── SYCL_LLDB_TEST_MIGRATION_PLAN.md   # Migration plan
    └── basic/
        ├── Makefile                        # Build configuration
        ├── simple_kernel.cpp               # SYCL test application
        ├── TestBreakpointHit.py            # MVP breakpoint test
        ├── TestBasicIntelGtPlugin.py       # 5 basic tests
        └── TestBasicManual.py              # Simple build test

lldb/packages/Python/lldbsuite/test/
├── __init__.py                             # run_suite() function
├── tools/gpu/
│   └── intelgt_testcase.py                 # IntelGtTestCaseBase
└── make/
    └── Makefile.rules                      # Standard build rules
```

---

## Prerequisites for Running Tests

1. **Build LLDB with IntelGT plugin**:
   ```bash
   cd ~/dev/llvm-project-intelgt/build
   ninja lldb lldb-server
   ```

2. **Enable GPU debugging**:
   ```bash
   export ZET_ENABLE_PROGRAM_DEBUGGING=1
   echo 1 | sudo tee /sys/class/drm/card*/prelim_enable_eu_debug
   ```

3. **Source oneAPI environment**:
   ```bash
   source ~/intel/oneapi/setvars.sh
   ```

4. **Run tests**:
   ```bash
   cd ~/dev/llvm-project-intelgt
   LLVM_AR=/usr/bin/ar ZET_ENABLE_PROGRAM_DEBUGGING=1 \
   ninja check-lldb-api-gpu-intelgt-basic
   ```

---

## IntelGT-Specific Workflow

Unlike AMD GPU debugging, IntelGT has specific requirements:

1. **Set GPU breakpoint BEFORE running**: 
   - GPU target doesn't exist until `zeModuleCreate`
   - But kernel may execute immediately after module creation
   - Solution: Set breakpoint on CPU target first, it becomes pending
   - When GPU target created, pending breakpoint resolves automatically

2. **GPU target auto-creation**:
   - IntelGT plugin intercepts `zeModuleCreate()` Level Zero API call
   - Creates second LLDB target with `spirv64-unknown-unknown` triple
   - Registers GPU process and shadow thread
   - All GPU breakpoints set on this target

3. **Two-target model**:
   - Target 0 (CPU): Host x86_64 executable
   - Target 1 (GPU): Device spirv64 module
   - Both targets share same lldb-server connection
   - Plugin manages both processes in single debugging session

4. **Environment inheritance**:
   - lldb spawns lldb-server with inherited environment
   - lldb-server must have `ZET_ENABLE_PROGRAM_DEBUGGING=1`
   - lldb-server must have oneAPI libraries in `LD_LIBRARY_PATH`
   - Test framework passes full environment via `SBLaunchInfo`

This architecture allows comprehensive GPU debugging testing while reusing LLDB's existing Python test infrastructure.
