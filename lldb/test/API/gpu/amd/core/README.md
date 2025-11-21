# AMD GPU Core File Debugging Tests

This directory contains tests for AMD GPU core file debugging functionality in LLDB.

## Quick Start

**1. Copy your merged GPU core file to the source directory:**
```bash
cp /path/to/your/merged_gpu.core lldb/test/API/gpu/amd/core/test.core
```

**2. Run all tests from your build directory:**
```bash
cd build/Debug  # or build/Release
./bin/llvm-lit -av <path-to-llvm-project>/lldb/test/API/gpu/amd/core/TestAmdGpuCoreFile.py
```

**3. Run a specific test method:**
```bash
./bin/llvm-lit -av <path-to-test>/TestAmdGpuCoreFile.py -a '--filter=test_load_amd_gpu_core_file'
```

## Prerequisites

- **GPU Core File**: Real merged CPU+GPU core from an actual GPU application crash
- **LLDB Build**: Built with `-DLLDB_ENABLE_AMDGPU=ON`
- **ROCm**: ROCr and DBGAPI libraries installed

## Important Notes

- Core file **must** be in the **source** directory (`lldb/test/API/gpu/amd/core/test.core`)
- Do **not** place it in the build directory - the test framework copies it automatically
- Without a core file, tests skip gracefully with `UNSUPPORTED` status
- Tests require **real** GPU cores from actual crashes, not synthetic/minimal cores

## Test Coverage

The test suite includes 6 focused tests for GPU core file debugging:

1. **`test_load_amd_gpu_core_file`** - Core loading creates CPU and GPU targets
2. **`test_gpu_target_process_state`** - GPU process is in stopped state
3. **`test_gpu_threads_present`** - GPU waves are enumerated as threads
4. **`test_gpu_register_context`** - GPU registers are accessible
5. **`test_gpu_architecture_detection`** - GPU target has amdgcn architecture
6. **`test_gpu_memory_reading`** - GPU memory can be read via DBGAPI callbacks

## Expected Results

### With Real GPU Core File

Tests will attempt to load your core file and validate GPU debugging functionality:

```
PASS: test_load_amd_gpu_core_file           # Core loaded, CPU+GPU targets created
PASS: test_gpu_target_process_state         # GPU process state is stopped  
PASS: test_gpu_threads_present              # GPU waves enumerated
PASS: test_gpu_register_context             # GPU registers accessible
PASS: test_gpu_architecture_detection       # amdgcn architecture detected
PASS: test_gpu_memory_reading               # GPU memory reading works
...
```

### Without Core File

Tests will skip gracefully:

```
UNSUPPORTED: test_load_amd_gpu_core_file (GPU core file not available - copy your core to: ...)
```

## Notes

- Tests require **real GPU core files** from actual crashes
- Synthetic/minimal cores are not sufficient for testing
- Core files can be large (several GB) and are not checked into the repository
- Tests are designed to work with AMD merged cores (CPU+GPU combined)
