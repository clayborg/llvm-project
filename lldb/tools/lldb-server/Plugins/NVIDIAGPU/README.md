# NVIDIA GPU support for lldb-server

## Requirements

The only requirement for building the plugin is that you need to specify the
CMake variable `LLDB_ENABLE_NVIDIAGPU_PLUGIN` as `ON`. Everything else
related to the build system should be taken care of automatically, except
for uncommon CUDA Toolkit and CUDA driver installations.

## Driver compatibility

This plugin relies on a header file provided by NVIDIA `cudadebugger.h` that
contains hardcoded version numbers of the CUDA driver this plugin will interact
with. This header is provided by the current CUDA Toolkit installation, unless
specified manually via the `NVIDIAGPU_DEBUGGER_INCLUDE_DIR` CMake variable.
Make sure that this version matches the one of your driver, otherwise this
plugin won't be able to initialize.

- The version numbers in this header file are specified by the variables
  `CUDBG_API_VERSION_MAJOR` and `CUDBG_API_VERSION_MINOR`.
- The version of your driver can be obtained via the `DRIVER version` section
  of the `nvidia-smi --version` output.

## CMake variables

- `LLDB_ENABLE_NVIDIAGPU_PLUGIN`: enables this plugin at the build system level.
- `NVIDIAGPU_DEBUGGER_INCLUDE_DIR`: path to the CUDA debugger headers required
  to build this plugin.