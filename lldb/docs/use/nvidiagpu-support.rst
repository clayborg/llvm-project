NVIDIA GPU Support in LLDB
==========================

System requirements
^^^^^^^^^^^^^^^^^^^

You need to have the CUDA Driver and the CUDA Toolkit installed. They can be
installed following the `official download page <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network>`_.

CMake requirements
^^^^^^^^^^^^^^^^^^

The only requirement for building the plugin is that you need to specify the
CMake variable `LLDB_ENABLE_NVIDIAGPU_PLUGIN` as `ON`. Everything else
related to the build system should be taken care of automatically, except
for uncommon CUDA Toolkit and CUDA driver installations, which will require
you to provide some CMake variables manually. See the CMake variables section
below.

CMake variables
^^^^^^^^^^^^^^^

- `LLDB_ENABLE_NVIDIAGPU_PLUGIN`: enables this plugin at the build system level.
- `NVIDIAGPU_DEBUGGER_INCLUDE_DIR`: path to the CUDA debugger headers required
  to build this plugin. This is the folder that contains the `cudadebugger.h`
  header file, e.g. `/usr/local/cuda/extras/Debugger/include`. If the CUDA
  Toolkit is installed in a standard location, this variable will be deduced
  automatically.
- `NVIDIAGPU_NVCC_PATH`: path to the NVCC compiler to use in tests. If the CUDA
  Toolkit is installed in a standard location, this variable will be deduced
  automatically.

Driver compatibility
^^^^^^^^^^^^^^^^^^^^

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

Running the tests
^^^^^^^^^^^^^^^^^

```
# First build the test runner
ninja lldb-dotest
# Now you can run the tests
./bin/llvm-lit ../llvm-project/lldb/test/API/gpu/nvidia/ -a -v
```


Remote platforms
^^^^^^^^^^^^^^^^

You can connect to a remote platform and connect the the GPU lldb-server by
setting up the following environment variables:

- `NVIDIAGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST`: the host to listen to for remote
  connections.
- `NVIDIAGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT`: the port to listen to for remote
  connections.
- `NVIDIAGPU_DEBUGGER_REMOTE_HOST`: the host to connect to for remote
  connections.

Example:

First launch the lldb-server in platform mode on the machine with IP address
`10.112.215.212` with these variables set:

```
NVIDIAGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT=0 \
NVIDIAGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST="*" \
NVIDIAGPU_DEBUGGER_REMOTE_HOST="10.112.215.212" \
./bin/lldb-server platform --listen  "*:12346" --server
```

Then connect remotely to the lldb-server with the following command:

```
lldb
> platform select remote-linux
> platform connect connect://10.112.215.212:12346
> file /remote/path/to/a/program
> run
```