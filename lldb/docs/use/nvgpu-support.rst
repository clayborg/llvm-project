NVIDIA GPU Support in LLDB
==========================

System requirements
^^^^^^^^^^^^^^^^^^^

Building the plugin needs no CUDA Driver or CUDA Toolkit: the CUDA debugger API
headers are vendored in the source tree (see below). The test suite does need
`nvcc`: `NVGPU_NVCC_PATH` is deduced from an installed CUDA Toolkit, and with
`LLDB_INCLUDE_TESTS` on, configure fails if it cannot be deduced or set. So
building on a machine with no CUDA Toolkit means configuring with
`LLDB_INCLUDE_TESTS=OFF`.

Debugging needs the CUDA Driver, and building CUDA programs to debug needs the
CUDA Toolkit. They can be installed following the `official download page <https://developer.nvidia.com/cuda-downloads>`_.
The minimal version of the CUDA Toolkit and CUDA driver that is supported is
13.0.0.

CUDA debugger API headers
^^^^^^^^^^^^^^^^^^^^^^^^^

`cudadebugger.h` and `cudacoredump.h` are vendored in
`lldb/third-party/cuda`, and those copies are what every build compiles
against. A clean checkout builds without an installed CUDA Toolkit.

They are imported verbatim from the CUDA driver sources.

To build against a different copy without committing it, point
`NVGPU_DEBUGGER_INCLUDE_DIR` at the folder holding both headers:

.. code-block:: bash

  cmake -DNVGPU_DEBUGGER_INCLUDE_DIR=/path/to/cuda/debugger/headers ...

The override must be from the same supported CUDA major release (see
`Driver compatibility` below) and no older than the vendored copy. The tree
uses the API the vendored header describes with no compile-time gating, so an
older revision will not compile.

CMake requirements
^^^^^^^^^^^^^^^^^^

The only requirement for building the plugin is that you need to specify the
CMake variable `LLDB_ENABLE_NVGPU_PLUGIN` as `ON`. Everything else
related to the build system should be taken care of automatically, except
for uncommon CUDA Toolkit and CUDA driver installations, which will require
you to provide some CMake variables manually. See the CMake variables section
below.

CMake variables
^^^^^^^^^^^^^^^

- `LLDB_ENABLE_NVGPU_PLUGIN`: enables this plugin at the build system level.
- `NVGPU_DEBUGGER_INCLUDE_DIR`: folder containing the `cudadebugger.h` and
  `cudacoredump.h` header files to build against. Leave it unset to use the
  copies vendored in `lldb/third-party/cuda`, which is the default. See the
  CUDA debugger API headers section above.
- `NVGPU_NVCC_PATH`: path to the NVCC compiler to use in tests. If the CUDA
  Toolkit is installed in a standard location, this variable will be deduced
  automatically.
- `NVGPU_CUDBG_INJECTION_PATH`: path to the CUDA debugger injection library.
  Can also be set at runtime. See Environment variables section for more
  details.
- `NVGPU_CUDA_VISIBLE_DEVICES`: controls which CUDA devices are visible to the
  application being debugged. Can also be set at runtime. See Environment
  variables section for more details.
- `NVGPU_CUDA_DEVICE_ORDER`: controls the ordering of CUDA devices. Can also
  be set at runtime. See Environment variables section for more details.
- `NVGPU_CUDA_LAUNCH_BLOCKING`: when set to "1", forces CUDA kernel launches
  to be synchronous. Can also be set at runtime. See Environment variables
  section for more details.
- `NVGPU_INITIALIZATION_SYMBOL`: an optional CPU symbol to use to identify
  that the CUDA runtime is being initialized.
- `NVGPU_DISASSEMBLER_PATH`: a default path to the nvdisasm disassembler that
  will be hardcoded in LLDB. It bypasses the regular disassembler discovery
  logic.
- `NVGPU_TEST_LD_LIBRARY_PATH`: a way to override LD_LIBRARY_PATH for NVGPU
  tests. This doesn't affect regular builds.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

The following environment variables can be set to control CUDA debugging
behavior when lldb-server starts:

- `CUDBG_INJECTION_PATH`: path to the CUDA debugger injection library. This
  specifies which debugger library should be injected into CUDA applications
  for debugging support. When set, this will be automatically applied when
  the NVIDIA GPU plugin initializes.

- `CUDA_VISIBLE_DEVICES`: controls which CUDA devices are visible to the
  application being debugged. This can be used to restrict debugging to
  specific GPUs. For example, setting this to "0" will make only GPU 0
  visible to the debugged application.

- `CUDA_DEVICE_ORDER`: controls the ordering of CUDA devices. Common values
  are "FASTEST_FIRST" to prioritize faster devices, or "PCI_BUS_ID" to use
  PCI bus ordering. This affects which device gets which device ID.

- `CUDA_LAUNCH_BLOCKING`: when set to "1", forces CUDA kernel launches to be
  synchronous, which can be helpful for debugging by making kernel execution
  more predictable and easier to trace.

- `NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP`: when set to "1", disables the automatic
  suspension of the CPU process when the GPU is stopped.

These environment variables can be set in the shell environment before
starting lldb-server.

Attaching to a running CUDA application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to launching a program under the debugger, you can attach to a CUDA
application that is already running. When you attach to a process that has an
active CUDA context (including one with kernels currently executing), the NVGPU
plugin transparently initializes the debugger API, brings up the GPU target,
and exposes the in-flight kernel's threads in the ``thread list`` command.

.. code-block:: bash

  lldb
  > process attach -p <pid>
  # Resume so the driver can complete the attach procedure.
  > continue

Once the attach completes, a second (GPU) target appears alongside the CPU
target. Select it to inspect device state:

.. code-block:: bash

  > target list
  > target select <gpu-target-index>
  > thread list

How it works
""""""""""""

Because the application is already running, the ``cuInit``-style initialization
breakpoint used by the launch path has already been passed. Instead, the plugin
uses the driver's safe attach mechanism:

#. When LLDB attaches, it tells the GPU plug-ins (via ``jGPUPluginInitialize``)
   that this is an attach. The NVGPU plugin then sets a breakpoint on
   ``CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED``.
#. On the first stop after attaching, ``lldb-server`` resolves the driver's
   attach handshake symbols itself -- it locates ``libcuda`` in the inferior via
   ``/proc/<pid>/maps`` and reads its dynamic symbol table -- so no extra
   gdb-remote round-trip is needed during attach. If the running process
   advertises a usable safe-attach handler (``CUDBG_ATTACH_HANDLER_AVAILABLE``),
   the plugin writes the client handshake globals and a magic byte to the file
   descriptor exported in ``CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD``. This
   asks the driver to inject the debug engine at a point it determines is safe,
   avoiding the unsafe forced function call used by the deprecated mechanism.
#. When the driver finishes injecting the debug engine it calls
   ``CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED``; the plugin's breakpoint fires and
   it initializes the CUDA debugger API exactly like the launch path.
#. The plugin then waits for ``CUDBG_EVENT_ATTACH_COMPLETE``, suspends all
   devices, refreshes device state, and reports the GPU as stopped so the
   kernel's threads appear in the thread list.

LLDB supports **only** this safe late attach mechanism, which requires a CUDA
driver that exports ``cudbgInitiateDebuggerAttachProcedureFd``. The legacy
``cudbgApiAttach()`` injection path (which forces an unsafe dynamic function
call in the inferior, and is error-prone when the application is stopped in a
signal-unsafe state) is intentionally not supported. Attaching to a process
running on a CUDA driver that is too old to provide the safe attach procedure
is not supported; use a newer driver.

Driver compatibility
^^^^^^^^^^^^^^^^^^^^

This plugin is built against a single CUDA debugger-API header
(`cudadebugger.h`, vendored in `lldb/third-party/cuda` or supplied via the
`NVGPU_DEBUGGER_INCLUDE_DIR` CMake variable). The header's
`CUDBG_API_VERSION_MAJOR` identifies the CUDA *major* release the build
targets.

Support policy: a build works with any CUDA driver -- and reads any GPU
coredump -- within that same major release. There is no cross-major-release
compatibility. In other words, an lldb-server (or corefile reader) built
against, say, CUDA 13.x supports every 13.x driver/coredump, but not 12.x or
14.x.

- Live debugging: the plugin queries the driver's API version
  (`cudbgGetAPIVersion`) and negotiates the API revision down to the lesser of
  the driver's and the compiled header's. An older in-major driver therefore
  attaches successfully; API features newer than that driver are simply
  unavailable. A driver from a different major release is rejected with a clear
  error -- rebuild lldb-server against that major's header.
- Coredumps: the reader applies the same policy. It reads any in-major
  coredump and ignores fields the producing driver predates, using the
  producer version recorded in the coredump's metadata section. A coredump
  produced by a different major release is loaded best-effort with a warning,
  since field layouts are not guaranteed across majors.
- The version of your driver can be obtained via the `DRIVER version` section
  of the `nvidia-smi --version` output.

Running the tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # First build the test runner
  ninja lldb-dotest
  # Now you can run the tests
  ./bin/llvm-lit ../llvm-project/lldb/test/API/gpu/nvidia/ -a -v


Remote platforms - Linux
^^^^^^^^^^^^^^^^^^^^^^^^

You can connect to a remote platform and connect the the GPU lldb-server by
setting up the following environment variables:

- `NVGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST`: the host to listen to for remote
  connections.
- `NVGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT`: the port to listen to for remote
  connections.
- `NVGPU_DEBUGGER_REMOTE_HOST`: the host to connect to for remote
  connections.

Example:

First launch the lldb-server in platform mode on the machine with IP address
`10.112.215.212` with these variables set:

.. code-block:: bash

  NVGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT=0 \
  NVGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST="*" \
  NVGPU_DEBUGGER_REMOTE_HOST="10.112.215.212" \
  ./bin/lldb-server platform --listen  "*:12346" --server

Then connect remotely to the lldb-server with the following command:

.. code-block:: bash

  lldb
  > platform select remote-linux
  > platform connect connect://10.112.215.212:12346
  > file /remote/path/to/a/program
  > run

Remote platforms - Android
^^^^^^^^^^^^^^^^^^^^^^^^^^

Android support is much simpler than the regular Linux remote support as you
don't need to set any environment variables.

Example:

First launch the lldb-server in platform mode on the Android device with the
correct port forwarding set up:

.. code-block:: bash

  # On the host
  adb forward tcp:5039 tcp:5039
  # On the Android device
  ./lldb-server platform --listen '*:5039' --server

Then connect remotely to the lldb-server with the following command:

.. code-block:: bash

  lldb
  > platform select remote-android
  > platform connect connect://:5039
  > file /remote/path/to/a/program
  > run
