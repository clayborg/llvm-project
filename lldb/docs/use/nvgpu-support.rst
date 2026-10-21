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

The GPU target
^^^^^^^^^^^^^^

When the program initializes CUDA, LLDB adds a second target, for the GPU, next
to the CPU target. The GPU target has no executable of its own, so
``target list`` and stop messages show it by its session name, which is also the
name lldb-dap gives the GPU session:

.. code-block:: text

  (lldb) target list
  Current targets:
    target #0: /path/to/a.out ( arch=x86_64-unknown-linux-gnu, platform=host, pid=12345, state=stopped )
  * target #1: NVIDIA GPU Session ( arch=nvptx64-nvidia-cuda, platform=nvgpu, pid=1, state=stopped )

If the CUDA debugger API fails to initialize, for example because no CUDA
device is visible, LLDB reports the error and adds no GPU target. The program
keeps running, without GPU debugging.

GPU modules
^^^^^^^^^^^

Each cubin that the driver loads becomes a module of the GPU target. A cubin
built with device debug information (``-G``) is named after the source file it
was compiled from: the cubin for ``main.cu`` is ``main.cubin``. Any other cubin,
including one built with ``-lineinfo`` alone, whose line table does not say
which file is the main one, is named after the first kernel in its symbol table.
The kernel's name is demangled without parameters, template arguments, or
anonymous namespaces, as in ``vectorAdd.cubin`` or ``physics::integrate.cubin``.
A cubin with no kernels is named after its first global function.

If an earlier cubin in the process already has the name, LLDB appends ``-2``,
``-3``, and so on. A cubin with nothing to name it after, or whose function name
does not demangle, is named after the driver's handle for it, as in
``cuda_elf_93825011057424.cubin``.

A cubin with several kernels is named after only one of them, and a cubin
built from several source files, such as one device-linked with ``-rdc``, after
only the first. The name is a label for display and may change. It does not
identify a kernel or a source file: use the module's symbols and line tables
for that.

Running the tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # First build the test runner
  ninja lldb-dotest
  # Now you can run the tests
  ./bin/llvm-lit ../llvm-project/lldb/test/API/gpu/nvidia/ -a -v

lit prints the GPUs it finds in a ``nvgpu:`` note. Where the tests that need a
GPU cannot run (no GPU or driver, or a driver of another CUDA major), they are
reported UNSUPPORTED, and ``-a`` shows why. They run only on the machine that
runs lit, so they are also UNSUPPORTED on a remote platform
(``LLDB_TEST_PLATFORM_URL``). If the configured nvcc
(``NVGPU_NVCC_PATH``) cannot build for the machine's GPUs, those tests fail
instead. ``--param nvgpu-no-device`` hides the GPUs, to run on a GPU machine
what a machine without one would.

Writing GPU tests
^^^^^^^^^^^^^^^^^

A test that debugs a live CUDA program, or makes a core file by running one,
needs a GPU and is skipped on a machine that cannot run it. A test that only
reads core files or other files from disk needs no GPU and runs anywhere. The
test base classes in ``lldbsuite.test.tools.gpu`` apply this by default; its
``nvgpu_decorators`` module describes how to mark a test that differs.

A test that needs no GPU but builds a CUDA program sets the GPU architecture in
its Makefile, for example ``NVCC_ARCHFLAGS := -arch=sm_80``. The default,
``-arch=native``, builds for the GPUs of the machine that runs the test, so the
program would differ from one machine to the next.

To bring a live program to its first stop, use
``run_to_cpu_line(source, marker)`` from ``NVGPUTestCaseBase`` rather than
``lldbutil.run_to_line_breakpoint``. When lldb creates the GPU target it copies
every CPU breakpoint into it. A copied file-and-line breakpoint that may move
to the nearest code resolves in a cubin only inside a function that begins at
or before the marked line, with one line of slack, so a marker on the line
directly above a kernel's definition would stop the GPU in that kernel.
``run_to_cpu_line`` sets a breakpoint that matches the line exactly, so its
copy never resolves to device code, wherever the kernels are defined. Set GPU
breakpoints on ``self.gpu_target`` once the program has stopped there.

Running the tests under AddressSanitizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure a separate build with clang and ``-DLLVM_USE_SANITIZER=Address``,
then build and run the tests with the commands above. No other setup is needed.

A test file fails when lldb, lldb-server, or any other process the test started
reports an ASan error. The report appears in the test output, with function
names and source lines. Because lldb runs lldb-server with its output on
``/dev/null``, ASan writes each report to a file named
``report.<process>.<pid>``, in a directory next to lit's output for the test
file:

.. code-block:: text

  <build>/tools/lldb/test/API/gpu/nvidia/<area>/Test<Name>.py.asan/

On such a build, the NVIDIA test directories set up the rest:

- They preload the ASan runtime into dotest's Python, which is not instrumented
  but loads the instrumented liblldb.
- They add ``protect_shadow_gap=0`` to ``ASAN_OPTIONS``. Without it, the CUDA
  driver cannot initialize the debug API inside the instrumented lldb-server.
- They turn off leak detection, so leaks are not reported.

The CUDA test programs are not instrumented, so they are not checked. A report
that lldb-server writes while it shuts down can be cut short, because lldb stops
the server soon after it starts to exit. The test file still fails in that case.

Building GPU test programs
^^^^^^^^^^^^^^^^^^^^^^^^^^

A GPU test's ``Makefile`` lists its CUDA files in ``NVCC_SOURCES``.
``Makefile.rules`` compiles them with ``-arch=native -G -g -O0`` and links with
nvcc. Before the ``include``, a test can set:

- ``CFLAGS_EXTRAS``: compiler options, which the link also gets.
- ``LD_EXTRAS``: libraries, such as ``-lcuda`` for the driver API.
- ``NVCC_DEBUGFLAGS``: replaces ``-G``. Use ``-lineinfo`` for line tables only,
  or leave it empty for no device debug information; either way nvcc optimizes
  the device code. To change only some files, set it per object after the
  ``include``, as in ``other.o: NVCC_DEBUGFLAGS := -lineinfo``.

A variable in the dictionary a test passes to ``build()`` goes on make's command
line, where it overrides every assignment in a Makefile. A variable that a lit
configuration puts in ``config.environment`` reaches make through dotest's
environment instead, where a Makefile's ``:=`` wins over it and ``?=`` takes
it; lit passes the tests only what its configuration sets, not the shell's
environment. A test whose subject is one of these settings pins it with
``override``, as ``lineinfo/Makefile`` does with
``override NVCC_DEBUGFLAGS := -lineinfo``, which holds against either route.

Each file's device code is a separate cubin, so a kernel can only call device
functions in its own file. ``CFLAGS_EXTRAS := -rdc=true`` links the device code
of all the files together, for device code that calls across files or launches
kernels; nvcc adds the device runtime library itself.

A program that loads device code through the driver API needs rules that build
it next to ``a.out``; see ``lldb/test/API/gpu/nvidia/loaded_modules/Makefile``.
Build a fatbin rather than a cubin, and PTX with ``NVCC_ARCHFLAGS`` empty:
``-cubin`` and ``-ptx`` each take one architecture, and ``-arch=native`` gives
one for each kind of GPU in the machine. When a second test needs these rules,
move them into ``Makefile.rules`` instead of copying them.


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
