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
- `NVGPU_DISASSEMBLER_PATH`: a default path to the nvdisasm disassembler that
  will be hardcoded in LLDB. It bypasses the regular disassembler discovery
  logic.
- `NVGPU_TEST_LD_LIBRARY_PATH`: the LD_LIBRARY_PATH that lit gives the
  NVIDIA GPU tests, the lldb they run, and the programs it debugs, for
  example the directory of a `libcuda.so.1` that is not installed. Empty leaves
  LD_LIBRARY_PATH as is. This doesn't affect regular builds.

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

When the program initializes CUDA, or when LLDB attaches to a program that
already has, LLDB adds a second target, for the GPU, next to the CPU target. The
GPU target has no executable of its own, so ``target list`` and stop messages
show it by its session name, which is also the name lldb-dap gives the GPU
session:

.. code-block:: text

  (lldb) target list
  Current targets:
    target #0: /path/to/a.out ( arch=x86_64-unknown-linux-gnu, platform=host, pid=12345, state=stopped )
  * target #1: NVIDIA GPU Session ( arch=nvptx64-nvidia-cuda, platform=nvgpu, pid=1, state=stopped )

If the CUDA debugger API fails to initialize, for example because no CUDA
device is visible, LLDB reports the error and adds no GPU target. The program
keeps running, without GPU debugging.

If the CUDA debugger API reports an internal error, a call into it fails while
LLDB handles its events or resumes the GPU, or LLDB cannot decode the state of
a stopped GPU that the API reports, the GPU stops with an exception that the
error describes. LLDB does not use the API again in that session, so the GPU
can no longer be resumed, nor its memory read: continuing or stepping it
reports the same exception again.

A call into the API that fails while LLDB reads a GPU thread's registers makes
only the registers it reads unavailable.

``process interrupt`` on the GPU target stops the GPU, along with any kernel it
is running, and lists that kernel's threads. The selected thread stops with
signal ``SIGSTOP``, as an interrupted CPU process does. A thread that hits a
breakpoint or raises an exception before the GPU stops reports that stop
instead. The CPU process keeps running, unlike when the GPU stops at a
breakpoint or an exception: LLDB also interrupts the GPU to set, remove, or
disable a breakpoint while a kernel runs, and resumes it at once, with no stop
that the user sees, and lldb-server cannot tell the two interrupts apart.

Interrupting the GPU needs a CUDA 13.2 or newer driver. With an older driver,
no stop answers the interrupt, and LLDB ends the GPU process with "lost
connection" once ``target.process.interrupt-timeout`` expires. Changing a GPU
breakpoint while a kernel runs does the same.

While the GPU process is stopped, lldb-server leaves the CUDA driver's events
for when it resumes. A CPU thread that the driver holds until the debugger
handles its event, such as one that creates a CUDA context or loads a module,
waits until then.

Attaching to a running CUDA application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLDB can attach to a CUDA application that is already running, including one
whose kernels are running:

.. code-block:: bash

  lldb
  > process attach -p <pid>

``process attach`` returns with the CPU process stopped, as for any program.
The GPU comes up once the application runs again: when you continue, the CUDA
driver attaches to LLDB, and LLDB adds the GPU target. Once the driver has
reported the CUDA contexts and modules that the application already had, the
GPU stops, and its ``thread list`` shows the threads of the kernels that were
running. The selected thread stops with signal ``SIGSTOP``, as a CPU process
does when LLDB attaches to it, unless a thread hits a breakpoint or raises an
exception first. The CPU process stops too, unless
`NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP` is set.

Stopping the GPU when the attach finishes needs a CUDA 13.2 or newer driver, as
interrupting it does. With an older driver, the GPU target comes up running.

If the application has not initialized CUDA yet when LLDB attaches, the GPU
target comes up when the application does, as for a program that LLDB
launched. If it has initialized CUDA but has no CUDA context yet, the driver
can attach only once the application creates one. ``lldb-server`` asks it to
attach each time the CPU process stops, so the GPU target comes up once the
CPU process has stopped after the application created a context, at a
breakpoint or an interrupt for example, and runs again.

All of the attach happens in ``lldb-server``, so it works the same through a
platform (see `Remote platforms - Linux`_), and with a debug server that had
already attached when LLDB connected (``lldb-server gdbserver --attach``).

.. note::

  ``lldb-server`` does not yet request the driver's capabilities for lazy
  loading, which CUDA applications use by default. Without them, the driver
  attaches without reporting the application's CUDA contexts and modules, and
  the GPU stops with an error that says so. To debug the GPU of an application
  that you attach to, start it with lazy loading turned off:

  .. code-block:: bash

    CUDA_MODULE_LOADING=EAGER CUDA_DISABLE_FUNCTION_LAZY_FINALIZATION=1 ./app

LLDB attaches through the CUDA driver's attach procedure, which needs a driver
whose ``libcuda`` has ``cudbgInitiateDebuggerAttachProcedureFd``. It does not
support the older ``cudbgApiAttach`` mechanism, which calls into the
application wherever it stopped, even in code that is not safe to interrupt.
With a driver too old to have the attach procedure, LLDB attaches to the CPU
process only, and ``lldb-server`` logs why to its ``gdb-remote plugin`` log
channel.

GPU thread coordinates
^^^^^^^^^^^^^^^^^^^^^^

LLDB reads a GPU thread's block and thread indices from its structured CUDA
metadata. Thread names are display labels and are not parsed for coordinates.
Threads whose metadata is absent, incomplete, or malformed remain visible in
the aggregated ``thread list`` output, but without a coordinate range. A row
containing multiple such threads is labeled ``CUDA thread coordinates
unavailable``; a single thread keeps its display name when available.

Simulated module-load stops have no current GPU coordinate snapshot. Threads
retained from an earlier real GPU stop are labeled ``CUDA thread coordinates
unavailable`` until the next real GPU stop. These simulated stops are visible
to the user when ``target.process.stop-on-sharedlibrary-events`` is ``true``.
Only ``RZ``, ``URZ``, and PC remain readable. PC retains its previous
value, or zero if no previous value is available.

``thread select --blockidx`` and ``thread select --threadidx`` match that same
metadata. A supplied coordinate axis matches only when the candidate thread
has that axis and its value is equal. When an entire block or thread-index
group is unconstrained, a thread may match even if that group is unavailable.
The first matching thread in process order is selected.

When only ``--threadidx`` is supplied, LLDB also uses the selected thread's
entire block index when it is available. When only ``--blockidx`` is supplied,
it similarly uses the selected thread's entire thread index. If the selected
thread does not have the inherited group, that group stays unconstrained.

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

A cubin that is not a valid ELF file becomes no module. lldb-server logs why to
its ``gdb-remote plugin`` log channel.

Breakpoints in kernels
^^^^^^^^^^^^^^^^^^^^^^

When the program starts CUDA, LLDB copies each breakpoint of the CPU target
into the new GPU target, so a breakpoint set before then on a kernel's name or
on a line of a kernel stops the GPU in the kernel. LLDB does not copy a
breakpoint set on the CPU target after that: set a breakpoint in device code on
the GPU target once it exists.

For each kernel, nvcc also compiles host code that launches it: a function with
the kernel's name and parameters, whose code is attributed to the kernel's
first and last lines, and a launch stub that the function calls. A breakpoint
on a kernel's name or on a line of its body therefore also resolves in that
host code, where the CPU would stop each time it launches the kernel. Once the
GPU target exists, LLDB disables those locations in the CPU target, both the
ones that breakpoints already have and the ones they get later, so that the
program stops only in the kernel. ``breakpoint list`` shows them as disabled.

To stop where the CPU launches a kernel, enable the location again with
``breakpoint enable``, or set a breakpoint that names the launch code: on an
address, on the launch stub's name (``__device_stub_`` followed by the kernel's
mangled name, such as ``__device_stub__Z6kerneli`` for ``void kernel(int)``),
or on the kernel's name limited to the program's module with ``--shlib``. LLDB
leaves the locations of those breakpoints enabled.

LLDB finds the launch code from the names nvcc gives it, so it misses some:

- In host code without debug information, the launch code of an ``extern "C"``
  kernel, and that of a kernel whose launch stub the host compiler inlined, as
  an optimizing one does with those of static and template kernels.
- The launch code that runs before LLDB creates the GPU target. When the
  program's first CUDA call is a kernel launch, CUDA starts in the launch stub,
  so a location before the call to the stub can stop the CPU that one time.
  Host code compiled with optimization puts a breakpoint on a kernel's name
  there.

In host code without debug information, LLDB can also take a host function for
launch code: one with the name and parameters of a static or template kernel in
another source file of the same module. A breakpoint on that name then has its
location in the host function disabled as well; enable it with
``breakpoint enable``.

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

lit runs the NVIDIA GPU tests in parallel, except with a driver whose CUDA
debugger API is older than 13.5, such as R615's. With several tests on the GPU
at once, such a driver can lose a GPU stop or exception, and the test that
waits for it times out, so lit runs the GPU tests one at a time and says so in
a note. ``--param nvgpu-jobs=N`` runs N GPU tests at once with any driver, and
``--param nvgpu-jobs=1`` one at a time.

Run the GPU tests through lit. ``lldb-dotest`` does not pass them the plugin,
the nvcc, or the environment that lit does, so a test run with it alone fails
to build its program.

Writing GPU tests
^^^^^^^^^^^^^^^^^

A test that debugs a live CUDA program, or makes a core file by running one,
needs a GPU and is skipped on a machine that cannot run it. A test that only
reads core files or other files from disk needs no GPU and runs anywhere. The
test base classes in ``lldbsuite.test.tools.gpu`` apply this by default; its
``nvgpu_decorators`` module describes how to mark a test that differs.

A test method that needs a particular GPU or driver, such as a minimum compute
capability or CUDA driver version, carries ``@skipUnlessNVGPU`` with the
requirements from ``nvgpu_decorators``, and is skipped on a machine that does
not meet them:

.. code-block:: python

  @skipUnlessNVGPU(SM(at_least=90))  # device 0 is sm_90 or newer
  def test_clusters(self):
      ...

  @skipUnlessNVGPU(Driver(at_least=(13, 2)))  # a CUDA 13.2 or newer driver
  def test_interrupt(self):
      ...

Device 0 is the first GPU the program sees, in CUDA's device order, which puts
the fastest GPU first unless ``CUDA_DEVICE_ORDER`` says otherwise. A test needs
a newer driver than the CUDA 13.0 minimum when LLDB uses a part of the CUDA
debugger API that older drivers lack.

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

A program that makes no CUDA call before its first kernel launch has no such
line: lldb creates the GPU target only when the launch initializes CUDA. Set
its breakpoints on the CPU target before the run, as a user does with
``breakpoint set`` and ``run``. ``run_to_gpu_line(source, marker)`` does that
for a marked line and runs the program to the first GPU stop;
``run_to_gpu_breakpoint(target, breakpoint)`` does it for a breakpoint the test
set itself. Both return the GPU target's copy of the breakpoint. lldb copies
every CPU breakpoint into the GPU target when it creates it; a copy keeps the
original's settings but is numbered in the order it is made, so it has the
original's ID only if no CPU breakpoint was deleted first.
``gpu_breakpoint_copy`` finds the copy by its settings, not its ID. Mark a
line the CPU never runs, such as one in a ``__device__`` function.

A test program can be a copy of a `cuda-samples
<https://github.com/NVIDIA/cuda-samples>`_ program. Each copy has a directory
of its own under ``gpu/nvidia/samples/``, with the Makefile and the test file
for that sample. The copy adds a ``// Based on`` line with the URL of the file
at the commit of the release tag it was copied from, and keeps the sample's
license header and formatting (the ``.clang-format`` in ``samples/`` turns
formatting off there), so that a diff against that file shows only the
changes the copy makes. To copy a newer version, merge the sample's changes
since that commit into the copy, for example with
``git merge-file <copy> <sample at that commit> <sample at the newer tag>``,
and update the URL. Comments that mark breakpoint lines go at the end of the
sample's own lines, and a change whose reason is not obvious gets a comment.

When a test needs the program to run differently, prefer the sample's own
options, as the matrixMul tests size its matrices with ``-wA`` and the
others. Failing that, set a variable from LLDB before the program first
reads it, as they set ``block_size`` before the launch; the program's code
must handle the new value. Change the copy only when neither works.

The test methods of a class that all run one program can build it once with
``build_shared(exe)``. The program is then in a directory of its own,
``<Module>.<Class>.<exe>``, next to the methods' build directories, which stay
empty. Debug that program to reproduce a failure by hand. It still runs in the
method's build directory. ``make_shared_target(exe, args)`` returns a target
for it whose launches pass ``args`` to the program, as ``settings set
target.run-args`` does, for a sample that takes its problem size as arguments.

lit runs the methods of a test file one after another. With the files in
parallel, as lit runs them with most drivers, the slowest file sets how long
the GPU tests take, so a sample whose methods would make its test file much
slower than the others splits them among several files, one for each workflow
it tests. What they share that depends on the
program, such as its launch arguments and the values it computes, goes in a
Python module in the sample's directory, which the test files import.

LLDB prints an error that it reports outside a command, such as a breakpoint
condition that fails, unless something listens for it, and a condition that
fails in every thread of a kernel reports an error for each thread at each
stop. A test of such a condition collects the errors with ``collect_errors()``
and checks them, so that they do not fill its output.

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

To attach to a CUDA application that is already running on the remote machine,
connect the platform the same way, then attach instead of launching:

.. code-block:: bash

  lldb
  > platform select remote-linux
  > platform connect connect://10.112.215.212:12346
  > process attach -p <pid>

You need ``platform select`` and ``platform connect`` only to attach through a
platform. Without one, start ``lldb-server`` as a debug server attached to the
application on the remote machine, with the same environment variables:

.. code-block:: bash

  NVGPU_DEBUGGER_REMOTE_LISTEN_TO_PORT=0 \
  NVGPU_DEBUGGER_REMOTE_LISTEN_TO_HOST="*" \
  NVGPU_DEBUGGER_REMOTE_HOST="10.112.215.212" \
  ./bin/lldb-server gdbserver --attach <pid> "*:12345"

Then connect LLDB to it:

.. code-block:: bash

  lldb
  > gdb-remote 10.112.215.212:12345

A debug server started without ``--attach <pid>`` attaches when LLDB asks it
to: connect with ``process connect connect://10.112.215.212:12345``, then run
``process attach -p <pid>``.

If lldb cannot reach the GPU server, for example because
`NVGPU_DEBUGGER_REMOTE_HOST` names an address it cannot connect to, lldb reports
that it failed to connect and goes on debugging the CPU process alone. The
application waits in CUDA initialization until the session ends. The CUDA
driver then usually lets it run on without a debugger, but can crash it
instead.

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
