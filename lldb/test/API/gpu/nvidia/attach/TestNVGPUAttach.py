"""
Test attaching to a CUDA program that lldb did not launch.
"""

import os
import signal
import subprocess

import lldb
import lldbgdbserverutils
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_decorators import Driver, skipUnlessNVGPU
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "attach.cu"
# The GPU stops when the attach finishes only on a driver that suspends the
# devices in the background, which CUDA 13.2 added.
ATTACH_STOP_DRIVER = Driver(at_least=(13, 2))
# The driver attaches without reporting the program's contexts and modules
# unless the debugger requests the capabilities for lazy function loading and
# finalization, which lldb-server does not yet, so the program runs with both
# turned off.
LATE_ATTACH_ENV = [
    "CUDA_MODULE_LOADING=EAGER",
    "CUDA_DISABLE_FUNCTION_LAZY_FINALIZATION=1",
]


class TestNVGPUAttach(NVGPUTestCaseBase):
    """The program runs on its own, and lldb attaches to it. When the program
    already uses CUDA, lldb-server asks the driver to attach as lldb attaches
    to the CPU process, the driver attaches once the program runs, and then the
    GPU target comes up."""

    def wait_for_the_program(self, name):
        """Wait for the program to create the file `name` in the build
        directory, failing if it exits first: a program that cannot use the
        GPU exits at its first CUDA call."""
        path = self.getBuildArtifact(name)

        def created():
            self.assertIsNone(self.program.poll(), "the program exited")
            return os.path.exists(path)

        self.assertTrue(self.wait_until(created), f"the program did not create {name}")

    def start_program(self, mode, extra_env=(), lazy_loading=False):
        """Start the program outside lldb in `mode`, and return its process ID
        once it is ready for lldb to attach. attach.cu describes the modes.
        Unless `lazy_loading` is true, the program loads its modules eagerly."""
        env = ["CUDA_MODULE_LOADING=LAZY"] if lazy_loading else LATE_ATTACH_ENV
        self.program = self.spawnSubprocess(
            self.build_shared("attach"),
            args=[mode, self.getBuildDir()],
            extra_env=env + list(extra_env),
        )
        self.wait_for_the_program("ready")
        return self.program.pid

    def launch_the_kernel(self):
        """Let the program, which waits in a mode other than kernel, launch the
        kernel, and wait until it runs."""
        open(self.getBuildArtifact("go"), "w").close()
        self.wait_for_the_program("running")

    def start_debug_server(self, *server_args):
        """Start lldb-server gdbserver with `server_args`, as the user would on
        another machine, and return the port that it listens on."""
        # The server writes its port to this pipe once its read end is open.
        port_pipe = self.getBuildArtifact("port")
        os.mkfifo(port_pipe)
        reader = os.open(port_pipe, os.O_RDONLY | os.O_NONBLOCK)
        self.addTearDownHook(lambda: os.close(reader))
        # A shell that exits at once starts the server, so that the server is
        # not a child of this process, as when the user starts it.
        server_pid = int(
            subprocess.check_output(
                [
                    "sh",
                    "-c",
                    '"$0" gdbserver 127.0.0.1:0 --named-pipe "$@" '
                    ">/dev/null 2>&1 & echo $!",
                    lldbgdbserverutils.get_lldb_server_exe(),
                    port_pipe,
                    *server_args,
                ]
            )
        )

        def kill_server():
            try:
                os.kill(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.addTearDownHook(kill_server)

        port = bytearray()

        def read_port():
            try:
                chunk = os.read(reader, 64)
            except BlockingIOError:
                return False
            port.extend(chunk)
            # The server writes the port followed by a NUL, and closes the pipe.
            return b"\0" in port or (bool(port) and not chunk)

        self.assertTrue(
            self.wait_until(read_port), "the debug server did not report its port"
        )
        return port.split(b"\0")[0].decode()

    def select_host_platform(self):
        # process attach makes its target with the selected platform, which an
        # earlier test in this process can leave as the GPU target's, and that
        # platform cannot attach.
        self.runCmd("platform select host")

    def attach(self, pid):
        self.select_host_platform()
        self.runCmd(f"process attach -p {pid}")

    def continue_until_the_gpu_stops(self):
        """Resume the CPU process, which lldb has attached to, and wait for the
        GPU to stop where the attach ends."""
        self.continue_and_wait_for_gpu_target()
        self.wait_for_gpu_to_stop()
        if not self.cpu_stops_with_gpu():
            return
        self._settle_cpu(self.CPU_STOP_GRACE_SECONDS)
        self.assertIn(
            lldb.eStopReasonTargetSync,
            [thread.GetStopReason() for thread in self.cpu_process.threads],
            "the CPU did not stop with the GPU",
        )

    def assert_stopped_in_the_running_kernel(self):
        """The GPU is stopped with both warps of the kernel, and its selected
        thread stopped with SIGSTOP, as a CPU process does when lldb attaches to
        it."""
        self.assertEqual(self.gpu_target.GetTargetSessionName(), "NVIDIA GPU Session")
        self.assertEqual(
            sorted(self.thread_coords(thread) for thread in self.gpu_process.threads),
            [((0, 0, 0), (x, 0, 0)) for x in range(64)],
            self._describe_gpu_threads(),
        )
        thread = self.gpu_process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            self.gpu_process.GetUnixSignals().GetSignalNumberFromName("SIGSTOP"),
            thread.GetStopDescription(256),
        )
        self.assertIn("spinKernel", thread.GetFrameAtIndex(0).GetFunctionName())

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_to_a_running_kernel(self):
        """Once the program runs, the GPU stops with the threads of the kernel
        that was running, and the session then debugs the kernel as one that
        lldb launched does."""
        self.attach(self.start_program("kernel"))
        self.continue_until_the_gpu_stops()
        self.assert_stopped_in_the_running_kernel()

        breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in the loop")
        )
        self.assertGreater(breakpoint.GetNumLocations(), 0, "no breakpoint location")
        self.continue_gpu_until_breakpoint(breakpoint)

    def attach_before_the_first_cuda_context(self, mode):
        """Attach to the program in `mode`, which has initialized CUDA but has
        no CUDA context yet, and let it launch the kernel. The driver can attach
        only once the program has a context, so lldb-server asks it to attach
        at the next stop of the CPU process, and the GPU comes up once the
        program runs again."""
        self.attach(self.start_program(mode))
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue(), "continue CPU process")
        self.launch_the_kernel()

        self.assertIsNone(self.gpu_target, "the GPU target came up without a stop")
        self.assertEqual(
            self.pull_state(self.cpu_process),
            lldb.eStateRunning,
            "the CPU process should be running",
        )
        self.assertSuccess(self.cpu_process.Stop(), "interrupt CPU process")
        self.wait_for_stop(self.cpu_process)

        self.continue_until_the_gpu_stops()
        self.assert_stopped_in_the_running_kernel()

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_before_the_first_cuda_context(self):
        """The program has called cuInit already, which initialized the driver,
        so only the driver's attach procedure can bring the GPU up."""
        self.attach_before_the_first_cuda_context("after-init")

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_before_the_program_calls_cuinit_again(self):
        """The program calls cuInit again before it creates a context, as a
        library that makes sure that CUDA is initialized does. Only the first
        call initializes the driver, so the driver does not pick up the
        debugger API that lldb-server sets up at the second, and lldb-server
        must leave the GPU to the driver's attach procedure."""
        self.attach_before_the_first_cuda_context("init-again")

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_before_the_first_cuda_context_and_cuinit_after_it(self):
        """The program creates a context and launches the kernel, then calls
        cuInit again. The driver's attach descriptor is valid by then, so
        lldb-server asks the driver to attach at that call's breakpoint, and
        the GPU comes up without an interrupt."""
        self.attach(self.start_program("init-after-kernel"))
        open(self.getBuildArtifact("go"), "w").close()
        self.continue_until_the_gpu_stops()
        self.assert_stopped_in_the_running_kernel()

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_through_a_debug_server_started_by_hand(self):
        """Attach through a debug server that lldb did not start, with no
        platform connected, as to a program on another machine. lldb-server
        asks the driver to attach by itself, so the GPU comes up all the
        same."""
        pid = self.start_program("kernel")
        port = self.start_debug_server()

        self.select_host_platform()
        target = self.dbg.CreateTarget(self.build_shared("attach"))
        # Not gdb-remote, which waits for a stop: this server has no process
        # until the attach.
        error = lldb.SBError()
        process = target.ConnectRemote(
            self.dbg.GetListener(), f"connect://127.0.0.1:{port}", "gdb-remote", error
        )
        self.assertSuccess(error)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateConnected]
        )
        self.runCmd(f"process attach -p {pid}")

        self.continue_until_the_gpu_stops()
        self.assert_stopped_in_the_running_kernel()

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_through_a_debug_server_that_attached_first(self):
        """Connect to a debug server that attached to the program before lldb
        connected. lldb handles the attach's stop before it loads the program's
        libraries, so libcuda's symbols reach lldb-server only once lldb has
        loaded them, and the GPU comes up all the same."""
        port = self.start_debug_server("--attach", str(self.start_program("kernel")))

        self.select_host_platform()
        self.dbg.CreateTarget(self.build_shared("attach"))
        self.runCmd(f"gdb-remote 127.0.0.1:{port}")

        self.continue_until_the_gpu_stops()
        self.assert_stopped_in_the_running_kernel()

    def initialize_cuda_and_stop_in_the_kernel(self):
        """Let the program, which waits to initialize CUDA, initialize it and
        launch the kernel. The initialization breakpoints bring the GPU target
        up, as on launch, and the kernel stops at a breakpoint that lldb copies
        into the GPU target."""
        # Only the kernel has code on this line, so the breakpoint must match
        # it exactly: one that may move would resolve in the CPU process.
        self.cpu_target.BreakpointCreateByLocation(
            lldb.SBFileSpec(SOURCE),
            line_number(SOURCE, "// breakpoint in the loop"),
            0,
            0,
            lldb.SBFileSpecList(),
            False,
        )
        open(self.getBuildArtifact("go"), "w").close()
        self.continue_and_wait_for_gpu_target()
        self.wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(self.gpu_target.GetBreakpointAtIndex(0))
        if self.cpu_stops_with_gpu():
            self._settle_cpu(self.CPU_STOP_GRACE_SECONDS)

    def test_attach_before_cuda_is_initialized(self):
        """Attach while libcuda is not loaded yet. There is no driver to ask to
        attach then, so the initialization breakpoints bring the GPU target up
        once the program initializes CUDA, as on launch."""
        self.attach(self.start_program("before-init"))
        self.initialize_cuda_and_stop_in_the_kernel()

    @skipUnlessNVGPU(ATTACH_STOP_DRIVER)
    def test_attach_to_a_program_that_loads_lazily(self):
        """CUDA programs load their modules lazily by default. The driver then
        attaches without reporting the program's contexts and modules, so the
        GPU stops with an error that says how to start the program instead."""
        self.attach(self.start_program("kernel", lazy_loading=True))
        self.continue_and_wait_for_gpu_target()
        self.wait_for_gpu_to_stop()
        thread = self.gpu_process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonException)
        self.assertIn("CUDA_MODULE_LOADING=EAGER", thread.GetStopDescription(512))

    def test_attach_before_the_driver_is_initialized(self):
        """Attach while libcuda is loaded but CUDA is not initialized, so the
        driver has not published its attach descriptor yet. lldb-server must
        not ask the driver to attach then, and the initialization breakpoints
        bring the GPU target up once the program initializes CUDA."""
        # Loading libcuda does not initialize the driver; cuInit does.
        self.attach(
            self.start_program("before-init", extra_env=["LD_PRELOAD=libcuda.so.1"])
        )
        self.initialize_cuda_and_stop_in_the_kernel()
