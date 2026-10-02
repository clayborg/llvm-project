import os
import time

from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUAttach(NVGPUTestCaseBase):
    """Tests for attaching to an already-running CUDA application.

    Unlike the other NVGPU tests, which launch the inferior under the debugger
    and rely on the cuInit-style initialization breakpoint, this test starts the
    CUDA process independently (with a resident kernel already executing) and
    then attaches to it. lldb must transparently detect the running CUDA
    driver, start the driver's attach procedure, bring up the GPU target, and
    surface the in-flight kernel's threads in the thread list.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def _set_attach_wait_timeout_ms(self, timeout_ms):
        """Set how long lldb keeps the process running for the GPU attach.
        lldb runs in this test's process, so it reads this environment."""
        os.environ["NVGPU_ATTACH_WAIT_TIMEOUT_MS"] = str(timeout_ms)
        self.addTearDownHook(
            lambda: os.environ.pop("NVGPU_ATTACH_WAIT_TIMEOUT_MS", None)
        )

    def _run_until_the_gpu_attach_completes(self):
        """Resume the CPU process, which the late attach needs running, and wait
        for it to bring up the GPU target and stop it."""
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue(), "continue CPU process")
        self.assertTrue(
            self.wait_for(lambda: self.gpu_target is not None),
            "GPU target was not created once the process ran",
        )
        self.wait_for_gpu_to_stop()

    def _start_waiting_to_initialize_cuda(self, extra_env=()):
        """Start the inferior so that it waits before initializing CUDA, and
        return it along with the marker file that releases it."""
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("ready.marker")
        go_marker = self.getBuildArtifact("go.marker")
        for marker in (ready_marker, go_marker):
            if os.path.exists(marker):
                os.remove(marker)
        popen = self.spawnSubprocess(
            exe,
            args=[ready_marker, go_marker],
            extra_env=self.LATE_ATTACH_INFERIOR_ENV + list(extra_env),
        )
        self.assertTrue(
            self.wait_for(lambda: os.path.exists(ready_marker)),
            "the inferior never reported that it was waiting to initialize CUDA",
        )
        return popen, go_marker

    def _initialize_cuda_and_wait_for_the_gpu(self, go_marker):
        """Release the inferior to initialize CUDA and wait for the cuInit
        breakpoint to bring up the GPU target."""
        self.setAsync(True)
        open(go_marker, "w").close()
        self.assertSuccess(self.cpu_process.Continue(), "continue CPU process")
        self.assertTrue(
            self.wait_for(lambda: self.gpu_target is not None),
            "GPU target was not created once the application initialized CUDA",
        )

    def test_attach_to_running_cuda_app(self):
        """Attach to a process whose CUDA kernel is already executing and verify
        that a GPU target comes up with the kernel's threads."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        popen = self.start_resident_kernel(exe, ready_marker)

        self.attach_to_running_cuda_app(popen.pid)

        self.find_thread_by_function("spinKernel")

    def test_attach_without_waiting_for_the_gpu(self):
        """With wait-for-gpu-attach off, process attach returns as soon as the
        CPU is attached, and the GPU target comes up once the process runs."""
        self.runCmd("settings set plugin.process.gdb-remote.wait-for-gpu-attach false")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings clear plugin.process.gdb-remote.wait-for-gpu-attach"
            )
        )
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        popen = self.start_resident_kernel(exe, ready_marker)

        self.runCmd("process attach -p %d" % popen.pid)
        self.assertIsNone(self.gpu_target, "the GPU target came up before the run")

        self._run_until_the_gpu_attach_completes()

    def test_attach_times_out_waiting_for_the_gpu(self):
        """With no time to wait for the GPU attach, lldb does not run the
        process, so process attach returns without the GPU target, and the GPU
        target comes up once the process runs."""
        self._set_attach_wait_timeout_ms(0)
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        popen = self.start_resident_kernel(exe, ready_marker)

        self.runCmd("process attach -p %d" % popen.pid)
        self.assertIsNone(self.gpu_target, "the GPU attach finished within 0 ms")

        self._run_until_the_gpu_attach_completes()

    def test_attach_before_cuda_is_initialized(self):
        """Attach while libcuda is not loaded yet. There is nothing to hand off
        then, so lldb skips the attach handshake and the GPU target
        comes up through the launch-style initialization breakpoints once the
        application initializes CUDA."""
        self.build()
        popen, go_marker = self._start_waiting_to_initialize_cuda()

        self.runCmd("process attach -p %d" % popen.pid)

        self._initialize_cuda_and_wait_for_the_gpu(go_marker)

    def test_attach_before_the_driver_is_initialized(self):
        """Attach while libcuda is loaded but CUDA is not initialized, so the
        driver has not published its attach descriptor yet. process attach must
        not wait for a GPU attach that cannot start, and the GPU target comes up
        through the launch-style initialization breakpoints once the application
        initializes CUDA."""
        # Long enough that an attach that waits cannot pass for one that did not.
        self._set_attach_wait_timeout_ms(60 * 1000)
        self.build()
        # Loading libcuda does not initialize the driver; cuInit does.
        popen, go_marker = self._start_waiting_to_initialize_cuda(
            ["LD_PRELOAD=libcuda.so.1"]
        )

        start = time.time()
        self.runCmd("process attach -p %d" % popen.pid)
        self.assertLess(
            time.time() - start,
            30,
            "process attach waited for a GPU attach that could not start",
        )
        self.assertIsNone(self.gpu_target, "the GPU target came up before cuInit")

        self._initialize_cuda_and_wait_for_the_gpu(go_marker)
