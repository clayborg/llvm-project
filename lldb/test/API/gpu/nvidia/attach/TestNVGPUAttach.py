import lldb
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUAttach(NVGPUTestCaseBase):
    """Tests for attaching to an already-running CUDA application.

    Unlike the other NVGPU tests, which launch the inferior under the debugger
    and rely on the cuInit-style initialization breakpoint, this test starts the
    CUDA process independently (with a resident kernel already executing) and
    then attaches to it. The lldb-server NVGPU plugin must transparently detect
    the running CUDA driver, initiate the safe attach procedure, bring up the
    GPU target, and surface the in-flight kernel's threads in the thread list.
    """

    NO_DEBUG_INFO_TESTCASE = True

    def test_attach_to_running_cuda_app(self):
        """Attach to a process whose CUDA kernel is already executing and verify
        that a GPU target comes up with the kernel's threads."""
        self.skip_if_no_cuda_device()

        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        popen = self.start_resident_kernel(exe, ready_marker)

        self.runCmd("process attach -p %d" % popen.pid)

        cpu_target = self.cpu_target
        self.assertTrue(cpu_target and cpu_target.IsValid(), "no CPU target after attach")

        # The driver needs the application running to inject the debug engine
        # and call CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED. This has to be async:
        # the CPU host of a resident kernel never stops on its own, so a
        # synchronous Continue() would block forever.
        self.setAsync(True)
        cpu_process = cpu_target.GetProcess()
        self.assertTrue(cpu_process and cpu_process.IsValid(), "no CPU process after attach")
        cpu_process.Continue()

        self.assertTrue(
            self.wait_for(lambda: self.gpu_target is not None),
            "GPU target was not created after attaching to the running CUDA app",
        )
        self.assertTrue(
            self.wait_for_gpu_process_stopped(),
            "GPU process did not stop after attach",
        )

        self.select_gpu()
        self.assertGreater(
            len(self.gpu_process.threads),
            0,
            "expected the attached kernel's threads to appear in the thread list",
        )
