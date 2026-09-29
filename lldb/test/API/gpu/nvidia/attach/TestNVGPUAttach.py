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
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_marker = self.getBuildArtifact("kernel_ready.marker")
        popen = self.start_resident_kernel(exe, ready_marker)

        self.attach_to_running_cuda_app(popen.pid)

        self.select_gpu()
        self.assertGreater(
            len(self.gpu_process.threads),
            0,
            "expected the attached kernel's threads to appear in the thread list",
        )
