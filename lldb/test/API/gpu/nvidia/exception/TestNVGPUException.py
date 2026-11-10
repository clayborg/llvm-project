"""
Test what the program sees after a GPU exception.
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUException(NVGPUTestCaseBase):
    @expectedFailureAll(
        "after a continue from a GPU exception, the program's CUDA calls "
        "report success, so the program never learns that its kernel failed"
    )
    def test_continue_after_a_fault_does_not_report_success(self):
        """After `continue` from the stop at a kernel's fault, the program is
        not told that its kernel succeeded. Without a debugger,
        cudaDeviceSynchronize returns cudaErrorIllegalAddress, which this
        program expects: it exits 1 only if the call returns cudaSuccess."""
        self.build()
        self.launch_and_wait_for_gpu_target(self.createTestTarget())
        self.wait_for_gpu_to_stop()
        self.assert_gpu_stop_reason(lldb.eStopReasonException)

        self.setAsync(True)
        self.assertSuccess(self.gpu_process.Continue(), "continue GPU process")
        status = self.wait_for_exit(self.cpu_process)
        self.assertNotEqual(
            status, 1, f"exit status; the program's output:\n{self.read_stdout()}"
        )
