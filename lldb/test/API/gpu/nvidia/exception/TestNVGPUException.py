"""
Test how a GPU exception or trap is reported, and what the program sees after
one.
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "exception.cu"
EXE = "exception"


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
        self.launch_and_wait_for_gpu_target(self.make_shared_target(EXE))
        self.wait_for_gpu_to_stop()
        self.assert_gpu_stop_reason(lldb.eStopReasonException)

        self.setAsync(True)
        self.assertSuccess(self.gpu_process.Continue(), "continue GPU process")
        status = self.wait_for_exit(self.cpu_process)
        self.assertNotEqual(
            status, 1, f"exit status; the program's output:\n{self.read_stdout()}"
        )

    @expectedFailureAll(
        "lldb-server reports a `__trap()` as a breakpoint stop, and lldb, which "
        "has no breakpoint at that address, resumes the GPU past the trap "
        "without showing a stop"
    )
    def test_device_trap_stops_the_gpu(self):
        """`__trap()` in device code stops the GPU, as a failed `assert()`
        does. The thread that ran it reports SIGTRAP with a "trap" stop
        description, matching the core-file reader, and its backtrace shows the
        kernel at the call."""
        self.launch_and_wait_for_gpu_target(self.make_shared_target(EXE, args=["trap"]))
        # The kernel is the program's first CUDA call, so the trap follows the
        # GPU target at once. Resume the CPU if lldb-server holds it in a
        # target-sync stop without a matching GPU stop, so a missed trap lets
        # the program finish instead of timing out.
        self.wait_for_gpu_to_stop(keep_cpu_running=True)

        thread = self.find_some_thread(
            lambda thread: self.thread_coords(thread) == ((0, 0, 0), (5, 0, 0)),
            description="with threadIdx.x 5, the one that runs __trap()",
        )
        self.assertStopReason(
            thread.GetStopReason(),
            lldb.eStopReasonSignal,
            f"stop reason of {thread.GetName()}, which ran __trap()",
        )
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            self.gpu_process.GetUnixSignals().GetSignalNumberFromName("SIGTRAP"),
            thread.GetStopDescription(256),
        )
        self.assertIn("trap", thread.GetStopDescription(256))
        # With -G, nvcc compiles __trap() as CALL.ABS.NOINC __trap, a function
        # whose first instruction is BPT.TRAP 0x1, so frame 0 is __trap and the
        # kernel is its caller.
        kernel = "trap_in_one_thread()"
        functions = [frame.GetFunctionName() for frame in thread.frames]
        self.assertIn(kernel, functions, f"backtrace of {thread.GetName()}")
        self.assert_frame_at(
            thread.frames[functions.index(kernel)], kernel, SOURCE, "// trap"
        )
