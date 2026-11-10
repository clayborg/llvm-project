"""
Test `process kill` at a GPU stop, on the CPU target and on the GPU target.
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "kill.cu"


class TestNVGPUKill(NVGPUTestCaseBase):
    def run_to_gpu_stop(self) -> None:
        """Run the program to the breakpoint in its kernel, and pull the stop
        that the CPU makes with the GPU's."""
        self.run_to_cpu_line(
            SOURCE,
            "// cpu breakpoint before launch",
            exe_name=self.build_shared("kill"),
        )
        breakpoint = self.gpu_breakpoint(SOURCE, "// breakpoint in kernel")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.one_thread_at(breakpoint)
        if self.cpu_stops_with_gpu():
            # A kill of a CPU process that has stopped but still looks
            # running interrupts it first, and waits for a stop that is
            # already queued.
            self.assert_process_enters(self.cpu_process, lldb.eStateStopped, "CPU")

    def assert_process_enters(
        self, process: lldb.SBProcess, state: int, what: str, timeout_seconds: int = 5
    ) -> None:
        """`process`, the `what` ("CPU" or "GPU") process, must be in `state`
        within `timeout_seconds`. Unlike the wait helpers, this does not need
        the state's event to be still queued, which a kill may have taken."""
        if not self.wait_until(
            lambda: self.pull_state(process) == state, timeout_seconds
        ):
            self.fail(
                f"the {what} process is "
                f"{lldb.SBDebugger.StateAsCString(process.GetState())}, not "
                f"{lldb.SBDebugger.StateAsCString(state)}, after "
                f"{timeout_seconds} s"
            )

    def test_variable_read_fails_after_kill_of_the_cpu_process(self):
        """Reading a variable of a GPU thread after `process kill` on the CPU
        target fails with an error: the variable is in a register, which
        lldb-server can no longer read."""
        self.run_to_gpu_stop()
        # Not read before the kill, so lldb has no value of it to show.
        frame = self.gpu_process.GetSelectedThread().frame[0]
        self.assertSuccess(self.cpu_process.Kill(), "kill the CPU process")
        self.assert_process_enters(self.cpu_process, lldb.eStateExited, "CPU")
        value = frame.GetValueForVariablePath("sum")
        self.assertTrue(
            value.GetError().Fail(),
            f"reading sum after the kill gave {value.GetValue()}",
        )

    @expectedFailureAll(
        "nothing tells the GPU target that the program is gone, so after a "
        "kill of the CPU process the GPU process stays stopped, with the "
        "threads of its last stop"
    )
    def test_kill_of_the_cpu_process_ends_the_gpu_process(self):
        """`process kill` on the CPU target at a GPU stop ends the program,
        and the GPU process exits with it, as it does when the program exits
        by itself."""
        self.run_to_gpu_stop()
        self.assertSuccess(self.cpu_process.Kill(), "kill the CPU process")
        self.assert_process_enters(self.cpu_process, lldb.eStateExited, "CPU")
        self.assert_process_enters(self.gpu_process, lldb.eStateExited, "GPU")

    @expectedFailureAll(
        "lldb-server's GPU process has no kill: lldb waits a packet timeout "
        "for the reply, then calls the GPU process exited, and the program "
        "is still there"
    )
    def test_kill_of_the_gpu_process_ends_the_program(self):
        """`process kill` at a GPU stop, where the GPU target is the selected
        one, ends the program: both processes exit."""
        # Until the GPU process can be killed, the kill costs a packet
        # timeout. The test suite sets it to 60 s before each test; this is
        # lldb's default.
        self.runCmd("settings set plugin.process.gdb-remote.packet-timeout 5")
        self.run_to_gpu_stop()
        self.assertSuccess(self.gpu_process.Kill(), "kill the GPU process")
        self.assert_process_enters(self.cpu_process, lldb.eStateExited, "CPU")
        self.assert_process_enters(self.gpu_process, lldb.eStateExited, "GPU")
