"""
Tests on the cuda-samples program asyncAPI, copied in asyncAPI.cu. The CPU
spins on cudaEventQuery while the kernel increments an array, and the program
exits 1 if the array it copies back is wrong, so its exit status is its check.
"""

from typing import List

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "asyncAPI.cu"


class TestNVGPUSampleAsyncAPI(NVGPUTestCaseBase):
    def run_to_launch(self) -> None:
        """Run the program to its `// cpu breakpoint before launch` line.

        CUDA is initialized there and no kernel has run yet, so the GPU target
        exists and a breakpoint set on it now is in place for the launch.
        """
        self.run_to_cpu_line(
            SOURCE,
            "// cpu breakpoint before launch",
            exe_name=self.build_shared("asyncAPI"),
        )
        self.assertEqual(
            self.dbg.GetNumTargets(), 2, "CUDA is initialized, so there is a GPU target"
        )

    def read_global_ints(self, address: int, count: int) -> List[int]:
        """Read `count` ints from GPU global memory at `address`. lldb asks
        lldb-server for GPU memory on every read, while it keeps a stopped
        thread's registers for the whole stop."""
        error = lldb.SBError()
        space = self.gpu_process.GetAddressSpaceID("global", error)
        self.assertSuccess(error)
        data = self.gpu_process.ReadMemory(
            lldb.SBProcessAddress(address, space), 4 * count, error
        )
        self.assertSuccess(error)
        return [
            int.from_bytes(data[i : i + 4], "little", signed=True)
            for i in range(0, len(data), 4)
        ]

    def test_runs_to_exit(self):
        """asyncAPI runs to exit under LLDB, and its own check of the data the
        GPU computed passes."""
        self.run_to_launch()
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue())
        self.assert_exit_status(0)
        self.assertIn("CPU executed", self.read_stdout())

    def test_cpu_stepi_does_not_move_gpu(self):
        """While the GPU is stopped at a breakpoint, stepping the CPU by
        instruction moves only the CPU: the GPU process is not resumed, and no
        GPU thread stores its element of the array."""
        self.run_to_launch()
        gpu_breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// gpu breakpoint")
        )
        wait_breakpoint = self.cpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// cpu breakpoint while waiting")
        )

        self.continue_cpu_and_wait_for_gpu_to_stop()
        thread = self.assert_gpu_stop_at_breakpoint(gpu_breakpoint)[0]
        frame = thread.frame[0]
        self.assert_frame_at(
            frame, "increment_kernel(int*, int)", SOURCE, "// gpu breakpoint"
        )
        grid_dim = self.read_vec3_register(frame, "gridDim")
        block_dim = self.read_vec3_register(frame, "blockDim")
        self.assertEqual(grid_dim, [2, 1, 1])
        self.assertEqual(block_dim, [512, 1, 1])
        block = self.read_vec3_register(frame, "blockIdx")
        index = self.read_vec3_register(frame, "threadIdx")
        idx = frame.FindVariable("idx")
        self.assertSuccess(idx.GetError())
        self.assertEqual(idx.GetValueAsSigned(), block[0] * block_dim[0] + index[0])
        inc_value = frame.FindVariable("inc_value")
        self.assertSuccess(inc_value.GetError())
        self.assertEqual(inc_value.GetValueAsSigned(), 26)

        # The host zeroed the array and copied it to the GPU before the launch,
        # and the breakpoint is on the line that stores each thread's element,
        # so no thread has stored one yet. Reading the array through the
        # pointer's value (`frame variable g_data[0]`) fails for now, so read
        # it from global memory, where g_data points.
        g_data = frame.FindVariable("g_data")
        self.assertSuccess(g_data.GetError())
        array = g_data.GetValueAsUnsigned()
        n = grid_dim[0] * block_dim[0]
        self.assertEqual(self.read_global_ints(array, n), [0] * n)

        # The CPU waits in its loop for the GPU work to finish.
        self.wait_for_stop(self.cpu_process)
        cpu_threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.cpu_process, wait_breakpoint
        )
        self.assertEqual(len(cpu_threads), 1, "the CPU stopped at no breakpoint")
        cpu_thread = cpu_threads[0]
        self.assert_frame_at(
            cpu_thread.frame[0], "main", SOURCE, "// cpu breakpoint while waiting"
        )

        gpu_stop_id = self.gpu_process.GetStopID()
        self.setAsync(False)
        for _ in range(2):
            cpu_pc = cpu_thread.frame[0].GetPC()
            error = lldb.SBError()
            cpu_thread.StepInstruction(False, error)
            self.assertSuccess(error)
            self.assertNotEqual(cpu_thread.frame[0].GetPC(), cpu_pc)
        self.assertEqual(cpu_thread.frame[0].GetFunctionName(), "main")

        # Had the GPU been resumed, lldb would have reported a state change of
        # the GPU process, and its threads would have stored their elements.
        # The GPU threads' PCs prove nothing here: lldb would show the ones it
        # read at the stop.
        event = lldb.SBEvent()
        self.assertFalse(
            self.dbg.GetListener().WaitForEventForBroadcasterWithType(
                1,
                self.gpu_process.GetBroadcaster(),
                lldb.SBProcess.eBroadcastBitStateChanged,
                event,
            ),
            "the GPU process changed state while the CPU stepped",
        )
        self.assertState(self.gpu_process.GetState(), lldb.eStateStopped)
        self.assertEqual(self.gpu_process.GetStopID(), gpu_stop_id)
        self.assertEqual(
            self.read_global_ints(array, n),
            [0] * n,
            "GPU threads stored their elements while the CPU stepped",
        )

        # Let the sample finish: it checks that every element was incremented.
        self.gpu_target.BreakpointDelete(gpu_breakpoint.GetID())
        self.cpu_target.BreakpointDelete(wait_breakpoint.GetID())
        self.setAsync(True)
        self.assertSuccess(self.gpu_process.Continue())
        self.assertSuccess(self.cpu_process.Continue())
        self.assert_exit_status(0)
