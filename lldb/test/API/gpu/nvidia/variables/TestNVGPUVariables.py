"""
Test reading the variables of a stopped GPU thread.
"""

import lldb
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "variables.cu"


class TestNVGPUVariables(NVGPUTestCaseBase):
    def assert_variable(self, frame: lldb.SBFrame, name: str, expected: int) -> None:
        variable = frame.FindVariable(name)
        self.assertSuccess(variable.GetError(), f"reading {name}")
        self.assertEqual(variable.GetValueAsSigned(), expected, name)

    def global_index(self, thread: lldb.SBThread) -> int:
        """The thread's index in the grid, which the program computes as
        `index`."""
        (block_x, _, _), (thread_x, _, _) = self.thread_coords(thread)
        block_dim = self.read_vec3_register(thread.GetFrameAtIndex(0), "blockDim")
        return block_x * block_dim[0] + thread_x

    def test_register_variables_in_every_function(self):
        """Variables in PTX registers read correctly in every function. The
        cubin maps each function's PTX registers to SASS registers in a block of
        its own, and the functions here reuse the same PTX register names."""
        self.build()
        self.run_to_cpu_line(SOURCE, "// cpu breakpoint before launch")
        triple_breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in triple")
        )
        accumulate_breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// breakpoint in accumulate")
        )

        self.continue_cpu_and_wait_for_gpu_to_stop()
        thread = self.assert_gpu_stop_at_breakpoint(triple_breakpoint)[0]
        frame = thread.GetFrameAtIndex(0)
        self.assert_frame_at(frame, "triple(int)", SOURCE, "// breakpoint in triple")
        index = self.global_index(thread)
        self.assert_variable(frame, "value", index)
        self.assert_variable(frame, "tripled", 3 * index)

        triple_breakpoint.SetEnabled(False)
        thread = self.continue_gpu_until_breakpoint(accumulate_breakpoint)[0]
        frame = thread.GetFrameAtIndex(0)
        self.assert_frame_at(
            frame, "accumulate(int*)", SOURCE, "// breakpoint in accumulate"
        )
        index = self.global_index(thread)
        self.assert_variable(frame, "index", index)
        self.assert_variable(frame, "total", 3 * index + 10)

        self.gpu_target.DeleteAllBreakpoints()
        self.assertSuccess(self.gpu_process.Continue())
        self.assert_exit_status(0)
