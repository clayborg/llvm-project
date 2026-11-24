"""
Test reading and writing the variables of a stopped GPU thread.

Register-backed variables round-trip through `expr`.
"""

import lldb
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "variables.cu"

# Values the kernel stores, mirrored here for assertions.
REG_SCALAR_BASE = 0x1000  # reg_scalar = tid + 0x1000


class TestNVGPUVariables(NVGPUTestCaseBase):
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
        triple_breakpoint = self.run_to_gpu_line(
            SOURCE,
            "// breakpoint in triple",
            exe_name=self.build_shared("a.out"),
        )
        accumulate_breakpoint = self.gpu_breakpoint(
            SOURCE, "// breakpoint in accumulate"
        )

        thread = self.assert_gpu_stop_at_breakpoint(triple_breakpoint)[0]
        frame = thread.GetFrameAtIndex(0)
        self.assert_frame_at(frame, "triple(int)", SOURCE, "// breakpoint in triple")
        index = self.global_index(thread)
        self.assert_int_variable(frame, "value", index)
        self.assert_int_variable(frame, "tripled", 3 * index)

        triple_breakpoint.SetEnabled(False)
        thread = self.continue_gpu_until_breakpoint(accumulate_breakpoint)[0]
        frame = thread.GetFrameAtIndex(0)
        self.assert_frame_at(
            frame, "accumulate(int*)", SOURCE, "// breakpoint in accumulate"
        )
        index = self.global_index(thread)
        self.assert_int_variable(frame, "index", index)
        self.assert_int_variable(frame, "total", 3 * index + 10)

        self.continue_gpu_to_exit()

    @expectedFailureAll("expressions cannot read a variable that is in local memory")
    def test_expressions_read_local_memory_variables(self):
        """Expressions, which `p` and `expr` evaluate, read a variable that is
        in a thread's local memory, such as an array, as `frame variable`
        does."""
        breakpoint = self.run_to_gpu_line(
            SOURCE,
            "// breakpoint in accumulate",
            exe_name=self.build_shared("a.out"),
        )
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].GetFrameAtIndex(0)
        self.assert_int_variable(frame, "steps[2]", 3)
        for expression, expected in (("steps[2]", 3), ("steps[0] + steps[3]", 5)):
            self.assertEqual(
                self.evaluate_expression(frame, expression).GetValueAsSigned(),
                expected,
                expression,
            )
        self.continue_gpu_to_exit()

    @expectedFailureAll("frame variables keep their old values after a register write")
    def test_frame_variable_shows_assigned_value(self):
        """`frame variable` shows the value that an expression assigned to a
        variable that lives in a register."""
        breakpoint = self.run_to_gpu_line(
            SOURCE, "// gpu breakpoint", exe_name=self.build_shared("a.out")
        )
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].GetFrameAtIndex(0)
        # lldb keeps the value it reads here.
        self.assertNotEqual(self.read_int_variable(frame, "reg_scalar"), 0x7ABC)
        self.evaluate_expression(frame, "reg_scalar = 0x7abc")
        self.assert_int_variable(frame, "reg_scalar", 0x7ABC)
        self.continue_gpu_to_exit()

    @expectedFailureAll("a variable that lives in two registers is not written")
    def test_expressions_assign_64_bit_variables(self):
        """An expression assigns a 64-bit variable, which lives in two
        registers."""
        breakpoint = self.run_to_gpu_line(
            SOURCE, "// gpu breakpoint", exe_name=self.build_shared("a.out")
        )
        frame = self.assert_gpu_stop_at_breakpoint(breakpoint)[0].GetFrameAtIndex(0)
        self.assertNotEqual(self.read_float_variable(frame, "reg_double"), 9.25)
        self.evaluate_expression(frame, "reg_double = 9.25")
        value = self.evaluate_expression(frame, "reg_double")
        self.assertEqual(self.value_as_float(value, "reg_double"), 9.25)
        self.continue_gpu_to_exit()

    def test_variables_at_gpu_breakpoint(self):
        """Round-trip a variable that lives in a register through `expr` at
        the GPU breakpoint, and check that the kernel then uses the value
        written, in the lane it was written in only."""
        self.run_to_cpu_line(
            SOURCE,
            "// before kernel launch",
            exe_name=self.build_shared("a.out"),
        )
        breakpoint = self.gpu_breakpoint(SOURCE, "// gpu breakpoint")
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(breakpoint)
        self.select_gpu()

        self._test_roundtrip_register_backed()

        self.continue_gpu_to_exit()
        # The program prints what lanes 0 and 1 stored after the breakpoint:
        # the last value written in lane 0, and lane 1's own.
        self.assert_stdout_contains(f"reg_scalar: 0x1234 {REG_SCALAR_BASE + 1:#x}")

    def _test_roundtrip_register_backed(self):
        """Round-trip a register-backed scalar through expr."""
        frame = self.select_gpu_thread((0,), (0,)).GetFrameAtIndex(0)

        self.assertEqual(
            self.evaluate_expression(frame, "reg_scalar").GetValueAsUnsigned(),
            REG_SCALAR_BASE,
        )
        self.expr_roundtrip(frame, "reg_scalar", 0x7ABC)
        self.expr_roundtrip(frame, "reg_scalar", 0x1234)

    def run_to_scale(self) -> lldb.SBThread:
        """Run to the breakpoint in Scaler::scale, which scale_all calls on an
        object in its thread's local memory, and return a thread there."""
        breakpoint = self.run_to_gpu_line(
            SOURCE,
            "// breakpoint in scale",
            exe_name=self.build_shared("a.out"),
        )
        thread = self.assert_gpu_stop_at_breakpoint(breakpoint)[0]
        self.assert_frame_at(
            thread.GetFrameAtIndex(0),
            "Scaler::scale(int) const",
            SOURCE,
            "// breakpoint in scale",
        )
        # The object reads correctly by its name in the caller's frame.
        self.assert_int_variable(thread.GetFrameAtIndex(1), "scaler.factor", 2)
        return thread

    @expectedFailureAll(
        "`frame variable` reads what a const pointer such as `this` points to "
        "in global memory, not in generic memory, so it cannot read a member "
        "of an object in the thread's local memory"
    )
    def test_frame_variable_reads_members_through_this(self):
        """In a device member function, `frame variable this->factor` reads a
        member of the object that the function was called on."""
        frame = self.run_to_scale().GetFrameAtIndex(0)
        self.assert_int_variable(frame, "this->factor", 2)
        self.continue_gpu_to_exit()

    @expectedFailureAll(
        "expressions cannot use `this` in a device member function: lldb "
        "reports \"invalid use of 'this' outside of a non-static member "
        'function"'
    )
    def test_expressions_read_members_through_this(self):
        """In a device member function, expressions, which `p` and `expr`
        evaluate, read the object's members through `this` and by their
        names, as they do in a host member function. The object is in the
        thread's local memory, so this also needs expressions to read
        memory through a generic pointer, which they cannot do yet."""
        thread = self.run_to_scale()
        frame = thread.GetFrameAtIndex(0)
        index = self.global_index(thread)
        for expression, expected in (
            ("this->factor", 2),
            ("factor * value", 2 * index),
        ):
            self.assertEqual(
                self.evaluate_expression(frame, expression).GetValueAsSigned(),
                expected,
                expression,
            )
        self.continue_gpu_to_exit()
