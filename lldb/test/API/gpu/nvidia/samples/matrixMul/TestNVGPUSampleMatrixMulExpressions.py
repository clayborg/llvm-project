"""
Tests on the cuda-samples program matrixMul (see matrixmul.py): expressions,
which `p` and `expr` evaluate, in one of the kernel's threads, on its locals,
the kernel's parameters and the CUDA built-in variables. The tests run
LAUNCH_16, where blockDim is not the warp size. simpleCooperativeGroups tests
expressions on shared memory.
"""

from lldbsuite.test.decorators import expectedFailureAll

from matrixmul import LAUNCH_16, THREAD, MatrixMulTestCase


class TestNVGPUSampleMatrixMulExpressions(MatrixMulTestCase):
    def test_expressions_on_register_variables(self):
        """Expressions, which `p` and `expr` evaluate, read the kernel's locals
        that are in registers, and compute with them."""
        _, thread = self.run_to_thread("// gpu breakpoint at tile load")
        frame = thread.frame[0]
        (_, (tx, ty, _)) = THREAD
        for expression, expected in (
            ("ty", ty),
            ("tx + 32 * ty", tx + 32 * ty),
            ("aEnd - aBegin", LAUNCH_16.width - 1),
        ):
            self.assertEqual(
                self.evaluate_expression(frame, expression).GetValueAsSigned(),
                expected,
                expression,
            )
        self.assertEqual(
            self.value_as_float(self.evaluate_expression(frame, "Csub"), "Csub"),
            0.0,
        )
        self.finish()

    @expectedFailureAll(
        "expressions cannot read a kernel parameter, which is in parameter memory"
    )
    def test_expressions_read_kernel_parameters(self):
        """Expressions read the kernel's parameters, and compute with them and
        the locals in registers."""
        _, thread = self.run_to_thread("// gpu breakpoint at tile load")
        frame = thread.frame[0]
        (_, (tx, ty, _)) = THREAD
        width = LAUNCH_16.width
        for expression, expected in (("wA", width), ("wA * ty + tx", width * ty + tx)):
            self.assertEqual(
                self.evaluate_expression(frame, expression).GetValueAsSigned(),
                expected,
                expression,
            )
        self.finish()

    @expectedFailureAll(
        "expressions do not know the CUDA built-in variables threadIdx, "
        "blockIdx and blockDim"
    )
    def test_expressions_know_cuda_builtins(self):
        """Expressions read the CUDA built-in variables, and compute with
        them."""
        _, thread = self.run_to_thread("// gpu breakpoint at tile load")
        frame = thread.frame[0]
        (bx, by, _), (tx, ty, _) = THREAD
        for expression, expected in (
            ("threadIdx.x", tx),
            ("blockIdx.y", by),
            ("blockDim.x * blockIdx.x + threadIdx.x", LAUNCH_16.block * bx + tx),
        ):
            self.assertEqual(
                self.evaluate_expression(frame, expression).GetValueAsSigned(),
                expected,
                expression,
            )
        self.finish()
