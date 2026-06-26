import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPURegisters(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def read_vec3_register(self, frame, name):
        """Read a 3-component (dim3/uint3) vector register as [x, y, z]."""
        reg = frame.FindRegister(name)
        self.assertTrue(reg.IsValid(), f"{name} should be a valid register")
        data = reg.GetData()
        vals = []
        for i in range(3):
            err = lldb.SBError()
            vals.append(data.GetUnsignedInt32(err, i * 4))
            self.assertTrue(
                err.Success(), f"reading {name}[{i}] failed: {err.GetCString()}"
            )
        return vals

    def test_gpu_showing_registers(self):
        """Test that we know when the GPU has asserted."""
        self.killCPUOnTeardown()

        self.build()
        source = "registers.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.assertEqual(self.gpu_process.state, lldb.eStateStopped)
        some_thread_with_exception = self.find_thread_by_stop_reason(lldb.eStopReasonException)
        self.assertIn("CUDA Exception(6): Warp Misaligned Address at 0x", str(some_thread_with_exception))

        frame = some_thread_with_exception.frame[0]

        errorpc = frame.FindRegister("errorpc")
        self.assertNotEqual(errorpc.GetValueAsAddress(), lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(errorpc.GetValueAsAddress(), 0)

        r0 = frame.FindRegister("R0")
        self.assertTrue(r0.IsValid())

        rz = frame.FindRegister("RZ")
        self.assertTrue(rz.IsValid())
        self.assertEqual(rz.GetValueAsUnsigned(), 0)

        ur0 = frame.FindRegister("UR0")
        self.assertTrue(ur0.IsValid())

        urz = frame.FindRegister("URZ")
        self.assertTrue(urz.IsValid())
        self.assertEqual(urz.GetValueAsUnsigned(), 0)

        p0 = frame.FindRegister("P0")
        self.assertTrue(p0.IsValid())

        up0 = frame.FindRegister("UP0")
        self.assertTrue(up0.IsValid())

        # We check that RZ and URZ are in the right group
        regular_registers = frame.GetRegisters().GetFirstValueByName("Regular Registers")
        rz = regular_registers.GetChildAtIndex(regular_registers.GetNumChildren() - 1)
        self.assertEqual(rz.GetName(), "RZ")

        uniform_registers = frame.GetRegisters().GetFirstValueByName("Uniform Registers")
        urz = uniform_registers.GetChildAtIndex(uniform_registers.GetNumChildren() - 1)
        self.assertEqual(urz.GetName(), "URZ")

        # CUDA builtin (virtual) registers
        threadIdx = self.read_vec3_register(frame, "threadIdx")
        self.assertLess(threadIdx[0], 256)
        self.assertEqual(threadIdx[1:], [0, 0])

        blockIdx = self.read_vec3_register(frame, "blockIdx")
        self.assertLess(blockIdx[0], 20)
        self.assertEqual(blockIdx[1:], [0, 0])

        blockDim = self.read_vec3_register(frame, "blockDim")
        self.assertEqual(blockDim, [256, 1, 1])

        gridDim = self.read_vec3_register(frame, "gridDim")
        self.assertEqual(gridDim, [20, 1, 1])

        warpSize = frame.FindRegister("warpSize")
        self.assertTrue(warpSize.IsValid())
        self.assertEqual(warpSize.GetValueAsUnsigned(), 32)

        builtins_regset = frame.GetRegisters().GetFirstValueByName("CUDA Builtins")
        self.assertTrue(builtins_regset.IsValid())
