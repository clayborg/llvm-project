"""
Test NVGPU core file debugging functionality.
"""

import lldb

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase


class TestNVGPUCoreFiles(NVGPUCoreTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    NUM_BLOCKS = 32
    THREADS_PER_BLOCK = 4
    SOURCE = "corefiles.cu"

    # Maps the exception-type command-line argument to the (stop reason,
    # description substring) the kernel triggers. See the switch in
    # crash_kernel_2 in corefiles.cu. An argument that hits no switch case
    # spins without faulting and has no stop reason.
    STOP_REASONS = {
        1: (lldb.eStopReasonSignal, "trap"),
        2: (lldb.eStopReasonException, "CUDA Exception: Warp MMU Fault"),
        3: (lldb.eStopReasonException, "CUDA Exception: Warp Misaligned Address"),
        4: (lldb.eStopReasonException, "CUDA Exception: Warp Out-of-range Address"),
        5: (lldb.eStopReasonException, "CUDA Exception: Warp Misaligned PC"),
        6: (lldb.eStopReasonException, "CUDA Exception: Warp Invalid Address Space"),
    }

    def get_thread_name(self, block, thread):
        return f"blockIdx(x={block} y=0 z=0) threadIdx(x={thread} y=0 z=0)"

    def get_thread(self, block, thread):
        return self.find_thread_by_name(self.get_thread_name(block, thread))

    def get_stopped_thread(self):
        return self.get_thread(13, 0)

    def test_core_files(self):
        """Exercise all NVGPU core file behaviors within a single run directory."""
        self._test_thread_list()
        self._test_stop_reasons()
        self._test_backtrace("")
        self._test_backtrace("skip_nonrelocated_elf_images")
        self._test_backtrace("faulted_contexts_only")
        self._test_backtrace("no_errbar_at_exit")
        self._test_read_global_memory()
        self._test_read_constant_memory()
        self._test_read_shared_memory()
        self._test_read_local_memory()
        self._test_frame_variables()
        self._test_read_pc()
        self._test_register_sanity()

    def _test_thread_list(self):
        """Thread list shows block/thread coordinates from the kernel launch."""
        _, gpu_process = self.generate_and_load_core()

        self.assertEqual(gpu_process.GetNumThreads(), self.THREADS_PER_BLOCK * self.NUM_BLOCKS)

        thread = self.get_stopped_thread()
        self.assertTrue(thread.IsValid())

    def _test_stop_reasons(self):
        """Each exception type, selected via a command-line argument, produces
        the expected stop reason on the faulting CTA."""
        for exception_type, (expected_reason, expected_desc) in self.STOP_REASONS.items():
            self.generate_and_load_core(args=[str(exception_type)])
            thread = self.get_stopped_thread()
            self.assertTrue(thread.IsValid())
            self.assertIn(expected_desc, thread.GetStopDescription(256))
            self.assertEqual(thread.GetStopReason(), expected_reason)

    def _test_backtrace(self, flags):
        """Cubin modules from the core file provide symbols and debug info for backtraces."""
        gpu_target, gpu_process = self.generate_and_load_core(flags=flags)
        thread = self.get_stopped_thread()
        gpu_process.SetSelectedThread(thread)

        # Relocated cubin images embedded in the core become target modules.
        self.assertGreater(
            gpu_target.GetNumModules(),
            0,
            "Core file should load at least one module",
        )

        # Symbols from embedded cubins resolve for image lookup.
        self.expect("image lookup -n crash_kernel", substrs=["crash_kernel"])

        expected_frames = [
            "crash_kernel_2",
            "crash_kernel_1",
            "crash_kernel_0",
            "crash_kernel"
        ]

        # Lower bound on backtrace frames.
        self.assertGreaterEqual(
            thread.GetNumFrames(),
            len(expected_frames),
            f"Expected at least {len(expected_frames)} backtrace frames, "
            f"got {thread.GetNumFrames()}",
        )

        # Check order and symbol names of backtrace frames.
        for i, expected_func in enumerate(
            expected_frames
        ):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid(), f"Frame {i} should be valid")

            actual_func = frame.GetFunctionName()
            self.assertIsNotNone(actual_func, f"Frame {i} function name is None")
            self.assertEqual(
                expected_func,
                actual_func,
                f"Frame {i} expected '{expected_func}', got '{actual_func}'",
            )

        self.assertLess(
            thread.GetNumFrames(),
            20,
            "Backtrace should terminate without excessive frames",
        )

    def _test_read_global_memory(self):
        """Global memory in global_data matches values written by the faulting CTA."""

        _, gpu_process = self.generate_and_load_core()
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        for block in range(self.NUM_BLOCKS):
            offset = block * self.THREADS_PER_BLOCK
            expected = " ".join(f"0x{x+offset:08x}" for x in range(self.THREADS_PER_BLOCK))
            self.expect(
                f"memory read &global_data[{block}][0] "
                f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
                substrs=[expected],
            )

        # Check that missing global memory is handled gracefully.
        _, gpu_process = self.generate_and_load_core(flags="skip_global_memory")
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        self.expect(
            f"memory read &global_data[0][0] "
            f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
            error=True,
            substrs=["core file does not contain"],
        )

    def _test_read_constant_memory(self):
        """Constant memory in constant_data matches the static initializer values."""

        _, gpu_process = self.generate_and_load_core()
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        self.expect(
            f"memory read -p const &constant_data[0] "
            f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
            substrs=["0x10000000 0x10000001 0x10000002 0x10000003"],
        )

        # Check that missing constant memory is handled gracefully.
        _, gpu_process = self.generate_and_load_core(
            flags="skip_global_memory,skip_constbank_memory")
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        self.expect(
            f"memory read -p const &constant_data[0] "
            f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
            error=True,
            substrs=["core file does not contain"],
        )

    def _test_read_shared_memory(self):
        """Shared memory in shared_data matches values written by the faulting CTA."""

        _, gpu_process = self.generate_and_load_core()
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        gpu_process.GetSelectedThread().SetSelectedFrame(3)
        self.expect(
            f"memory read -p shared &shared_data[0] "
            f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
            substrs=["0xabcd0000 0xabcd0001 0xabcd0002 0xabcd0003"],
        )

        # Check that missing shared memory is handled gracefully.
        _, gpu_process = self.generate_and_load_core(flags="skip_shared_memory")
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        gpu_process.GetSelectedThread().SetSelectedFrame(3)
        self.expect(
            f"memory read -p shared &shared_data[0] "
            f"--format x --size 4 -c {self.THREADS_PER_BLOCK}",
            error=True,
            substrs=["core file does not contain"],
        )

    def _test_read_local_memory(self):
        """Local memory at middle_scalar matches the value written by the faulting lane."""
        _, gpu_process = self.generate_and_load_core()
        thread = self.get_stopped_thread()
        gpu_process.SetSelectedThread(thread)

        # FIXME: We should be able to use `&middle_scalar` here, once backtrace is more robust.

        # skip_local_memory also drops the debug info for middle_scalar, so resolve
        # the address of middle_scalar here.
        frame = thread.GetFrameAtIndex(1)
        var = frame.FindVariable("middle_scalar")
        self.assertTrue(var.IsValid(), "Should find 'middle_scalar' variable")
        self.assertTrue(
            var.GetError().Success(),
            f"Reading 'middle_scalar' failed: {var.GetError().GetCString()}",
        )
        local_addr = var.GetLoadAddress()
        self.assertNotEqual(local_addr, lldb.LLDB_INVALID_ADDRESS)

        self.expect(
            f"memory read -p local {local_addr:#x} --format x --size 4 -c 1",
            substrs=["0xcafebabe"],
        )

        # Check that missing local memory is handled gracefully.
        _, gpu_process = self.generate_and_load_core(flags="skip_local_memory")
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        self.expect(
            f"memory read -p local {local_addr:#x} --format x --size 4 -c 1",
            error=True,
            substrs=["core file does not contain"],
        )

    def _test_frame_variables(self):
        """Typed locals (scalar, array, struct) in a middle (caller) frame are
        reconstructed after unwinding and format correctly via `frame variable`."""
        _, gpu_process = self.generate_and_load_core()
        thread = self.get_stopped_thread()
        gpu_process.SetSelectedThread(thread)

        thread.SetSelectedFrame(1)
        self.assertEqual(thread.GetSelectedFrame().GetFunctionName(), "crash_kernel_1")
        self.expect(
            "frame variable middle_scalar -f hex",
            substrs=["middle_scalar = 0xcafebabe"]
        )
        self.expect(
            "frame variable middle_array -f hex",
            substrs=[
                "middle_array",
                "[0] = 0x00000010",
                "[1] = 0x00000011",
            ],
        )
        self.expect(
            "frame variable middle_point -f hex",
            substrs=["middle_point", "x = 0x0000000d", "y = 0x00000000"],
        )

    def _test_read_pc(self):
        """PC register on the trapped lane points within the trap source line."""
        gpu_target, gpu_process = self.generate_and_load_core()
        gpu_process.SetSelectedThread(self.get_stopped_thread())

        # Resolve the address range of the fault source line.
        file_spec = lldb.SBFileSpec(self.SOURCE)
        fault_line = line_number(self.SOURCE, "*TRAP*")
        bp = gpu_target.BreakpointCreateByLocation(file_spec, fault_line)
        self.assertGreaterEqual(bp.GetNumLocations(), 1)
        line_entry = bp.GetLocationAtIndex(0).GetAddress().GetLineEntry()
        line_start = line_entry.GetStartAddress().GetLoadAddress(gpu_target)
        line_end = line_entry.GetEndAddress().GetLoadAddress(gpu_target)
        self.assertNotEqual(line_start, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(line_end, lldb.LLDB_INVALID_ADDRESS)

        # Ensure the PC register points within the trap source line.
        frame = gpu_process.GetSelectedThread().GetFrameAtIndex(0)
        pc = frame.FindRegister("PC").GetValueAsUnsigned()
        self.assertGreaterEqual(pc, line_start)
        self.assertLessEqual(pc, line_end)

    def _test_register_sanity(self):
        """RZ, URZ, and register groups are correct on the faulting lane."""
        _, gpu_process = self.generate_and_load_core()
        gpu_process.SetSelectedThread(self.get_stopped_thread())
        frame = gpu_process.GetSelectedThread().GetFrameAtIndex(0)

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

        regular_registers = frame.GetRegisters().GetFirstValueByName("Regular Registers")
        rz = regular_registers.GetChildAtIndex(regular_registers.GetNumChildren() - 1)
        self.assertEqual(rz.GetName(), "RZ")

        uniform_registers = frame.GetRegisters().GetFirstValueByName("Uniform Registers")
        urz = uniform_registers.GetChildAtIndex(uniform_registers.GetNumChildren() - 1)
        self.assertEqual(urz.GetName(), "URZ")
