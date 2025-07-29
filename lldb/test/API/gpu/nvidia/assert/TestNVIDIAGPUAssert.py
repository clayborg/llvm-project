import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import TestBase, line_number

class TestNVIDIAGPUAssert(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        cpu = self.dbg.GetTargetAtIndex(0)
        gpu = self.dbg.GetTargetAtIndex(1)

        # We switch to async mode to wait for state changes in the GPU target while the CPU resumes.
        self.setAsync(True)
        listener = self.dbg.GetListener()
        cpu.process.Continue()
        lldbutil.expect_state_changes(self, listener, gpu.process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(gpu.process.state, lldb.eStateStopped)
        self.assertIn("CUDA Exception(12): Warp - Assert", str(gpu.process.thread[0]))

        # Now let's test that the disass can print at least one entry
        self.expect("disassemble", patterns=[".*cuda_elf.*\\.cubin`.*:.*"])

        frame = gpu.process.thread[0].frame[0]

        # We don't expect to see an errorpc set
        self.assertNotIn("CUDA Exception(12): Warp - Assert at 0x", str(gpu.process.thread[0]))
        errorpc = frame.FindRegister("errorpc").GetValueAsAddress()
        self.assertEqual(errorpc, lldb.LLDB_INVALID_ADDRESS)

        # We check we can read up to register R31.
        self.assertTrue(frame.FindRegister("R31").IsValid())
        self.assertFalse(frame.FindRegister("R32").IsValid())

        # As our kernel crashes in the prologue of assert, the RA register should be the same as the PC.
        self.assertEqual(
            frame.FindRegister("RA").GetValueAsAddress(),
            frame.FindRegister("PC").GetValueAsAddress(),
        )

    def test_cubin_sections_have_load_addresses(self):
        """Test that all executable text sections of all cubins have a load address."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        cpu = self.dbg.GetTargetAtIndex(0)
        gpu = self.dbg.GetTargetAtIndex(1)

        # We switch to async mode to wait for state changes in the GPU target while the CPU resumes.
        self.setAsync(True)
        listener = self.dbg.GetListener()
        cpu.process.Continue()
        lldbutil.expect_state_changes(self, listener, gpu.process, [lldb.eStateRunning, lldb.eStateStopped])

        self.assertEqual(gpu.process.state, lldb.eStateStopped)

        # Check all modules on the GPU target (all will be cubins)
        self.assertGreater(gpu.GetNumModules(), 0, "No modules found on GPU target")

        total_executable_sections = 0
        modules_without_executable_sections = []
        executable_sections_without_load_address = []
        executable_sections_with_load_address = 0

        # Gather all statistics for all cubins
        for module_idx in range(gpu.GetNumModules()):
            module = gpu.GetModuleAtIndex(module_idx)
            module_name = module.GetFileSpec().GetFilename()

            module_executable_sections = 0

            for section in module.section_iter():
                # Check if this is an executable text section
                section_name = section.GetName()
                section_permissions = section.GetPermissions()

                # Look for NVIDIA cubin text sections (e.g., .text.memcpyDtoD2D_unalignedSmallHeight64)
                is_text_section = section_name and section_name.startswith(".text")
                is_executable = section_permissions & lldb.ePermissionsExecutable

                if is_text_section and is_executable:
                    module_executable_sections += 1
                    total_executable_sections += 1

                    load_address = section.GetLoadAddress(gpu)
                    if load_address == lldb.LLDB_INVALID_ADDRESS or load_address == 0:
                        executable_sections_without_load_address.append(f"{module_name}:{section_name}")
                    else:
                        executable_sections_with_load_address += 1

            if module_executable_sections == 0:
                modules_without_executable_sections.append(module_name)

        # Assert that all executable text sections have valid load addresses
        self.assertEqual(
            len(executable_sections_without_load_address),
            0,
            f"Executable text sections without load addresses: {executable_sections_without_load_address}",
        )

        # Verify that we have at least some executable text sections overall
        self.assertGreater(total_executable_sections, 0, "No executable text sections found in any cubin modules")

        # Log information about modules without executable sections (if any)
        if self.TraceOn and modules_without_executable_sections:
            print(f"Modules without executable text sections: {modules_without_executable_sections}")

        # Verify that we have at least some executable text sections with non-zero load addresses
        self.assertGreater(
            executable_sections_with_load_address,
            0,
            "No executable text sections found with valid non-zero load addresses",
        )
