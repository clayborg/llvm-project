import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvidiagpu_testcase import NvidiaGpuTestCaseBase


class TestNVIDIAGPUCubins(NvidiaGpuTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_cubin_sections_have_load_addresses(self):
        """Test that all executable text sections of all cubins have a load address."""
        self.build()
        source = "assert.cu"
        cpu_bp_line: int = line_number(source, "// breakpoint1")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)

        self.assertEqual(self.dbg.GetNumTargets(), 2)

        # We switch to async mode to wait for state changes in the GPU target while the CPU resumes.
        self.continue_cpu_and_wait_for_gpu_to_stop()

        # Check all modules on the GPU target (all will be cubins)
        self.assertGreater(self.gpu_target.GetNumModules(), 0, "No modules found on GPU target")

        total_executable_sections = 0
        modules_without_executable_sections = []
        executable_sections_without_load_address = []
        executable_sections_with_load_address = 0

        # Gather all statistics for all cubins
        for module_idx in range(self.gpu_target.GetNumModules()):
            module = self.gpu_target.GetModuleAtIndex(module_idx)
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

                    load_address = section.GetLoadAddress(self.gpu_target)
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
