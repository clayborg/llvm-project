# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Test AMD GPU core file placeholder module support.

When a GPU code object's binary cannot be found on disk or read from memory,
a placeholder module should be created so it still appears in 'image list'
and can later be re-hydrated (e.g. via a symbol server).

This test generates a GPU core via rocgdb, then deletes the binary so that
file-backed code objects become unfindable. The placeholder module support
should create placeholder modules for those missing code objects.
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.amdgpu_core_testbase import AmdGpuCoreTestBase


class TestAmdGpuCorePlaceholderModule(AmdGpuCoreTestBase):
    """Test placeholder modules for missing GPU code objects in core files"""

    HIP_SOURCE = "variables.hip"
    GPU_BREAKPOINT_PATTERN = "// GPU BREAKPOINT"

    @skipIfRemote
    def test_placeholder_modules_created_for_missing_files(self):
        """Test that placeholder modules are created when code object files
        cannot be found on disk."""
        # Delete the binary BEFORE loading the core so file-backed code
        # objects can't be found and must become placeholders.
        binary_path = self.getBuildArtifact("a.out")
        self.assertTrue(
            os.path.isfile(binary_path),
            f"Binary should exist at {binary_path}",
        )
        os.remove(binary_path)

        # Load the core — file-backed code objects should get placeholder
        # modules instead of being silently dropped.
        gpu_target, gpu_process = self.load_core()
        num_modules = gpu_target.GetNumModules()

        self.assertGreater(
            num_modules,
            0,
            "GPU target should have at least one module",
        )

        # Verify at least one module is a placeholder (no UUID) and at least
        # one is a real module (has UUID). Memory-backed code objects (e.g.
        # amd_memory_kernel) are read from core memory and produce real ELF
        # modules with UUIDs. File-backed code objects whose binary was
        # deleted should produce placeholders without UUIDs.
        modules_with_uuid = 0
        modules_without_uuid = 0
        for i in range(num_modules):
            module = gpu_target.GetModuleAtIndex(i)
            self.assertTrue(
                module.IsValid(), f"Module at index {i} should be valid"
            )
            filename = module.GetFileSpec().GetFilename()
            self.assertIsNotNone(
                filename, f"Module at index {i} should have a filename"
            )

            uuid = module.GetUUIDString()
            if uuid and len(uuid) > 0:
                modules_with_uuid += 1
            else:
                modules_without_uuid += 1

            # Every module (real or placeholder) should have at least one
            # section with a valid load address.
            num_sections = module.GetNumSections()
            self.assertGreater(
                num_sections,
                0,
                f"Module '{filename}' should have at least one section",
            )
            has_valid_load_addr = False
            for j in range(num_sections):
                section = module.GetSectionAtIndex(j)
                load_addr = section.GetLoadAddress(gpu_target)
                if load_addr != lldb.LLDB_INVALID_ADDRESS:
                    has_valid_load_addr = True
                    break
            self.assertTrue(
                has_valid_load_addr,
                f"Module '{filename}' should have a valid load address",
            )

        self.assertGreater(
            modules_with_uuid,
            0,
            "Should have at least one real module (with UUID) "
            "from memory-backed code objects",
        )
        self.assertGreater(
            modules_without_uuid,
            0,
            "Should have at least one placeholder module (without UUID) "
            "for missing code object files",
        )
