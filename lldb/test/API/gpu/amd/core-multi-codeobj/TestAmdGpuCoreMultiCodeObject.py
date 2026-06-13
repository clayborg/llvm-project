# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Multiple GPU code objects embedded in one binary (sharing a FileSpec but at
different object-file slices) must each load as a distinct module, with symbols,
and unwinding must not crash."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.amdgpu_core_testbase import AmdGpuCoreTestBase


class TestAmdGpuCoreMultiCodeObject(AmdGpuCoreTestBase):
    HIP_SOURCE = "kob.hip"
    GPU_BREAKPOINT_PATTERN = "// BETA BREAKPOINT"

    @skipIfRemote
    def test_multiple_code_objects_kept_distinct(self):
        gpu_target, gpu_process = self.load_core()

        # Both code objects' modules and symbols must survive, not just one.
        aout_modules = 0
        alpha_found = 0
        beta_found = 0
        for i in range(gpu_target.GetNumModules()):
            module = gpu_target.GetModuleAtIndex(i)
            if "a.out" not in (module.GetFileSpec().GetFilename() or ""):
                continue
            aout_modules += 1
            alpha_found += module.FindSymbols("kernel_alpha").GetSize()
            beta_found += module.FindSymbols("kernel_beta").GetSize()

        self.assertGreaterEqual(
            aout_modules, 2, "GPU code objects collapsed into one module"
        )
        self.assertGreater(alpha_found, 0, "kernel_alpha symbol missing")
        self.assertGreater(beta_found, 0, "kernel_beta symbol missing")

        # Unwinding the stopped wave must not crash and must be symbolized.
        frame = gpu_process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        self.assertIn("kernel_beta", frame.GetFunctionName() or "")
