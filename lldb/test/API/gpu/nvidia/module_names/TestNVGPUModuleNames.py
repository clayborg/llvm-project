import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUModuleNames(NVGPUTestCaseBase):
    SESSION_NAME = "NVIDIA GPU Session"

    def run_to_breakpoint_after_launches(self):
        """Stop the CPU once every kernel has run, so both cubins are loaded."""
        self.build()
        source = "main.cu"
        line = line_number(source, "// breakpoint after launches")
        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), line)
        self.assertEqual(self.dbg.GetNumTargets(), 2)

    def test_gpu_target_name(self):
        """Test that the GPU target is described by its session name."""
        self.killCPUOnTeardown()
        self.run_to_breakpoint_after_launches()

        self.assertEqual(self.gpu_target.GetTargetSessionName(), self.SESSION_NAME)
        description = lldb.SBStream()
        self.gpu_target.GetDescription(description, lldb.eDescriptionLevelBrief)
        self.assertEqual(description.GetData(), self.SESSION_NAME)
        self.expect("target list", substrs=[f"{self.SESSION_NAME} ( arch=nvptx64"])

    def test_cubin_module_names(self):
        """Test that a cubin is named after its source file, or else its kernel."""
        self.killCPUOnTeardown()
        self.run_to_breakpoint_after_launches()

        names = [
            module.GetFileSpec().GetFilename()
            for module in self.gpu_target.module_iter()
        ]
        # main.cu is built with -G, so its cubin has a compile unit.
        self.assertIn("main.cubin", names)
        # nodebug.cu is built without -G. The name leaves out the anonymous
        # namespace its kernel is in.
        self.assertIn("physics::integrate.cubin", names)
        self.assertEqual(len(names), len(set(names)), f"duplicate names in {names}")

    def test_rerun_creates_new_cubin_modules(self):
        """Test that a second run gets its own cubin modules, not the first run's."""
        self.killCPUOnTeardown()
        self.run_to_breakpoint_after_launches()
        cubin = lldb.SBFileSpec("main.cubin", False)
        first_gpu_target = self.gpu_target
        first_module = first_gpu_target.FindModule(cubin)
        self.assertTrue(first_module.IsValid())

        self.assertSuccess(self.cpu_process.Kill())
        # Each run adds a GPU target and leaves the previous one behind, and
        # gpu_target returns the first GPU target it finds.
        self.dbg.DeleteTarget(first_gpu_target)
        process = self.cpu_target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertState(process.GetState(), lldb.eStateStopped)

        second_module = self.gpu_target.FindModule(cubin)
        self.assertTrue(second_module.IsValid())
        self.assertNotEqual(first_module, second_module)
