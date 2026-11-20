"""
Test how lldb loads the libraries that a GPU plug-in reports.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase


class TestMockGPUDynamicLoader(GpuTestCaseBase):
    def test_section_the_module_does_not_have(self):
        """With the dyld log on, lldb logs a section that the plug-in asks it
        to load and the library's module does not have, naming the section's
        path and the library, and goes on debugging."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        # The plug-in reports lib6.so at a path that does not exist, with a
        # section that no program has. Map the path to this program, so that
        # the GPU target makes a module for lib6.so and looks the section up.
        # With no target yet, this changes the global setting, which the GPU
        # target copies when lldb creates it. Teardown hooks run after the
        # targets are deleted, so the hook clears the global setting again.
        self.runCmd(f'settings set target.object-map /usr/lib/lib6.so "{exe}"')
        self.addTearDownHook(lambda: self.runCmd("settings clear target.object-map"))
        log_file = self.getBuildArtifact("dyld.log")
        self.runCmd(f'log enable -f "{log_file}" lldb dyld')
        self.addTearDownHook(lambda: self.runCmd("log disable lldb dyld"))

        # lldb loads the GPU target's libraries when the plug-in connects the
        # GPU target, before the process stops here.
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        gpu_target = self.gpu_target
        self.assertTrue(gpu_target, "the plug-in connected its GPU target")
        self.assertTrue(
            gpu_target.FindModule(lldb.SBFileSpec(exe)).IsValid(),
            "the GPU target has a module for lib6.so",
        )

        expected = (
            'Failed to find section "PT_LOAD[1]/.text/missing" '
            'in module "/usr/lib/lib6.so"'
        )
        with open(log_file) as log:
            lib6_lines = [line for line in log if "lib6.so" in line]
        # lldb starts each line with the name of the thread that logged it.
        self.assertTrue(
            any(line.endswith(" " + expected + "\n") for line in lib6_lines),
            "the dyld log does not name the section:\n" + "".join(lib6_lines),
        )
