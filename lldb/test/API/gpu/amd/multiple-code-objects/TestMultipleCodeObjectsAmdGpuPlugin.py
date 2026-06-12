"""
Tests for live AMDGPU processes with multiple code objects.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from amdgpu_testcase import *
from lldbsuite.test.lldbtest import *


class MultipleCodeObjectsAmdGpuTestCase(AmdGpuTestCaseBase):
    def run_to_gpu_trap(self):
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be valid")
        self.assertTrue(self.gpu_process.IsValid(), "GPU process should be valid")

        self.setAsync(True)
        listener = self.dbg.GetListener()

        self.select_gpu()
        error = self.gpu_process.Continue()
        self.assertSuccess(error, "continue GPU process")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        self.select_cpu()
        error = self.cpu_process.Continue()
        self.assertSuccess(error, "continue CPU process")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateStopped]
        )

    def get_frame_names(self, frame):
        names = []

        function_name = frame.GetFunctionName()
        if function_name:
            names.append(function_name)

        display_function_name = frame.GetDisplayFunctionName()
        if display_function_name:
            names.append(display_function_name)

        function = frame.GetFunction()
        if function.IsValid():
            name = function.GetName()
            if name:
                names.append(name)

        symbol = frame.GetSymbol()
        if symbol.IsValid():
            name = symbol.GetName()
            if name:
                names.append(name)

        return names

    def test_multiple_code_object_modules_are_preserved(self):
        self.run_to_gpu_trap()
        self.select_gpu()
        gpu_target = self.dbg.GetSelectedTarget()
        self.assertTrue(gpu_target.IsValid(), "selected GPU target should be valid")

        modules = []
        file_backed_code_objects = []
        memory_backed_code_objects = []
        module_count = gpu_target.GetNumModules()
        for idx in range(module_count):
            module = gpu_target.GetModuleAtIndex(idx)
            self.assertTrue(module.IsValid(), "module %d should be valid" % idx)

            filename = module.GetFileSpec().GetFilename()
            description = str(module)
            modules.append(description)
            if filename == "a.out":
                file_backed_code_objects.append(module)
            if "amd_memory_kernel[" in description:
                memory_backed_code_objects.append(module)

        self.assertGreaterEqual(
            module_count,
            2,
            "expected multiple GPU code object modules; got modules: "
            + repr(modules),
        )
        self.assertTrue(
            file_backed_code_objects,
            "expected a file-backed GPU code object module; got modules: "
            + repr(modules),
        )
        self.assertTrue(
            memory_backed_code_objects,
            "expected a memory-backed GPU code object module; got modules: "
            + repr(modules),
        )

        thread = self.gpu_process.GetSelectedThread()
        self.assertTrue(thread.IsValid(), "selected GPU thread should be valid")
        self.assertGreater(thread.GetNumFrames(), 0, "GPU thread should have frames")

        frame_names = []
        found_second_kernel = False
        for idx in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(idx)
            self.assertTrue(frame.IsValid(), "frame %d should be valid" % idx)

            names = self.get_frame_names(frame)
            frame_names.append((idx, names))
            if any("second_code_object_kernel" in name for name in names):
                found_second_kernel = True

        self.assertTrue(
            found_second_kernel,
            "expected GPU frames to include second_code_object_kernel; got frame names: "
            + repr(frame_names),
        )
