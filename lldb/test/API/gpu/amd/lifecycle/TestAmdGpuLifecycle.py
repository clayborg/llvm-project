"""
Tests for how the GPU target follows the native process's lifecycle.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SOURCE = "lifecycle.hip"
GPU_BREAKPOINT = "// GPU BREAKPOINT"


class AmdGpuLifecycleTestCase(AmdGpuTestCaseBase):
    def stop_at_gpu_breakpoint(self) -> lldb.SBListener:
        """Run until the waves stop at the GPU breakpoint, then stop the CPU."""
        self.build()
        threads = self.run_to_gpu_breakpoint(SOURCE, GPU_BREAKPOINT)
        self.assertTrue(threads, "GPU should be stopped at the breakpoint")
        listener = self.dbg.GetListener()
        self.stop_cpu_if_running(listener)
        return listener

    def continue_cpu_to_exit(self, listener: lldb.SBListener):
        """Continue the CPU process and check that the kernel ran to its end:
        the program exits with 0 only if the kernel wrote the right values."""
        cpu_process = self.cpu_process
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, cpu_process, [lldb.eStateRunning, lldb.eStateExited]
        )
        self.assertEqual(cpu_process.GetExitStatus(), 0)

    def test_gpu_detach_resumes_waves(self):
        """Test that detaching from the GPU resumes the stopped waves and
        leaves the CPU process under the debugger."""
        listener = self.stop_at_gpu_breakpoint()

        gpu_process = self.gpu_process
        self.assertSuccess(gpu_process.Detach())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(self.cpu_process.GetState(), lldb.eStateStopped)

        self.continue_cpu_to_exit(listener)

    def test_gpu_kill_detaches(self):
        """Test that killing the GPU process detaches from it instead, as the
        AMD GPU plug-in asks, since the CPU process owns the program. The detach
        resumes the stopped waves."""
        listener = self.stop_at_gpu_breakpoint()

        gpu_process = self.gpu_process
        self.assertSuccess(gpu_process.Kill())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(self.cpu_process.GetState(), lldb.eStateStopped)

        self.continue_cpu_to_exit(listener)

    def test_cpu_kill_detaches_gpu(self):
        """Test that killing the CPU process detaches from the GPU first."""
        self.stop_at_gpu_breakpoint()

        gpu_process = self.gpu_process
        cpu_process = self.cpu_process
        self.assertSuccess(cpu_process.Kill())
        self.assertState(gpu_process.GetState(), lldb.eStateDetached)
        self.assertState(cpu_process.GetState(), lldb.eStateExited)

    def test_rerun_hits_gpu_breakpoint_again(self):
        """Test that running the program again reuses the GPU target, so the
        breakpoint set on it in the first run is hit again, as in ROCgdb."""
        self.stop_at_gpu_breakpoint()
        gpu_target = self.gpu_target
        gpu_bkpt = gpu_target.GetBreakpointAtIndex(0)
        self.assertSuccess(self.cpu_process.Kill())

        # As in the first run, the launch stops once the GPU target is
        # connected.
        self.setAsync(False)
        cpu_target = self.cpu_target
        launch_info = cpu_target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        cpu_target.Launch(launch_info, error)
        self.assertSuccess(error)
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        self.assertTrue(self.gpu_target == gpu_target)

        self.select_gpu()
        threads = self.continue_to_gpu_breakpoint(gpu_bkpt.GetID())
        self.assertTrue(threads, "GPU should stop at the breakpoint again")

    def test_attach_stops_waves_and_detach_resumes_them(self):
        """Test that attaching to a program whose kernel is running creates
        the GPU target with the waves stopped, as in ROCgdb. Then release the
        kernel and detach, which detaches from the GPU first, after which the
        program finishes on its own."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        ready_file = self.getBuildArtifact("ready.txt")
        result_file = self.getBuildArtifact("result.txt")
        popen = self.spawnSubprocess(exe, ["spin", ready_file, result_file])
        lldbutil.wait_for_file_on_target(self, ready_file)

        target = self.dbg.CreateTarget(exe)
        error = lldb.SBError()
        target.AttachToProcessWithID(self.dbg.GetListener(), popen.pid, error)
        self.assertSuccess(error)
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        frame = self.gpu_process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        self.assertIn("spin_kernel", frame.GetFunctionName())

        # Release the kernel, as ROCgdb's "set var g_release = 1" would.
        gpu_target = self.gpu_target
        release = gpu_target.FindSymbols("g_release")[0].GetSymbol()
        address = release.GetStartAddress().GetLoadAddress(gpu_target)
        self.gpu_process.WriteMemory(address, (1).to_bytes(4, "little"), error)
        self.assertSuccess(error)

        # Detaching from the process detaches from the GPU first, which resumes
        # the waves.
        self.assertSuccess(self.cpu_process.Detach())
        result = lldbutil.wait_for_file_on_target(self, result_file)
        self.assertEqual(result.strip(), "released")

    def test_gpu_target_delete_keeps_cpu_process(self):
        """Test that deleting the GPU target detaches from the GPU only."""
        listener = self.stop_at_gpu_breakpoint()

        self.assertTrue(self.dbg.DeleteTarget(self.gpu_target))
        self.assertEqual(self.dbg.GetNumTargets(), 1)

        self.continue_cpu_to_exit(listener)
