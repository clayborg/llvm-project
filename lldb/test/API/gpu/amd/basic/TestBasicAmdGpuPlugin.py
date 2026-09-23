"""
Basic tests for the AMDGPU plugin.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SHADOW_THREAD_NAME = "AMD Native Shadow Thread"


class BasicAmdGpuTestCase(AmdGpuTestCaseBase):
    def test_gpu_target_created_on_demand(self):
        """Test that we create the gpu target automatically."""
        self.build()

        # There should be no targets before we run the program.
        self.assertEqual(self.dbg.GetNumTargets(), 0, "There are no targets")

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch. The GPU plugin
        # stops the CPU when GPU modules are loaded (auto_resume_native=false).
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")

        # Make sure the GPU target has the default thread.
        gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        self.assertEqual(
            gpu_thread.GetName(), SHADOW_THREAD_NAME, "GPU thread has the right name"
        )

        # The target should have the triple set correctly.
        self.assertIn("amdgcn-amd-amdhsa", self.gpu_target.GetTriple())

    def test_gpu_breakpoint_hit(self):
        """Test that we can hit a breakpoint on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

    def test_gpu_basic_step_over(self):
        """Test that a GPU thread can step over a source line."""
        self.build()

        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")
        self.step_over_gpu_thread(
            gpu_threads[0], line_number(source, "// GPU STEP OVER")
        )

    def test_gpu_all_thread_resume_preserves_stepped_lane(self):
        """Step a non-first lane, then perform a default all-thread continue to
        a later breakpoint. Because that continue does not identify a new
        preferred lane, the server must preserve and reselect the lane from
        the preceding explicit step rather than the wave's first lane."""
        self.build()

        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertGreater(len(gpu_threads), 1)

        # Step a non-first lane so losing its preference to the wave's first
        # active lane is observable.
        stepped_thread = sorted(
            gpu_threads, key=lambda thread: thread.GetLaneID()
        )[1]
        stepped_tid = stepped_thread.GetThreadID()

        listener = self.prepare_for_gpu_step()
        self.dbg.SetSelectedTarget(self.gpu_target)
        self.assertTrue(self.gpu_process.SetSelectedThread(stepped_thread))

        error = lldb.SBError()
        stepped_thread.StepInstruction(False, error)
        self.assertSuccess(error, "step one GPU instruction")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )
        self.assertEqual(
            lldb.eStopReasonPlanComplete, stepped_thread.GetStopReason()
        )
        self.assertEqual(
            stepped_tid, self.gpu_process.GetSelectedThread().GetThreadID()
        )

        next_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// GPU BREAKPOINT AFTER"
        )
        # This default all-thread action is indistinguishable to lldb-server
        # from an all-thread resume performed internally by the step plan.
        stopped_threads = self.continue_gpu_to_breakpoint(next_breakpoint_id)
        self.assertTrue(stopped_threads)
        self.assertTrue(stepped_thread.IsActive())
        self.assertEqual(
            stepped_tid, self.gpu_process.GetSelectedThread().GetThreadID()
        )

    def test_gpu_step_over_divergent_lane(self):
        """Test that stepping follows the selected lane through divergence."""
        self.build()

        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// DIVERGENT STEP")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        lane_zero_thread = None
        for thread in gpu_threads:
            idx = thread.GetFrameAtIndex(0).FindVariable("idx").GetValueAsUnsigned()
            if idx == 0:
                lane_zero_thread = thread
                break
        self.assertIsNotNone(lane_zero_thread)
        self.assertTrue(lane_zero_thread.IsActive())

        # The wave executes the nonzero branch first, where lane 0 is inactive.
        # Stepping lane 0 must continue until its own branch is active.
        self.step_over_gpu_thread(
            lane_zero_thread, line_number(source, "// LANE ZERO BRANCH")
        )
        self.assertTrue(lane_zero_thread.IsActive())

    def test_gpu_thread_specific_breakpoint_ignores_inactive_lane(self):
        """Test that an inactive lane does not claim its breakpoint."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertSuccess(error, "launch process")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        divergent_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// DIVERGENT STEP"
        )
        nonzero_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// NONZERO LANE BRANCH"
        )
        gpu_threads = self.continue_to_gpu_breakpoint(divergent_breakpoint_id)
        self.assertTrue(gpu_threads)

        lane_zero_thread = next(
            thread
            for thread in gpu_threads
            if thread.GetFrameAtIndex(0)
            .FindVariable("idx")
            .GetValueAsUnsigned()
            == 0
        )
        self.assertTrue(lane_zero_thread.IsActive())

        nonzero_breakpoint = self.gpu_target.FindBreakpointByID(
            nonzero_breakpoint_id
        )
        self.assertTrue(nonzero_breakpoint.IsValid())
        self.assertGreater(nonzero_breakpoint.GetNumLocations(), 0)
        nonzero_breakpoint.SetThreadID(lane_zero_thread.GetThreadID())
        self.assertEqual(
            lane_zero_thread.GetThreadID(), nonzero_breakpoint.GetThreadID()
        )

        # Lane 0 is inactive while the wave executes the nonzero branch, so
        # its thread-specific breakpoint must not interrupt lane 0's step.
        self.step_over_gpu_thread(
            lane_zero_thread, line_number(source, "// LANE ZERO BRANCH")
        )

    def _step_lane_zero_to_nonzero_branch_breakpoint(self):
        """Step lane 0 until the active nonzero lanes hit a breakpoint."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertSuccess(error, "launch process")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        divergent_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// DIVERGENT STEP"
        )
        nonzero_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// NONZERO LANE BRANCH"
        )
        sentinel_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// GPU BREAKPOINT"
        )
        gpu_threads = self.continue_to_gpu_breakpoint(divergent_breakpoint_id)
        self.assertTrue(gpu_threads)

        lane_zero_thread = next(
            thread
            for thread in gpu_threads
            if thread.GetFrameAtIndex(0)
            .FindVariable("idx")
            .GetValueAsUnsigned()
            == 0
        )
        self.assertTrue(lane_zero_thread.IsActive())

        self.setAsync(True)
        listener = self.dbg.GetListener()
        self.stop_cpu_if_running(listener)
        self.setAsync(False)

        self.dbg.SetSelectedTarget(self.gpu_target)
        self.assertTrue(
            self.gpu_process.SetSelectedThread(lane_zero_thread),
            "select lane 0",
        )

        error = lldb.SBError()
        lane_zero_thread.StepOver(lldb.eOnlyDuringStepping, error)
        self.assertSuccess(error, "step lane 0 to the divergent breakpoint")
        self.assertFalse(lane_zero_thread.IsActive())
        selected_thread = self.gpu_process.GetSelectedThread()
        self.assertTrue(selected_thread.IsActive())
        self.expect(
            "process plugin packet send qC",
            substrs=[f"response: QC{selected_thread.GetThreadID():x}"],
        )
        breakpoint_threads = lldbutil.get_threads_stopped_at_breakpoint_id(
            self.gpu_process, nonzero_breakpoint_id
        )
        self.assertTrue(
            breakpoint_threads,
            "the nonzero branch breakpoint should interrupt the step",
        )
        return (
            source,
            lane_zero_thread,
            breakpoint_threads,
            sentinel_breakpoint_id,
        )

    def _step_selected_lane_from_nonzero_branch_breakpoint(self):
        active_thread = self.gpu_process.GetSelectedThread()
        self.assertTrue(active_thread.IsActive())
        self.assertEqual(lldb.eStopReasonBreakpoint, active_thread.GetStopReason())

        before_pc = active_thread.GetFrameAtIndex(0).GetPC()
        listener = self.prepare_for_gpu_step()
        self.dbg.SetSelectedTarget(self.gpu_target)
        error = lldb.SBError()
        active_thread.StepInstruction(False, error)
        self.assertSuccess(error, "step the active nonzero lane")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )
        self.assertEqual(
            active_thread.GetThreadID(),
            self.gpu_process.GetSelectedThread().GetThreadID(),
        )
        self.assertEqual(lldb.eStopReasonPlanComplete, active_thread.GetStopReason())
        self.assertNotEqual(before_pc, active_thread.GetFrameAtIndex(0).GetPC())

    def test_gpu_continue_from_breakpoint_while_stepped_lane_inactive(self):
        """Lane 0 starts a step, becomes inactive while the wave executes the
        nonzero branch, and is interrupted by that branch's breakpoint.
        Continuing from the breakpoint without deleting it must step the wave
        past the breakpoint and finish lane 0's original step."""
        source, lane_zero_thread, _, _ = (
            self._step_lane_zero_to_nonzero_branch_breakpoint()
        )

        self.setAsync(True)
        listener = self.dbg.GetListener()
        error = self.gpu_process.Continue()
        self.assertSuccess(error, "continue the interrupted GPU step")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )
        self.setAsync(False)

        self.assertState(self.gpu_process.GetState(), lldb.eStateStopped)
        self.assertEqual(
            lldb.eStopReasonPlanComplete, lane_zero_thread.GetStopReason()
        )
        self.assertEqual(
            lane_zero_thread.GetThreadID(),
            self.gpu_process.GetSelectedThread().GetThreadID(),
        )
        self.assertTrue(lane_zero_thread.IsActive())
        self.assertEqual(
            line_number(source, "// LANE ZERO BRANCH"),
            lane_zero_thread.GetFrameAtIndex(0).GetLineEntry().GetLine(),
        )

    def test_gpu_step_from_breakpoint_while_other_lane_step_pending(self):
        """Lane 0 starts a step and is inactive when the nonzero branch hits a
        breakpoint. An instruction step on the selected active nonzero lane
        must take priority and report completion on that lane rather than
        allowing lane 0's older step plan to control the stop."""
        _, lane_zero_thread, _, _ = self._step_lane_zero_to_nonzero_branch_breakpoint()
        self._step_selected_lane_from_nonzero_branch_breakpoint()
        self.assertFalse(lane_zero_thread.IsActive())

    def test_gpu_source_step_resumes_after_other_lane_step_completes(self):
        """Lane 0 starts a source step and becomes inactive when the nonzero
        branch hits a breakpoint. Source-stepping the selected nonzero lane
        switches execution back to lane 0, whose older plan completes first.
        Continuing must then finish the nonzero lane's pending source step
        before reaching a later safety breakpoint."""
        source, lane_zero_thread, _, sentinel_breakpoint_id = (
            self._step_lane_zero_to_nonzero_branch_breakpoint()
        )
        self.gpu_target.FindBreakpointByID(sentinel_breakpoint_id).SetEnabled(False)
        later_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// GPU BREAKPOINT AFTER"
        )

        nonzero_thread = self.gpu_process.GetSelectedThread()
        self.assertTrue(nonzero_thread.IsActive())
        self.assertEqual(lldb.eStopReasonBreakpoint, nonzero_thread.GetStopReason())

        listener = self.prepare_for_gpu_step()
        self.dbg.SetSelectedTarget(self.gpu_target)
        error = lldb.SBError()
        nonzero_thread.StepOver(lldb.eOnlyDuringStepping, error)
        self.assertSuccess(error, "source-step the selected nonzero lane")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )

        self.assertEqual(
            lane_zero_thread.GetThreadID(),
            self.gpu_process.GetSelectedThread().GetThreadID(),
        )
        self.assertEqual(lldb.eStopReasonPlanComplete, lane_zero_thread.GetStopReason())

        listener = self.dbg.GetListener()
        error = self.gpu_process.Continue()
        self.assertSuccess(error, "continue the pending nonzero-lane source step")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )
        self.setAsync(False)

        self.assertFalse(
            lldbutil.get_threads_stopped_at_breakpoint_id(
                self.gpu_process, later_breakpoint_id
            ),
            "the nonzero lane's source step should complete before the safety breakpoint",
        )
        self.assertEqual(
            nonzero_thread.GetThreadID(),
            self.gpu_process.GetSelectedThread().GetThreadID(),
        )
        self.assertEqual(lldb.eStopReasonPlanComplete, nonzero_thread.GetStopReason())

    def test_gpu_continue_pending_step_after_stepping_breakpoint_lane(self):
        """Lane 0's source step is interrupted while inactive by a breakpoint
        in the nonzero branch. After instruction-stepping the selected active
        lane past that breakpoint, continuing must complete lane 0's pending
        step before reaching the later sentinel breakpoint."""
        _, lane_zero_thread, _, sentinel_breakpoint_id = (
            self._step_lane_zero_to_nonzero_branch_breakpoint()
        )
        self._step_selected_lane_from_nonzero_branch_breakpoint()
        self.assertFalse(lane_zero_thread.IsActive())

        self.setAsync(True)
        listener = self.dbg.GetListener()
        error = self.gpu_process.Continue()
        self.assertSuccess(error, "continue lane 0's pending step")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )
        self.setAsync(False)

        self.assertFalse(
            lldbutil.get_threads_stopped_at_breakpoint_id(
                self.gpu_process, sentinel_breakpoint_id
            ),
            "lane 0's step should complete before the sentinel breakpoint",
        )
        self.assertEqual(lldb.eStopReasonPlanComplete, lane_zero_thread.GetStopReason())
        self.assertEqual(
            lane_zero_thread.GetThreadID(),
            self.gpu_process.GetSelectedThread().GetThreadID(),
        )

    def test_gpu_resume_to_next_breakpoint(self):
        """Test that resuming GPU threads can hit a later breakpoint."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertFalse(error.Fail(), "Process launch failed: %s" % error.GetCString())
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        first_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT")
        next_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT AFTER")

        gpu_threads = self.continue_to_gpu_breakpoint(first_bkpt_id)
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        gpu_threads = self.continue_gpu_to_breakpoint(next_bkpt_id)
        self.assertNotEqual(None, gpu_threads, "GPU should hit next breakpoint")

    def test_gpu_continue_selects_active_lane(self):
        """Continue all lanes from a divergence point until the nonzero branch
        hits a breakpoint while lane 0 is inactive. With no explicit lane in
        the resume action, the server must select an active lane from the
        stopped wave rather than retaining inactive lane 0."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertSuccess(error, "launch process")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        divergent_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// DIVERGENT STEP"
        )
        nonzero_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// NONZERO LANE BRANCH"
        )
        self.continue_to_gpu_breakpoint(divergent_breakpoint_id)

        gpu_threads = self.continue_gpu_to_breakpoint(nonzero_breakpoint_id)
        self.assertTrue(gpu_threads, "nonzero lanes should hit the breakpoint")
        self.assertTrue(self.gpu_process.GetSelectedThread().IsActive())

    def test_gpu_thread_continue_selects_active_lane(self):
        """Explicitly continue lane 0 from a divergence point until the nonzero
        branch hits a breakpoint while lane 0 is inactive. The stopped wave
        must fall back to an active lane instead of selecting the explicitly
        resumed but inactive lane 0."""
        self.build()

        source = "hello_world.hip"
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, "Could not create a valid process")
        self.assertSuccess(error, "launch process")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        divergent_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// DIVERGENT STEP"
        )
        nonzero_breakpoint_id = self.set_gpu_source_breakpoint(
            source, "// NONZERO LANE BRANCH"
        )
        gpu_threads = self.continue_to_gpu_breakpoint(divergent_breakpoint_id)

        lane_zero_thread = next(
            thread
            for thread in gpu_threads
            if thread.GetFrameAtIndex(0)
            .FindVariable("idx")
            .GetValueAsUnsigned()
            == 0
        )
        self.assertTrue(lane_zero_thread.IsActive())

        listener = self.prepare_for_gpu_step()
        self.dbg.SetSelectedTarget(self.gpu_target)
        self.runCmd(f"thread continue {lane_zero_thread.GetIndexID()}")
        lldbutil.expect_state_changes(
            self,
            listener,
            self.gpu_process,
            [lldb.eStateRunning, lldb.eStateStopped],
        )

        gpu_threads = lldbutil.get_threads_stopped_at_breakpoint_id(
            self.gpu_process, nonzero_breakpoint_id
        )
        self.assertTrue(gpu_threads, "nonzero lanes should hit the breakpoint")
        self.assertTrue(self.gpu_process.GetSelectedThread().IsActive())

    def test_gpu_breakpoint_delete(self):
        """Test that deleting a GPU breakpoint removes the trap opcode."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should be created")

        source = "hello_world.hip"
        gpu_bkpt_id = self.set_gpu_source_breakpoint(source, "// GPU BREAKPOINT")
        self.assertTrue(
            self.gpu_target.BreakpointDelete(gpu_bkpt_id),
            "GPU breakpoint should be deleted",
        )

        self.setAsync(True)
        listener = self.dbg.GetListener()

        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateExited]
        )
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateExited]
        )

    def test_num_threads(self):
        """Test that we get the expected number of threads."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        # We launch one thread for each character in the output string.
        gpu_threads = self.gpu_process.threads
        num_expected_threads = len("Hello, world!")
        self.assertEqual(len(gpu_threads), num_expected_threads)

        # The shadow thread should not be listed once we have real threads
        for thread in gpu_threads:
            self.assertNotEqual(SHADOW_THREAD_NAME, thread.GetName())

        # All threads should be stopped at the breakpoint.
        self.assertEqual(len(gpu_threads_at_bp), num_expected_threads)

    def test_num_threads_divergent_breakpoint(self):
        """Test that we get the expected number of threads in a divergent breakpoint."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads_at_bp = self.run_to_gpu_breakpoint(
            source, "// DIVERGENT BREAKPOINT"
        )
        self.assertNotEqual(
            None, gpu_threads_at_bp, "GPU should be stopped at breakpoint"
        )

        # We launch one thread for each character in the output string.
        # So all threads should be present in the process.
        gpu_threads = self.gpu_process.threads
        total_num_threads = len("Hello, world!")
        self.assertEqual(len(gpu_threads), total_num_threads)

        # Since all the threads are in the same wave, they all share the same pc
        # and should be stopped at the same breakpoint. At some point, we need to
        # represent active/inactive threads in lldb, but that support does not yet
        # exist.
        self.assertEqual(len(gpu_threads_at_bp), total_num_threads)

    def test_no_unexpected_stop(self):
        """Test that when no user breakpoints are set, the process stops at the
        GPU internal breakpoint for GPU target creation, and exits normally
        when continued."""
        self.build()

        target = self.createTestTarget()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Process is valid")

        # The GPU target should be created after launch.
        self.assertTrue(self.gpu_target.IsValid(), "GPU target should exist")

        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Continue GPU (non-blocking in async mode)
        self.select_gpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateRunning]
        )

        # Continue CPU (non-blocking in async mode)
        self.select_cpu()
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateRunning]
        )

        # Now wait for both to exit
        lldbutil.expect_state_changes(
            self, listener, self.gpu_process, [lldb.eStateExited]
        )
        lldbutil.expect_state_changes(
            self, listener, self.cpu_process, [lldb.eStateExited]
        )

    def test_image_list(self):
        """Test that we can load modules on the gpu target."""
        self.build()

        # GPU breakpoint should get hit by at least one thread.
        source = "hello_world.hip"
        gpu_threads = self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        # There should two modules loaded for the gpu.
        # There should be one module loaded from the executable (the kernel) and one
        # loaded from memory (driver/debugger lib code).
        # File-backed modules keep their original file path (e.g. /path/to/a.out).
        # Memory-backed modules are named: amd_memory_kernel[start, end)
        gpu_modules = self.gpu_target.modules
        self.assertEqual(2, len(gpu_modules), "GPU should have two modules")

        # Check that one module contains "a.out" (file-backed, keeps original path)
        # and one starts with "amd_memory_kernel[" (memory-backed).
        module_names = [str(module.file) for module in gpu_modules]
        has_file_module = any("a.out" in name for name in module_names)
        has_memory_module = any("amd_memory_kernel[" in name for name in module_names)
        self.assertTrue(has_file_module,
                        f"Expected a file-backed module with 'a.out' in path, got: {module_names}")
        self.assertTrue(has_memory_module,
                        f"Expected a memory-backed module with 'amd_memory_kernel[' prefix, got: {module_names}")

        # Verify the "image list" command output shows the [offset-end) bracket
        # for the file-backed embedded GPU module (no space before the bracket).
        # Select GPU target so the command runs against it.
        self.dbg.SetSelectedTarget(self.gpu_target)
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("image list", result)
        output = result.GetOutput()
        self.assertTrue(result.Succeeded(), f"image list failed: {result.GetError()}")
        # File-backed module should show path[0x...-0x...) with no space before bracket.
        import re
        has_bracket_format = re.search(r'a\.out\[0x[0-9a-f]+-0x[0-9a-f]+\)', output)
        self.assertTrue(has_bracket_format,
                        f"Expected 'a.out[offset-end)' format in image list output, got:\n{output}")
        # Memory-backed module should show amd_memory_kernel[start, end)
        self.assertIn("amd_memory_kernel[", output,
                      f"Expected 'amd_memory_kernel[' in image list output, got:\n{output}")

    def test_wave_and_group_ids(self):
        """Test that a kernel smaller than a wave maps onto a single SIMD group."""
        self.build()

        source = "hello_world.hip"
        self.run_to_gpu_breakpoint(source, "// GPU BREAKPOINT")

        # A single block smaller than a wave lands entirely in one wave, so
        # every thread shares one SIMD id. Lane ids happen to be unique across
        # the process here only because of that; TestMultiWaveAmdGpuPlugin
        # covers a kernel spread over several waves, where they are not.
        gpu_threads = self.gpu_process.threads
        simd_ids = {thread.GetSIMD() for thread in gpu_threads}
        self.assertEqual(
            len(simd_ids), 1, f"all lanes should share one wave, got {simd_ids}"
        )
        self.assertNotIn(
            lldb.LLDB_INVALID_SIMD_ID, simd_ids, "GPU threads should have a SIMD id"
        )

        # A CPU thread is neither a lane nor part of a SIMD group.
        self.stop_cpu_if_running(self.dbg.GetListener())
        cpu_thread = self.cpu_process.GetThreadAtIndex(0)
        self.assertTrue(cpu_thread.IsValid(), "CPU thread should be valid")
        self.assertEqual(cpu_thread.GetLaneID(), lldb.LLDB_INVALID_LANE_ID)
        self.assertEqual(cpu_thread.GetSIMD(), lldb.LLDB_INVALID_SIMD_ID)
