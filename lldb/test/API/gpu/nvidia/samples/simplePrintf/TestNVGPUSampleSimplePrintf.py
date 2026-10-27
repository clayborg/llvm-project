"""
Tests on the cuda-samples program simplePrintf, copied in simplePrintf.cu. Its
kernel prints one line from each of its 32 threads. The program checks nothing
itself and always exits 0, so the tests check what it prints.
"""

import re
import time
from typing import List, Optional

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

SOURCE = "simplePrintf.cu"

# The line each thread prints: its linear block index (0-3), its linear thread
# index (0-7), and the kernel's argument.
PRINTED = re.compile(r"^\[(\d), (\d)\]:\t\tValue is:(\d+)\r?$", re.MULTILINE)


class TestNVGPUSampleSimplePrintf(NVGPUTestCaseBase):
    # A 2x2 grid of 2x2x2 blocks, as (blockIdx, threadIdx) of each thread.
    THREADS = {
        ((bx, by, 0), (tx, ty, tz))
        for bx in range(2)
        for by in range(2)
        for tx in range(2)
        for ty in range(2)
        for tz in range(2)
    }

    def make_target(self) -> lldb.SBTarget:
        """A target for the program, for a test that sets breakpoints before
        the run."""
        return lldbutil.run_to_breakpoint_make_target(
            self, self.build_shared("simplePrintf"), in_cwd=False
        )

    def run_to_launch(self, target: Optional[lldb.SBTarget] = None) -> None:
        """Run the program to its `// cpu breakpoint before launch` line.

        CUDA is initialized there and no kernel has run yet, so the GPU target
        exists and a breakpoint set on it now is in place for the launch.
        """
        if target is None:
            target = self.make_target()
        self.run_to_cpu_line(SOURCE, "// cpu breakpoint before launch", target=target)
        self.assertEqual(
            self.dbg.GetNumTargets(), 2, "CUDA is initialized, so there is a GPU target"
        )

    def check_output(self, stdout: str) -> None:
        """Each of the 32 threads printed its line exactly once."""
        expected = [(str(b), str(t), "10") for b in range(4) for t in range(8)]
        self.assertEqual(
            sorted(PRINTED.findall(stdout)), expected, f"stdout:\n{stdout}"
        )

    def stop_every_thread_at(
        self, breakpoint: lldb.SBBreakpoint, stopped: List[lldb.SBThread]
    ) -> None:
        """`stopped` are the threads at `breakpoint` in the current stop.
        Resume the GPU until every thread of the launch has stopped there,
        since they do not always all stop at once."""
        reached = {self.thread_coords(t) for t in stopped}
        for _ in range(len(self.THREADS)):
            if reached >= self.THREADS:
                break
            stopped = self.continue_gpu_until_breakpoint(breakpoint)
            reached |= {self.thread_coords(t) for t in stopped}
        self.assertEqual(reached, self.THREADS)

    def check_stop(self, breakpoint: lldb.SBBreakpoint) -> int:
        """The GPU is stopped at `breakpoint` in simplePrintfKernel. Check one
        stopped thread's frame, argument and CUDA built-ins, then that every
        thread of the launch reaches the breakpoint. Return the line of the
        first stop."""
        stopped = self.assert_gpu_stop_at_breakpoint(breakpoint)
        thread = stopped[0]
        frame = thread.frame[0]
        self.assertEqual(frame.GetFunctionName(), "simplePrintfKernel(int)")
        self.assertEqual(frame.GetLineEntry().GetFileSpec().GetFilename(), SOURCE)
        self.assertEqual(thread.GetNumFrames(), 1, "the kernel is the only frame")
        line = frame.GetLineEntry().GetLine()

        val = frame.FindVariable("val")
        self.assertSuccess(val.GetError())
        self.assertEqual(val.GetValueAsSigned(), 10)

        self.assertEqual(self.read_vec3_register(frame, "gridDim"), [2, 2, 1])
        self.assertEqual(self.read_vec3_register(frame, "blockDim"), [2, 2, 2])
        block = self.read_vec3_register(frame, "blockIdx")
        index = self.read_vec3_register(frame, "threadIdx")
        # The coordinates lldb-server reports and the thread's name agree with
        # the thread's registers.
        self.assertEqual(self.thread_coords(thread), (tuple(block), tuple(index)))
        self.assertEqual(
            thread.GetName(),
            f"blockIdx(x={block[0]} y={block[1]} z={block[2]}) "
            f"threadIdx(x={index[0]} y={index[1]} z={index[2]})",
        )

        self.stop_every_thread_at(breakpoint, stopped)
        return line

    def test_runs_to_exit(self):
        """simplePrintf runs to exit under LLDB: both targets report its exit
        status, and every GPU thread's printf output reaches its stdout."""
        self.run_to_launch()
        self.setAsync(True)
        self.assertSuccess(self.cpu_process.Continue())
        self.assert_exit_status(0)
        self.check_output(self.read_stdout())

    def test_breakpoint_by_file_and_line(self):
        """A breakpoint on a kernel line, set before the program runs, stops
        every GPU thread on that line with the sample's launch configuration
        and kernel argument.

        This is the one test that relies on lldb copying the CPU target's
        breakpoints into the GPU target when it creates it: that copy is how a
        breakpoint set before the run reaches the kernel. Other tests set GPU
        breakpoints on the GPU target once the program has stopped.
        """
        target = self.make_target()
        line = line_number(SOURCE, "// gpu breakpoint at printf")
        breakpoint = target.BreakpointCreateByLocation(SOURCE, line)
        self.run_to_launch(target)

        # The copy keeps the breakpoint's ID. lldb also resolved the line in
        # the kernel's host-side stub, and would stop the CPU there after the
        # launch (test_kernel_line_breakpoint_does_not_stop_cpu), so delete the
        # CPU target's breakpoint: only the GPU should stop.
        gpu_breakpoint = self.gpu_target.FindBreakpointByID(breakpoint.GetID())
        self.assertTrue(gpu_breakpoint.IsValid(), "the GPU target has no copy")
        self.assertTrue(target.BreakpointDelete(breakpoint.GetID()))

        # The copy resolves when the kernel's cubin loads, at the launch.
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assertEqual(gpu_breakpoint.GetNumLocations(), 1)
        self.assertEqual(self.check_stop(gpu_breakpoint), line)

    def test_breakpoint_by_function_name(self):
        """A breakpoint on the kernel's name stops every GPU thread at the start
        of the kernel. Once resumed, the program finishes, and the printf of
        each thread that was stopped reaches stdout."""
        self.run_to_launch()
        breakpoint = self.gpu_target.BreakpointCreateByName("simplePrintfKernel")

        self.continue_cpu_and_wait_for_gpu_to_stop()
        line = self.check_stop(breakpoint)
        self.assertEqual(breakpoint.GetNumLocations(), 1)
        # The kernel's first line after its prologue, which depends on the
        # code nvcc generates: check only that it is in the kernel's body.
        self.assertGreater(
            line, line_number(SOURCE, "__global__ void simplePrintfKernel")
        )

        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        self.assert_exit_status(0)
        self.check_output(self.read_stdout())

    def test_printf_output_at_gpu_stop(self):
        """Device printf output reaches the CPU process's stdout while the GPU
        is still stopped in the kernel, once its threads have run their
        printf, and not only when the program finishes."""
        self.run_to_launch()
        breakpoint = self.gpu_target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// gpu breakpoint after printf")
        )

        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.stop_every_thread_at(
            breakpoint, self.assert_gpu_stop_at_breakpoint(breakpoint)
        )
        # Every thread has run its printf and the GPU is stopped. The program
        # has written the lines, but lldb reads them on a thread of its own,
        # so give them a moment to arrive.
        stdout = self.read_stdout()
        deadline = time.time() + 10
        while len(PRINTED.findall(stdout)) < len(self.THREADS):
            if time.time() > deadline:
                break
            time.sleep(0.1)
            stdout += self.read_stdout()
        self.check_output(stdout)

        self.gpu_target.BreakpointDelete(breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        self.assert_exit_status(0)

    @expectedFailureAll(
        "a breakpoint on a kernel line also stops the CPU in the kernel's "
        "host-side stub"
    )
    def test_kernel_line_breakpoint_does_not_stop_cpu(self):
        """A breakpoint on a kernel line, set before the program runs, stops
        the GPU but never the CPU. lldb also resolves the line in the kernel's
        host-side stub and stops the CPU there after the launch; cuda-gdb
        creates that location disabled and stops only the GPU.

        The CPU must keep running while the GPU is stopped, as it does under
        lit's NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1. In the default mode the
        CPU stops with the GPU, and the test fails for that reason instead."""
        target = self.make_target()
        breakpoint = target.BreakpointCreateByLocation(
            SOURCE, line_number(SOURCE, "// gpu breakpoint at printf")
        )
        self.run_to_launch(target)
        gpu_breakpoint = self.gpu_target.FindBreakpointByID(breakpoint.GetID())
        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(gpu_breakpoint)

        self.gpu_target.BreakpointDelete(gpu_breakpoint.GetID())
        self.assertSuccess(self.gpu_process.Continue())
        # The CPU kept running while the GPU was stopped, and nothing should
        # stop it before the program exits.
        self.assertEqual(self.wait_for_exit(self.cpu_process), 0, "CPU exit")
        self.assertEqual(self.wait_for_exit(self.gpu_process), 0, "GPU exit")

    def test_kernel_name_breakpoint_before_run(self):
        """A breakpoint on the kernel's name, set before the program runs,
        stops the GPU at the start of the kernel."""
        target = self.make_target()
        breakpoint = target.BreakpointCreateByName("simplePrintfKernel")
        self.run_to_launch(target)
        gpu_breakpoint = self.gpu_target.FindBreakpointByID(breakpoint.GetID())
        self.assertTrue(gpu_breakpoint.IsValid(), "the GPU target has no copy")
        # The CPU target resolved the name in the kernel's host-side stub too;
        # delete its breakpoint so that only the GPU can stop.
        self.assertTrue(target.BreakpointDelete(breakpoint.GetID()))

        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.assert_gpu_stop_at_breakpoint(gpu_breakpoint)
