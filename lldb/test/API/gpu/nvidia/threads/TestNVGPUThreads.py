import re

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVGPUThreads(NVGPUTestCaseBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_thread_in_a_single_block(self):
        """Test that we can read all threads in a single block, and one of them has excepted."""
        self.killCPUOnTeardown()

        self.build()
        source = "threads.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)
        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.continue_cpu_and_wait_for_gpu_to_stop()

        self.select_gpu()

        self.assertEqual(len(self.gpu_process.threads), 512)

        thread_with_exception = self.find_thread_by_name("threadIdx(x=5 y=0 z=0)")
        self.assertEqual(thread_with_exception.GetStopReason(), lldb.eStopReasonException)

        thread_without_exception = self.find_thread_by_name("threadIdx(x=511 y=0 z=0)")
        self.assertEqual(thread_without_exception.GetStopReason(), lldb.eStopReasonNone)

    def test_thread_list_aggregation(self):
        """Test that `thread list` aggregates GPU threads by source line and that
        `-v` falls back to the legacy per-PC layout."""
        self.killCPUOnTeardown()

        self.build()
        source = "threads.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)
        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.select_gpu()

        # Default (aggregated) mode: groups are summarized by source line, the
        # raw "pc=0x" prefix is suppressed in favor of "module`function at
        # file:line", and large thread counts are visible in the summary line.
        self.expect(
            "thread list",
            substrs=[
                "thread(s):",
                f"at {source}:",
            ],
            matching=True,
        )
        # Non-faulting warps report errorPC == 0; those threads must fall back
        # to their per-thread PC and aggregate by source line. They must never
        # collapse onto a synthetic "pc=0x" or "errorPC=0x0" location (the
        # latter would mean a backend errorPC of 0 was mistaken for a real
        # fault address).
        self.expect(
            "thread list", substrs=["pc=0x", "errorPC=0x0"], matching=False
        )

        # Verbose mode opts out of GPU-specific aggregation and renders one
        # row per thread, like `thread list` on a CPU target. Each row uses
        # the standard `thread #N: tid = ...` prefix and embeds the CUDA
        # coordinates in the thread name.
        self.expect(
            "thread list -v",
            substrs=["thread #", "tid = 0x", "name = 'blockIdx"],
            matching=True,
        )
        self.expect("thread list -v", substrs=["thread(s):"], matching=False)

        # --stop-reason exception (default rendering): filters down to threads
        # whose stop reason is a CUDA exception. The thread that dereferenced
        # 0x03 must still be there; threads from non-faulting warps are folded
        # into a trailing summary row.
        self.expect(
            "thread list -s exception",
            substrs=["thread(s):", "hidden by --stop-reason filter"],
            matching=True,
        )
        self.expect(
            "thread list -s exception",
            substrs=["threadIdx(x=511 y=0 z=0), stop reason"],
            matching=False,
        )

        # An unknown stop reason name is rejected at option-parsing time rather
        # than being treated as a free-form substring that silently matches
        # nothing. "SIGTRAP" is not one of the canonical stop reason names
        # produced by `thread list` (and accepted by
        # Thread::StopReasonFromString), so the command must error out.
        self.expect(
            "thread list -s SIGTRAP",
            error=True,
            substrs=["invalid stop reason 'SIGTRAP'"],
        )

        # Combining -v and --stop-reason: per-thread rendering, filtered to
        # threads matching the filter expression.
        self.expect(
            "thread list -v -s exception",
            substrs=["thread #"],
            matching=True,
        )

    def test_thread_list_aggregation_row_format(self):
        """Pin down the exact format of an aggregated `thread list` row on a
        GPU target. This complements `test_thread_list_aggregation` (which
        covers high-level switches: aggregation vs `-v`, `--stop-reason`
        filtering, the trailing summary row) by verifying the precise
        rendering produced by PlatformNVGPU::GetGPUThreadStatus.

        Each group is rendered as two lines:

            <prefix> N thread(s): blockIdx(...) threadIdx(...)[, stop reason = ...]
                <cubin>`<function> at <basename>:<line>

        where <prefix> is `*` for the group containing the selected thread
        and a space otherwise, and the `, stop reason = ...` suffix is only
        emitted for groups whose stop reason is valid and should be shown.

        The kernel launches `<<<1, 512>>>`, so blockIdx is single-valued
        in every dimension (exercises the `name=V` form of FormatDimSet)
        and threadIdx.{y,z} are also single-valued. The on-device control
        flow happens to produce all three threadIdx.x forms in the same
        output, so this test exercises every FormatDimSet branch:

          * single value `x=5`             -- the lone faulting thread,
          * contiguous range `x=[32...511]` -- the threads still at the
            store after the second `__syncthreads()`,
          * wildcard `x=*`                  -- a warp where the surviving
            lanes form a non-contiguous set after the divergent branch.
        """
        self.killCPUOnTeardown()

        self.build()
        source = "threads.cu"
        cpu_bp_line: int = line_number(source, "// before kernel launch")
        exit_bp_line: int = line_number(source, "// breakpoint before exit")

        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(source), cpu_bp_line)
        self.cpu_target.BreakpointCreateByLocation(lldb.SBFileSpec(source), exit_bp_line)

        self.continue_cpu_and_wait_for_gpu_to_stop()
        self.select_gpu()

        # Capture the aggregated output once and run all assertions against
        # it; re-running `thread list` per assertion would also re-resolve
        # symbols on the GPU target and slow the test down for no gain.
        self.runCmd("thread list")
        aggregated = self.res.GetOutput()

        # Header line schema. blockIdx is single-valued in every dimension
        # because the kernel launches one block. threadIdx.x is one of the
        # three FormatDimSet renderings (single value, contiguous range, or
        # wildcard). threadIdx.{y,z} are always single-valued.
        header_pattern = re.compile(
            r"^[* ] \d+ thread\(s\): "
            r"blockIdx\(x=0 y=0 z=0\) "
            r"threadIdx\(x=(?:\d+|\[\d+\.\.\.\d+\]|\*) y=0 z=0\)"
            r"(?:, stop reason = .+)?$",
            re.MULTILINE,
        )
        header_matches = header_pattern.findall(aggregated)
        self.assertGreater(
            len(header_matches),
            0,
            "No header lines matched the aggregated row schema in:\n"
            + aggregated,
        )

        # Exactly one row must be marked with `*` -- the group that owns the
        # selected thread.
        selected_marker_pattern = re.compile(
            r"^\* \d+ thread\(s\): blockIdx\(", re.MULTILINE
        )
        self.assertEqual(
            len(selected_marker_pattern.findall(aggregated)),
            1,
            "Expected exactly one `* N thread(s):` row in:\n" + aggregated,
        )

        # All three FormatDimSet branches must appear in this run.
        self.assertRegex(
            aggregated,
            r"threadIdx\(x=\d+ y=0 z=0\)",
            "Missing single-value threadIdx form",
        )
        self.assertRegex(
            aggregated,
            r"threadIdx\(x=\[\d+\.\.\.\d+\] y=0 z=0\)",
            "Missing contiguous-range threadIdx form",
        )
        self.assertRegex(
            aggregated,
            r"threadIdx\(x=\* y=0 z=0\)",
            "Missing wildcard threadIdx form",
        )

        # Location-line schema (LineAndFunction tier): module name,
        # backtick, demangled function (which includes the parameter list
        # for CUDA kernels), " at ", source basename (never a full path),
        # colon, line number.
        self.assertRegex(
            aggregated,
            r"`setAndCopyKernel\(.*\) at threads\.cu:\d+",
            "Missing module`function-with-args at file:line location line",
        )
        # Defensive check: the basename rendering must not regress to
        # printing the absolute path.
        self.assertNotRegex(
            aggregated,
            r"at /.*/threads\.cu:",
            "Location line should print only the basename of the source file",
        )

        # The faulting thread (tid 5 reading 0x03) must show up as its own
        # group with a CUDA exception stop reason. This pins down the exact
        # combination of single-value threadIdx + stop reason rendering.
        self.assertRegex(
            aggregated,
            r"\d+ thread\(s\): blockIdx\(x=0 y=0 z=0\) "
            r"threadIdx\(x=5 y=0 z=0\), "
            r"stop reason = CUDA Exception",
            "Did not find the expected faulting-thread group",
        )

        # Aggregation must actually collapse threads. The 512 lanes should
        # be summarised in a handful of groups, never one row per lane.
        self.assertLess(
            aggregated.count(" thread(s): "),
            16,
            "Aggregated `thread list` produced too many rows for a 512-lane "
            "kernel:\n" + aggregated,
        )

        # `thread list -v` is the per-thread fallback. Confirm it does emit
        # one row per lane, so the aggregation above is a real reduction
        # rather than a coincidence of having few threads to begin with.
        self.runCmd("thread list -v")
        verbose = self.res.GetOutput()
        self.assertGreaterEqual(
            verbose.count("thread #"),
            512,
            "`thread list -v` should render one row per lane; got:\n" + verbose,
        )
