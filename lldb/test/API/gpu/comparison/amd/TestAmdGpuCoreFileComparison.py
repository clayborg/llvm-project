"""
AMD GPU Core File Comparison Test

Compares LLDB and ROCgdb behavior when debugging AMD GPU core files.
This test verifies that both debuggers produce equivalent results for
GPU debugging scenarios.

TEST STRUCTURE:
- TestAmdGpuCoreFileComparison inherits from GpuTestCaseBase and contains
  helper methods for loading core files, comparing variables, etc.
- For each *.core file in the Inputs/ subdirectory, test methods are
  dynamically generated:
    test_thread_count__<basename>
    test_registers__<basename>
    test_local_variables__<basename>

ARCHITECTURAL DIFFERENCE:
- LLDB: Creates TWO targets (CPU + GPU). Must use `target select` to switch
  between them. `thread list` only shows threads for the selected target.
- ROCgdb: Has a "flat view" where all threads (CPU and GPU) are visible together.

This means comparisons must explicitly select the correct target in LLDB to
match what ROCgdb shows.

CONFIGURATION:
- ROCgdb path: Looks for 'rocgdb' in PATH
- Core files: Place *.core files in the 'Inputs/' subdirectory next to this test
- ROCgdb environment: LD_PRELOAD and LD_LIBRARY_PATH can be configured via
  ROCGDB_LD_PRELOAD and ROCGDB_LD_LIBRARY_PATH environment variables
"""

import glob
import os
import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test import configuration

# Add parent directory to path to import the shared comparison framework
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from framework.comparator import ResultComparator
from framework.gdb_driver import GdbDriver
from framework.lldb_driver import LldbDriver


def _get_default_core_dir():
    """Get the default core file directory (Inputs subdirectory next to this test)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, "Inputs")


def _get_rocgdb_path():
    """Get ROCgdb path by looking in PATH.
    TODO: make this configurable via lit configuration.
    """
    return shutil.which("rocgdb")


def _get_core_files():
    """Get list of core files from the default Inputs directory."""
    core_dir = _get_default_core_dir()

    if not os.path.isdir(core_dir):
        return []

    pattern = os.path.join(core_dir, "*.core")
    return sorted(glob.glob(pattern))


class TestAmdGpuCoreFileComparison(GpuTestCaseBase):
    """Compares LLDB and ROCgdb behavior on AMD GPU core files.

    For each *.core file in Inputs/, test methods are dynamically added
    (e.g. test_thread_count__mycore, test_registers__mycore,
    test_local_variables__mycore).  Each method loads the core file in both
    debuggers, runs the comparison, and tears down.

    Inherits target selection helpers (cpu_target, gpu_target, select_cpu,
    select_gpu, cpu_process, gpu_process) from GpuTestCaseBase.
    """

    NO_DEBUG_INFO_TESTCASE = True

    # ------------------------------------------------------------------
    # Per-core-file setup / teardown helpers
    # ------------------------------------------------------------------

    def _load_core(self, core_path):
        """Load a core file in both debuggers and return (gdb_driver, lldb_driver, comparator)."""
        rocgdb_path = _get_rocgdb_path()
        if not rocgdb_path:
            self.skipTest("ROCgdb not found in PATH")

        if not os.path.exists(core_path):
            self.skipTest(f"Core file not found: {core_path}")

        self.trace(f"Using ROCgdb: {rocgdb_path}")
        self.trace(f"Testing with core file: {core_path}")

        ld_preload = configuration.rocgdb_ld_preload
        ld_library_path = configuration.rocgdb_ld_library_path

        gdb_driver = GdbDriver(
            rocgdb_path,
            ld_preload=ld_preload,
            ld_library_path=ld_library_path,
        )

        lldb_driver = LldbDriver(self.dbg)

        comparator = ResultComparator(
            ignore_thread_names=True,
            ignore_thread_ids=True,
            normalize_function_names=True,
            pc_tolerance=0,
        )

        gdb_driver.load_core(core_path)
        lldb_driver.load_core(core_path)

        # Store for cleanup in tearDown
        self._active_gdb_driver = gdb_driver
        self._active_lldb_driver = lldb_driver

        return gdb_driver, lldb_driver, comparator

    def tearDown(self):
        if hasattr(self, "_active_gdb_driver") and self._active_gdb_driver:
            self._active_gdb_driver.cleanup()
            self._active_gdb_driver = None
        if hasattr(self, "_active_lldb_driver") and self._active_lldb_driver:
            self._active_lldb_driver.cleanup()
            self._active_lldb_driver = None
        TestBase.tearDown(self)

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def _get_lldb_gpu_local_variables(self, frame):
        """Get local variables from an LLDB frame as a dict.

        Returns:
            dict mapping variable name -> {"value": str, "type": str}
        """
        lldb_vars = {}
        vars_list = frame.GetVariables(True, True, False, True)
        for i in range(vars_list.GetSize()):
            var = vars_list.GetValueAtIndex(i)
            lldb_vars[var.GetName()] = {
                "value": var.GetValue(),
                "type": var.GetTypeName(),
            }
        return lldb_vars

    def _compare_variable_sets(self, comparator, gdb_vars, lldb_vars):
        """Compare variable sets between GDB and LLDB.

        Returns:
            Tuple of (only_in_gdb, only_in_lldb, read_failures, value_mismatches)
        """
        gdb_var_names = {v.name for v in gdb_vars.variables}
        lldb_var_names = set(lldb_vars.keys())

        only_in_gdb = gdb_var_names - lldb_var_names
        only_in_lldb = lldb_var_names - gdb_var_names
        common = gdb_var_names & lldb_var_names

        read_failures = []
        value_mismatches = []
        for name in common:
            gdb_var = next(v for v in gdb_vars.variables if v.name == name)
            lldb_value = lldb_vars[name]["value"]
            gdb_value = gdb_var.value

            if lldb_value is None or lldb_value == "":
                read_failures.append(
                    {
                        "name": name,
                        "gdb_value": gdb_value,
                        "lldb_value": f"<read failed: {lldb_value}>",
                    }
                )
                continue

            normalized_gdb = comparator.normalize_pointer_value(gdb_value)
            normalized_lldb = comparator.normalize_pointer_value(lldb_value)

            if normalized_gdb != normalized_lldb:
                value_mismatches.append(
                    {
                        "name": name,
                        "gdb_value": gdb_value,
                        "lldb_value": lldb_value,
                        "normalized_gdb": normalized_gdb,
                        "normalized_lldb": normalized_lldb,
                    }
                )

        return only_in_gdb, only_in_lldb, read_failures, value_mismatches

    # ------------------------------------------------------------------
    # Core comparison logic (called by the generated test methods)
    # ------------------------------------------------------------------

    def _run_thread_count_comparison(self, core_path):
        """Compare total thread counts between debuggers for a core file."""
        gdb_driver, lldb_driver, comparator = self._load_core(core_path)

        gdb_result = gdb_driver.get_all_threads()

        self.select_cpu()
        cpu_proc = self.cpu_process
        if not cpu_proc or not cpu_proc.IsValid():
            self.skipTest("LLDB CPU target not found")

        lldb_cpu_thread_count = cpu_proc.GetNumThreads()

        self.trace(f"GDB total threads (flat view): {len(gdb_result.threads)}")
        self.trace(f"LLDB CPU threads: {lldb_cpu_thread_count}")

        self.select_gpu()
        gpu_proc = self.gpu_process
        if gpu_proc and gpu_proc.IsValid():
            self.trace(f"LLDB GPU threads: {gpu_proc.GetNumThreads()}")
            lldb_total = lldb_cpu_thread_count + gpu_proc.GetNumThreads()
            self.trace(f"LLDB total (CPU + GPU): {lldb_total}")

            self.assertEqual(
                len(gdb_result.threads),
                lldb_total,
                f"Total thread count mismatch: GDB={len(gdb_result.threads)}, LLDB={lldb_total}",
            )

    def _run_register_comparison(self, core_path):
        """Compare register values between debuggers for a core file."""
        gdb_driver, lldb_driver, comparator = self._load_core(core_path)

        gdb_result = gdb_driver.get_registers()
        lldb_result = lldb_driver.get_registers()

        self.trace(f"\nGDB registers: {len(gdb_result.registers)}")
        self.trace(f"LLDB registers: {len(lldb_result.registers)}")

        comparison = comparator.compare_registers(gdb_result, lldb_result)

        if comparison.differences:
            self.trace(f"\nRegister differences ({len(comparison.differences)}):")
            for diff in comparison.differences[:10]:
                self.trace(f"  {diff.description}")

    def _run_local_variables_comparison(self, core_path):
        """Compare GPU local variables between debuggers for a core file.

        Both debuggers select the crashing thread by default when loading a core.
        We rely on this default selection rather than searching for threads,
        which would change GDB's selected thread state.
        """
        gdb_driver, lldb_driver, comparator = self._load_core(core_path)

        self.select_gpu()
        gpu_proc = self.gpu_process
        if not gpu_proc or not gpu_proc.IsValid():
            self.skipTest("LLDB GPU target not found")

        if gpu_proc.GetNumThreads() == 0:
            self.skipTest("No GPU threads in LLDB")

        lldb_gpu_thread = gpu_proc.GetSelectedThread()
        if not lldb_gpu_thread.IsValid():
            lldb_gpu_thread = gpu_proc.GetThreadAtIndex(0)

        lldb_frame = lldb_gpu_thread.GetFrameAtIndex(0)
        self.trace(
            f"\nLLDB selected GPU thread: id={lldb_gpu_thread.GetThreadID()}, "
            f"PC={hex(lldb_frame.GetPC())}, "
            f"func={lldb_frame.GetFunctionName() or '<unknown>'}"
        )

        # Get local variables from GDB using the default selected thread.
        # IMPORTANT: Do NOT call get_all_threads() here as it changes GDB's
        # selected thread!
        gdb_vars = gdb_driver.get_local_variables()

        # Get local variables from LLDB
        lldb_vars = self._get_lldb_gpu_local_variables(lldb_frame)

        self.trace(f"\n=== Local Variables Comparison ===")
        self.trace(f"GDB variables: {len(gdb_vars.variables)}")
        self.trace(f"LLDB variables: {len(lldb_vars)}")

        for v in gdb_vars.variables:
            self.trace(f"  GDB: {v.name} ({v.type_name}) = {v.value}")
        for name, info in lldb_vars.items():
            self.trace(f"  LLDB: {name} ({info['type']}) = {info['value']}")

        # Compare
        only_in_gdb, only_in_lldb, read_failures, value_mismatches = (
            self._compare_variable_sets(comparator, gdb_vars, lldb_vars)
        )

        if only_in_gdb:
            self.trace(f"\nVariables only in GDB: {only_in_gdb}")
        if only_in_lldb:
            self.trace(f"Variables only in LLDB: {only_in_lldb}")
        if read_failures:
            self.trace(f"\nLLDB read failures ({len(read_failures)}):")
            for f in read_failures:
                self.trace(
                    f"  {f['name']}: GDB={f['gdb_value']}, LLDB={f['lldb_value']}"
                )
        if value_mismatches:
            self.trace(f"\nValue mismatches ({len(value_mismatches)}):")
            for m in value_mismatches:
                self.trace(
                    f"  {m['name']}: GDB={m['gdb_value']}, LLDB={m['lldb_value']}"
                )

        # Log per-variable match status
        gdb_var_names = {v.name for v in gdb_vars.variables}
        common = gdb_var_names & set(lldb_vars.keys())
        self.trace(f"\nValue comparison:")
        for name in common:
            gdb_var = next(v for v in gdb_vars.variables if v.name == name)
            gdb_val = gdb_var.value
            lldb_val = lldb_vars[name]["value"]
            normalized_gdb = comparator.normalize_pointer_value(gdb_val)
            normalized_lldb = comparator.normalize_pointer_value(lldb_val)
            match = "MATCH" if normalized_gdb == normalized_lldb else "MISMATCH"
            self.trace(f"  {name}: GDB={gdb_val}, LLDB={lldb_val} [{match}]")

        # Assertions
        self.assertEqual(
            len(only_in_gdb),
            0,
            f"Variables found in GDB but missing in LLDB: {only_in_gdb}",
        )

        self.assertEqual(
            len(read_failures),
            0,
            f"LLDB failed to read {len(read_failures)} variables:\n"
            + "\n".join(
                f"  {f['name']}: GDB={f['gdb_value']}, LLDB={f['lldb_value']}"
                for f in read_failures
            ),
        )

        self.assertEqual(
            len(value_mismatches),
            0,
            f"Variable value mismatches between GDB and LLDB:\n"
            + "\n".join(
                f"  {m['name']}: GDB={m['gdb_value']}, LLDB={m['lldb_value']}"
                for m in value_mismatches
            ),
        )

    # ------------------------------------------------------------------
    # Placeholder when no core files are available
    # ------------------------------------------------------------------

    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_placeholder(self):
        """Placeholder that skips when no core files are present."""
        if _get_core_files():
            return  # real tests exist; nothing to do
        self.skipTest(
            f"No GPU core files found. Place *.core files in "
            f"'{_get_default_core_dir()}'"
        )


# ------------------------------------------------------------------
# Dynamically add test methods for each core file
# ------------------------------------------------------------------

def _add_core_file_tests():
    """For each *.core in Inputs/, add test methods to the test class.

    Generates methods like:
        test_thread_count__mycore
        test_registers__mycore
        test_local_variables__mycore
    """
    core_files = _get_core_files()
    for core_path in core_files:
        basename = os.path.basename(core_path).replace(".", "_").replace("-", "_")

        # Use default-arg capture (core_path=core_path) to bind the loop variable.
        def make_thread_count_test(cp=core_path):
            @skipUnlessArch("x86_64")
            @skipUnlessPlatform(["linux"])
            def test(self):
                self._run_thread_count_comparison(cp)
            test.__doc__ = f"Thread count comparison for {os.path.basename(cp)}"
            return test

        def make_register_test(cp=core_path):
            @skipUnlessArch("x86_64")
            @skipUnlessPlatform(["linux"])
            def test(self):
                self._run_register_comparison(cp)
            test.__doc__ = f"Register comparison for {os.path.basename(cp)}"
            return test

        def make_local_variables_test(cp=core_path):
            @skipUnlessArch("x86_64")
            @skipUnlessPlatform(["linux"])
            def test(self):
                self._run_local_variables_comparison(cp)
            test.__doc__ = f"Local variables comparison for {os.path.basename(cp)}"
            return test

        setattr(
            TestAmdGpuCoreFileComparison,
            f"test_thread_count__{basename}",
            make_thread_count_test(),
        )
        setattr(
            TestAmdGpuCoreFileComparison,
            f"test_registers__{basename}",
            make_register_test(),
        )
        setattr(
            TestAmdGpuCoreFileComparison,
            f"test_local_variables__{basename}",
            make_local_variables_test(),
        )


_add_core_file_tests()
