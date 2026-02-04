"""
AMD GPU Core File Comparison Test

Compares LLDB and ROCgdb behavior when debugging AMD GPU core files.
This test verifies that both debuggers produce equivalent results for
GPU debugging scenarios.

TEST STRUCTURE:
- Test classes are dynamically created for each *.core file in the Inputs/
  subdirectory. Each core file gets its own test class (e.g.,
  TestAmdGpuCoreFile_mycore_core) with all comparison test methods.
- The shared comparison framework is located in ../framework/ and provides:
  - LldbDriver: In-process LLDB Python API wrapper (uses self.dbg from TestBase)
  - GdbDriver: ROCgdb subprocess wrapper (GDB's Python API only works inside GDB)
  - ResultComparator: Utilities for comparing debugger outputs

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
import unittest

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

# Add parent directory to path to import the shared comparison framework
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from framework.comparator import ResultComparator
from framework.gdb_driver import GdbDriver
from framework.lldb_driver import LldbDriver


def get_default_core_dir():
    """Get the default core file directory (Inputs subdirectory next to this test)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, "Inputs")


def get_rocgdb_path():
    """Get ROCgdb path by looking in PATH.
    TODO: make this configurable via lit configuration.
    """
    return shutil.which("rocgdb")


def get_core_files():
    """Get list of core files from the default Inputs directory.

    Returns a list of paths to all *.core files in the directory.
    """
    core_dir = get_default_core_dir()

    if not os.path.isdir(core_dir):
        return []

    # Find all .core files in the directory
    pattern = os.path.join(core_dir, "*.core")
    return sorted(glob.glob(pattern))


def make_test_class_for_core(core_path):
    """Factory function to create a test class for a specific core file.

    Creates a TestBase subclass with all comparison test methods, configured
    to use the specified core file. The class name is derived from the core
    file name (e.g., 'mycore.core' -> 'TestAmdGpuCoreFile_mycore_core').

    Args:
        core_path: Path to the GPU core file to test.

    Returns:
        A TestBase subclass configured for the specified core file.
    """

    class _DynamicTestClass(TestBase):
        """Compares LLDB and ROCgdb behavior on AMD GPU core files.

        Uses in-process LLDB Python API (self.dbg) for LLDB operations.
        Uses subprocess for GDB/ROCgdb since we can't load GDB Python in the same process.

        NOTE: LLDB uses a multi-target model (separate CPU and GPU targets) while
        ROCgdb uses a flat view. Tests must account for this difference.
        """

        # Don't test with multiple debug info formats - this test doesn't build code
        NO_DEBUG_INFO_TESTCASE = True

        # Store core path as class attribute
        CORE_PATH = core_path

        def setUp(self):
            TestBase.setUp(self)

            # Get ROCgdb path
            self.rocgdb_path = get_rocgdb_path()
            if not self.rocgdb_path:
                self.skipTest("ROCgdb not found in PATH")

            self.core_path = self.CORE_PATH
            if not os.path.exists(self.core_path):
                self.skipTest(f"Core file not found: {self.core_path}")

            self.trace(f"Using ROCgdb: {self.rocgdb_path}")
            self.trace(f"Testing with core file: {self.core_path}")

            # Get ROCgdb environment from lit configuration or use defaults
            ld_preload = os.environ.get("ROCGDB_LD_PRELOAD")
            ld_library_path = os.environ.get("ROCGDB_LD_LIBRARY_PATH")

            # Create fresh GDB driver for each test method
            self.gdb_driver = GdbDriver(
                self.rocgdb_path,
                ld_preload=ld_preload,
                ld_library_path=ld_library_path,
            )

            # LLDB driver uses in-process API with self.dbg from TestBase
            self.lldb_driver = LldbDriver(self.dbg)

            self.comparator = ResultComparator(
                ignore_thread_names=True,
                ignore_thread_ids=True,
                normalize_function_names=True,
                pc_tolerance=0,
            )

            # Load core file for both debuggers
            self.gdb_driver.load_core(self.core_path)
            self.lldb_driver.load_core(self.core_path)

        def tearDown(self):
            # Cleanup both drivers
            if hasattr(self, "gdb_driver") and self.gdb_driver:
                self.gdb_driver.cleanup()
            if hasattr(self, "lldb_driver") and self.lldb_driver:
                self.lldb_driver.cleanup()
            TestBase.tearDown(self)

        def select_lldb_cpu_target(self):
            """Select the CPU target in LLDB and return its process."""
            cpu_target = self.lldb_driver.find_cpu_target()
            if cpu_target:
                self.dbg.SetSelectedTarget(cpu_target)
                return cpu_target.GetProcess()
            return None

        def select_lldb_gpu_target(self):
            """Select the GPU target in LLDB and return its process."""
            gpu_target = self.lldb_driver.find_gpu_target()
            if gpu_target:
                self.dbg.SetSelectedTarget(gpu_target)
                return gpu_target.GetProcess()
            return None

        @skipUnlessArch("x86_64")
        @skipUnlessPlatform(["linux"])
        def test_total_thread_count_matches(self):
            """Test that both debuggers enumerate the same total number of threads (CPU + GPU)."""
            # ROCgdb shows all threads in flat view (CPU + GPU together)
            gdb_result = self.gdb_driver.get_all_threads()

            # For LLDB, select CPU target specifically
            cpu_process = self.select_lldb_cpu_target()
            if not cpu_process or not cpu_process.IsValid():
                self.skipTest("LLDB CPU target not found")

            lldb_cpu_thread_count = cpu_process.GetNumThreads()

            self.trace(f"GDB total threads (flat view): {len(gdb_result.threads)}")
            self.trace(f"LLDB CPU threads: {lldb_cpu_thread_count}")

            # Get GPU thread count too for reference
            gpu_process = self.select_lldb_gpu_target()
            if gpu_process and gpu_process.IsValid():
                self.trace(f"LLDB GPU threads: {gpu_process.GetNumThreads()}")
                lldb_total = lldb_cpu_thread_count + gpu_process.GetNumThreads()
                self.trace(f"LLDB total (CPU + GPU): {lldb_total}")

                # Compare totals
                self.assertEqual(
                    len(gdb_result.threads),
                    lldb_total,
                    f"Total thread count mismatch: GDB={len(gdb_result.threads)}, LLDB={lldb_total}",
                )

        @skipUnlessArch("x86_64")
        @skipUnlessPlatform(["linux"])
        def test_register_comparison(self):
            """Compare register values between debuggers."""
            gdb_result = self.gdb_driver.get_registers()
            lldb_result = self.lldb_driver.get_registers()

            self.trace(f"\nGDB registers: {len(gdb_result.registers)}")
            self.trace(f"LLDB registers: {len(lldb_result.registers)}")

            comparison = self.comparator.compare_registers(gdb_result, lldb_result)

            # Log differences
            if comparison.differences:
                self.trace(f"\nRegister differences ({len(comparison.differences)}):")
                for diff in comparison.differences[:10]:
                    self.trace(f"  {diff.description}")

        @skipUnlessArch("x86_64")
        @skipUnlessPlatform(["linux"])
        def test_gpu_local_variables_comparison(self):
            """Compare GPU local variables between LLDB and ROCgdb.

            Both debuggers select the crashing thread by default when loading a core.
            We rely on this default selection rather than searching for threads,
            which would change GDB's selected thread state.
            """
            # Select GPU target in LLDB
            gpu_process = self.select_lldb_gpu_target()
            if not gpu_process or not gpu_process.IsValid():
                self.skipTest("LLDB GPU target not found")

            if gpu_process.GetNumThreads() == 0:
                self.skipTest("No GPU threads in LLDB")

            # Get the SELECTED thread in LLDB (the crashing thread)
            lldb_gpu_thread = gpu_process.GetSelectedThread()
            if not lldb_gpu_thread.IsValid():
                lldb_gpu_thread = gpu_process.GetThreadAtIndex(0)

            lldb_frame = lldb_gpu_thread.GetFrameAtIndex(0)
            lldb_pc = lldb_frame.GetPC()
            lldb_func = lldb_frame.GetFunctionName() or "<unknown>"

            self.trace(
                f"\nLLDB selected GPU thread: id={lldb_gpu_thread.GetThreadID()}, PC={hex(lldb_pc)}, func={lldb_func}"
            )

            # Get local variables from GDB using the default selected thread
            # IMPORTANT: Do NOT call get_all_threads() here as it changes GDB's selected thread!
            gdb_vars = self.gdb_driver.get_local_variables()

            # Get local variables from LLDB
            lldb_vars = {}
            for i in range(lldb_frame.GetVariables(True, True, False, True).GetSize()):
                var = lldb_frame.GetVariables(True, True, False, True).GetValueAtIndex(
                    i
                )
                name = var.GetName()
                value = var.GetValue()
                type_name = var.GetTypeName()
                lldb_vars[name] = {"value": value, "type": type_name}

            self.trace(f"\n=== Local Variables Comparison ===")
            self.trace(f"GDB variables: {len(gdb_vars.variables)}")
            self.trace(f"LLDB variables: {len(lldb_vars)}")

            # Log GDB variables
            self.trace(f"\nGDB local variables:")
            for v in gdb_vars.variables:
                self.trace(f"  {v.name} ({v.type_name}) = {v.value}")

            # Log LLDB variables
            self.trace(f"\nLLDB local variables:")
            for name, info in lldb_vars.items():
                self.trace(f"  {name} ({info['type']}) = {info['value']}")

            # Compare variables
            gdb_var_names = {v.name for v in gdb_vars.variables}
            lldb_var_names = set(lldb_vars.keys())

            only_in_gdb = gdb_var_names - lldb_var_names
            only_in_lldb = lldb_var_names - gdb_var_names
            common = gdb_var_names & lldb_var_names

            if only_in_gdb:
                self.trace(f"\nVariables only in GDB: {only_in_gdb}")
            if only_in_lldb:
                self.trace(f"Variables only in LLDB: {only_in_lldb}")

            # Compare variable values, normalizing pointer formats
            read_failures = []
            value_mismatches = []
            for name in common:
                gdb_var = next(v for v in gdb_vars.variables if v.name == name)
                lldb_info = lldb_vars[name]
                lldb_value = lldb_info["value"]
                gdb_value = gdb_var.value

                # Check if LLDB failed to read the variable
                if lldb_value is None or lldb_value == "":
                    read_failures.append(
                        {
                            "name": name,
                            "gdb_value": gdb_value,
                            "lldb_value": f"<read failed: {lldb_value}>",
                        }
                    )
                    continue

                # Normalize and compare values
                normalized_gdb = self.comparator.normalize_pointer_value(gdb_value)
                normalized_lldb = self.comparator.normalize_pointer_value(lldb_value)

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

            # Log successful comparisons
            self.trace(f"\nValue comparison:")
            for name in common:
                gdb_var = next(v for v in gdb_vars.variables if v.name == name)
                lldb_info = lldb_vars[name]
                gdb_val = gdb_var.value
                lldb_val = lldb_info["value"]
                normalized_gdb = self.comparator.normalize_pointer_value(gdb_val)
                normalized_lldb = self.comparator.normalize_pointer_value(lldb_val)
                match = "MATCH" if normalized_gdb == normalized_lldb else "MISMATCH"
                self.trace(f"  {name}: GDB={gdb_val}, LLDB={lldb_val} [{match}]")

            # Assert that LLDB found the same variables as GDB
            self.assertEqual(
                len(only_in_gdb),
                0,
                f"Variables found in GDB but missing in LLDB: {only_in_gdb}",
            )

            # Assert that LLDB can read all variable values (no read failures)
            self.assertEqual(
                len(read_failures),
                0,
                f"LLDB failed to read {len(read_failures)} variables:\n"
                + "\n".join(
                    f"  {f['name']}: GDB={f['gdb_value']}, LLDB={f['lldb_value']}"
                    for f in read_failures
                ),
            )

            # Assert that values match between debuggers (after normalization)
            self.assertEqual(
                len(value_mismatches),
                0,
                f"Variable value mismatches between GDB and LLDB:\n"
                + "\n".join(
                    f"  {m['name']}: GDB={m['gdb_value']}, LLDB={m['lldb_value']}"
                    for m in value_mismatches
                ),
            )

    # Generate a unique class name based on the core file
    core_basename = os.path.basename(core_path).replace(".", "_").replace("-", "_")
    _DynamicTestClass.__name__ = f"TestAmdGpuCoreFile_{core_basename}"
    _DynamicTestClass.__qualname__ = f"TestAmdGpuCoreFile_{core_basename}"

    return _DynamicTestClass


def _create_test_classes():
    """Create test classes for each core file and register them in module globals.

    This function encapsulates the dynamic class creation to prevent loop variables
    from leaking into module scope, which would cause unittest to discover the same
    test class multiple times (once per module-level reference to it).
    """
    core_files = get_core_files()
    if core_files:
        for core_path in core_files:
            test_class = make_test_class_for_core(core_path)
            # Add the test class to the module's global namespace so unittest can find it
            globals()[test_class.__name__] = test_class
    else:
        # If no core files found, create a placeholder test that skips
        class TestAmdGpuCoreFilePlaceholder(TestBase):
            NO_DEBUG_INFO_TESTCASE = True

            @skipUnlessArch("x86_64")
            @skipUnlessPlatform(["linux"])
            def test_placeholder(self):
                self.skipTest(
                    f"No GPU core files found. Place *.core files in '{get_default_core_dir()}'"
                )

        globals()["TestAmdGpuCoreFilePlaceholder"] = TestAmdGpuCoreFilePlaceholder


# Create the test classes
_create_test_classes()
