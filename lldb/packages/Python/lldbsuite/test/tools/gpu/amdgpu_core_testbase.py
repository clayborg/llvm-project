import lldb
import os
import shutil
import subprocess

from lldbsuite.test.lldbtest import TestBase


# TODO: change to use GpuTestCaseBase
class AmdGpuCoreTestBase(TestBase):
    """Base class for GPU core file tests with automated rocgdb core generation."""

    NO_DEBUG_INFO_TESTCASE = True

    # --- Subclasses must set these ---
    HIP_SOURCE = None
    GPU_BREAKPOINT_PATTERN = None

    # --- Optional overrides ---
    ROCGDB_PATH = None
    ROCGDB_SETUP_COMMANDS = []

    # Class-level cache — generate once, reuse across test methods.
    _core_file_path = None
    _generation_error = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._core_file_path = None
        cls._generation_error = None

    def setUp(self):
        TestBase.setUp(self)

        rocgdb = self._find_rocgdb()
        if rocgdb is None:
            self.skipTest("rocgdb not found — required for GPU core generation")
        self.rocgdb_path = rocgdb

        if (
            self.__class__._core_file_path is None
            and self.__class__._generation_error is None
        ):
            self._generate_core()

        if self.__class__._generation_error:
            self.skipTest(f"Core generation failed: {self.__class__._generation_error}")

        self.core_file_path = self.__class__._core_file_path

    # -----------------------------------------------------------------
    # rocgdb discovery
    # -----------------------------------------------------------------

    def _find_rocgdb(self):
        if self.ROCGDB_PATH:
            return self.ROCGDB_PATH if os.path.isfile(self.ROCGDB_PATH) else None

        candidates = [shutil.which("rocgdb")]
        rocm_path = os.environ.get("ROCM_PATH")
        if rocm_path:
            candidates.insert(0, os.path.join(rocm_path, "bin", "rocgdb"))
        candidates.append("/opt/rocm/bin/rocgdb")

        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    # -----------------------------------------------------------------
    # Core generation
    # -----------------------------------------------------------------

    def _find_breakpoint_line(self):
        source_path = os.path.join(self.getSourceDir(), self.HIP_SOURCE)
        with open(source_path, "r") as f:
            for i, line in enumerate(f, 1):
                if self.GPU_BREAKPOINT_PATTERN in line:
                    return i
        self.fail(f"Pattern '{self.GPU_BREAKPOINT_PATTERN}' not found in {source_path}")

    def _generate_core(self):
        try:
            self.build()
            binary = self.getBuildArtifact("a.out")
            if not os.path.isfile(binary):
                raise RuntimeError(f"Build did not produce: {binary}")

            bp_line = self._find_breakpoint_line()
            bp_spec = f"{self.HIP_SOURCE}:{bp_line}"
            core_path = self.getBuildArtifact("gpu_test.core")

            script_path = self._build_rocgdb_script(binary, bp_spec, core_path)
            self._run_rocgdb(script_path)

            actual_core = self._find_generated_core(core_path)
            if actual_core is None:
                raise RuntimeError(
                    f"rocgdb gcore did not produce a core at {core_path}*"
                )

            self.__class__._core_file_path = actual_core

        except Exception as e:
            self.__class__._generation_error = str(e)

    def _build_rocgdb_script(self, binary, bp_spec, core_path):
        commands = [
            "set pagination off",
            "set confirm off",
            "set breakpoint pending on",
        ]
        commands.extend(self.ROCGDB_SETUP_COMMANDS)
        commands.extend(
            [
                f"file {binary}",
                f"break {bp_spec}",
                "run",
                f"gcore {core_path}",
                "quit",
            ]
        )

        script_path = self.getBuildArtifact("rocgdb_commands.txt")
        with open(script_path, "w") as f:
            f.write("\n".join(commands) + "\n")
        return script_path

    def _run_rocgdb(self, script_path, timeout_seconds=120):
        cmd = [self.rocgdb_path, "--batch", "-x", script_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=self._get_rocgdb_env(),
        )
        if result.stdout:
            self.trace(f"rocgdb stdout:\n{result.stdout}")
        if result.stderr:
            self.trace(f"rocgdb stderr:\n{result.stderr}")

    def _get_rocgdb_env(self):
        from lldbsuite.test import configuration

        env = os.environ.copy()
        env.setdefault("HSA_ENABLE_DEBUG", "1")
        if configuration.rocgdb_ld_preload:
            env["LD_PRELOAD"] = configuration.rocgdb_ld_preload
        if configuration.rocgdb_ld_library_path:
            existing = env.get("LD_LIBRARY_PATH", "")
            if existing:
                env["LD_LIBRARY_PATH"] = (
                    f"{configuration.rocgdb_ld_library_path}:{existing}"
                )
            else:
                env["LD_LIBRARY_PATH"] = configuration.rocgdb_ld_library_path
        return env

    def _find_generated_core(self, base_path):
        if os.path.isfile(base_path):
            return base_path
        parent = os.path.dirname(base_path)
        prefix = os.path.basename(base_path)
        for name in os.listdir(parent):
            if name.startswith(prefix) and os.path.isfile(os.path.join(parent, name)):
                return os.path.join(parent, name)
        return None

    # -----------------------------------------------------------------
    # Core loading
    # -----------------------------------------------------------------

    def load_core(self):
        """Load core file. Returns (gpu_target, gpu_process)."""
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid(), "Failed to create target")

        error = lldb.SBError()
        process = target.LoadCore(self.core_file_path, error)
        self.assertTrue(
            process.IsValid(),
            f"Failed to load core: {error.GetCString()}",
        )
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        gpu_target = self._find_target_by_triple("amdgcn")
        self.assertIsNotNone(gpu_target, "GPU target not found in core")

        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid())
        self.assertEqual(gpu_process.GetState(), lldb.eStateStopped)

        self.dbg.SetSelectedTarget(gpu_target)
        return gpu_target, gpu_process

    # TODO: refactor into GpuTestCaseBase for sharing
    def _find_target_by_triple(self, substring):
        for i in range(self.dbg.GetNumTargets()):
            t = self.dbg.GetTargetAtIndex(i)
            if substring in t.GetTriple():
                return t
        return None

    # -----------------------------------------------------------------
    # Wave-level variable checking helpers
    # -----------------------------------------------------------------

    def get_wave(self, gpu_process, wave_index=0):
        """Get a GPU wave (thread) by index."""
        self.assertGreater(
            gpu_process.GetNumThreads(),
            wave_index,
            f"Need at least {wave_index + 1} wave(s) in core",
        )
        thread = gpu_process.GetThreadAtIndex(wave_index)
        self.assertTrue(thread.IsValid())
        return thread

    def check_wave_variable(self, gpu_process, wave_index, name, expected_values):
        """Check a variable's value for a given wave.

        This asserts at the wave level — for private variables, the
        value shown is the default lane (lane 0), consistent with
        what rocgdb displays for the wave.
        """
        if isinstance(expected_values, str):
            expected_values = [expected_values]
        wave = self.get_wave(gpu_process, wave_index)
        gpu_process.SetSelectedThread(wave)
        self.expect(
            f"frame variable --flat --show-all-children {name}",
            substrs=expected_values,
        )
