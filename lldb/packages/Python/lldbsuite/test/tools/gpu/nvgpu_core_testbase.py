import os
import re
import subprocess

import lldb

from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class NVGPUCoreTestBase(NVGPUTestCaseBase):
    """Base class for NVGPU core file tests with automated core generation."""

    def setUp(self):
        NVGPUTestCaseBase.setUp(self)
        self._generated_cores = {}

    def generate_core(self, flags="", args=None):
        """Generate a core file if needed and return its path.

        Args:
            flags: CUDA_COREDUMP_GENERATION_FLAGS value controlling core contents.
            args: Optional list of command-line arguments passed to the test
                binary (e.g. to select which exception the kernel triggers).

        Core files are cached by (flags, args) for the duration of the test
        method.
        """
        args = list(args) if args else []
        cache_key = (flags, tuple(args))
        if cache_key in self._generated_cores:
            return self._generated_cores[cache_key]

        name_parts = ["test"]
        if flags:
            name_parts.append(re.sub(r"[^0-9a-zA-Z]", "_", flags))
        if args:
            name_parts.append(re.sub(r"[^0-9a-zA-Z]", "_", "_".join(args)))
        core_name = ".".join(name_parts) + ".nvcudmp"
        core_path = self.getBuildArtifact(core_name)
        build_dir = self.getBuildDir()

        env = os.environ.copy()
        env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] = "1"
        env["CUDA_COREDUMP_FILE"] = core_path
        env["CUDA_COREDUMP_GENERATION_FLAGS"] = flags

        if not getattr(self, "_built_core", False):
            self.build()
            self._built_core = True

        # A stale core from an earlier run must not satisfy the isfile check below.
        try:
            os.remove(core_path)
        except FileNotFoundError:
            pass

        # Nothing here is a skip. Whether this test may run at all (device,
        # driver, nvcc) was decided before it started, so every problem from
        # here on is a real failure and must surface as one. A program that
        # cannot be started after a successful build points at a broken
        # artifact path or build rule. A program that runs past the timeout
        # either never faults (an exception_type with no switch case spins
        # forever) or the driver did not write the core. A program that exits
        # cleanly, or exits without leaving a core, is the same kind of
        # failure. Turning any of these into a skip would let a coredump test
        # go green without ever loading a core.
        argv = [self.getBuildArtifact("a.out"), *args]
        try:
            result = subprocess.run(
                argv,
                cwd=build_dir,
                env=env,
                capture_output=True,
                timeout=120,
            )
        except OSError as e:
            self.fail(f"Core generation could not run {argv}: {e}")
        except subprocess.TimeoutExpired as e:
            stdout_tail = (e.stdout or b"").decode(errors="replace")[-500:]
            stderr_tail = (e.stderr or b"").decode(errors="replace")[-500:]
            self.fail(
                f"Core generation timed out after {e.timeout} s running {argv}; "
                f"core path {core_path} "
                f"(partial file present: {os.path.isfile(core_path)}); "
                f"stdout tail: {stdout_tail!r}; stderr tail: {stderr_tail!r}"
            )
        self.assertNotEqual(
            result.returncode,
            0,
            "Expected the test program to crash on the GPU, but it exited cleanly: "
            f"{result.stderr.decode(errors='replace')[-500:]}",
        )
        self.assertTrue(
            os.path.isfile(core_path),
            f"CUDA core dump not found at {core_path}; program stderr: "
            f"{result.stderr.decode(errors='replace')[-500:]}",
        )

        self._generated_cores[cache_key] = core_path
        return core_path

    def load_core(self, core_file_path):
        """Load a core file in LLDB. Returns (gpu_target, gpu_process)."""
        # Core tests load one core at a time; drop any previously loaded targets
        for target in list(self.dbg):
            self.dbg.DeleteTarget(target)
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid(), "Failed to create target")

        error = lldb.SBError()
        gpu_process = target.LoadCore(core_file_path, error)
        self.assertTrue(
            gpu_process.IsValid(),
            f"Failed to load core: {error.GetCString()}",
        )
        self.assertEqual(gpu_process.GetState(), lldb.eStateStopped)

        gpu_target = gpu_process.GetTarget()
        self.assertEqual(
            gpu_target.GetTriple(),
            "nvptx64-nvidia-cuda",
            "GPU target should take its triple from the core file",
        )

        self.assertEqual(
            gpu_process.GetPluginName(),
            "nvgpu-core",
            "EM_CUDA + ET_CORE corefiles must be handled by ProcessNVGPUCore"
        )

        self.dbg.SetSelectedTarget(gpu_target)
        return gpu_target, gpu_process

    def generate_and_load_core(self, flags="", args=None):
        """Generate a core file (if needed) and load it in LLDB."""
        return self.load_core(self.generate_core(flags, args))

    def generate_artificial_core(self, builder, name="artificial.nvcudmp"):
        """Serialize an NVGPUCoreBuilder to a .nvcudmp via yaml2obj.

        Returns the generated core file path.
        """
        yaml_path = self.getBuildArtifact(name + ".yaml")
        core_path = self.getBuildArtifact(name)
        builder.write_yaml(yaml_path)
        self.yaml2obj(yaml_path, core_path)
        return core_path

    def generate_and_load_artificial_core(self, builder,
                                          name="artificial.nvcudmp"):
        """Build an artificial core from an NVGPUCoreBuilder and load it.

        Returns (gpu_target, gpu_process).
        """
        return self.load_core(self.generate_artificial_core(builder, name))
