"""
Unit tests for the NVIDIA GPU test gating in nvgpu_host.py and
nvgpu_decorators.py.

They need no GPU and never run the real probe, so they check the gating on any
machine, including one without the hardware it is about. A wrong decision
either hides a failure behind UNSUPPORTED or stops a run that should go on, so
every decision to stop, warn or skip is tested here.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import lldb
import lldbsuite
from lldbsuite.test import configuration
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test.tools.gpu import nvgpu_decorators
from lldbsuite.test.tools.gpu import nvgpu_host
from lldbsuite.test.tools.gpu.nvgpu_core_testbase import NVGPUCoreTestBase
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase

gpu_info = nvgpu_host.gpu_info
TWO_GPUS = [[12, 0], [8, 6]]
MAJOR = nvgpu_host.build_major(lldbsuite.lldb_root)


class FakeLibcuda:
    """Stands in for libcuda.so.1 in probe_child(). Like the real library,
    each function writes its result through the ctypes.byref() pointers it
    is given."""

    def __init__(
        self, debugger_major=13, init_result=0, devices=TWO_GPUS, attribute_result=0
    ):
        self.debugger_major = debugger_major
        self.init_result = init_result
        self.devices = devices
        self.attribute_result = attribute_result

    def cudbgGetAPIVersion(self, major, minor, revision):
        major._obj.value = self.debugger_major
        return 0

    def cuInit(self, flags):
        return self.init_result

    def cuGetErrorName(self, result, name):
        name._obj.value = b"CUDA_ERROR_UNKNOWN"
        return 0

    def cuDeviceGetCount(self, count):
        count._obj.value = len(self.devices)
        return 0

    def cuDeviceGet(self, device, ordinal):
        device._obj.value = ordinal
        return 0

    def cuDeviceGetAttribute(self, value, attribute, device):
        if self.attribute_result:
            return self.attribute_result
        # 75 and 76 are CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR.
        value._obj.value = self.devices[device.value][{75: 0, 76: 1}[attribute]]
        return 0


class FakeLitConfig:
    """Records what lit_local_config() prints."""

    def __init__(self, params=None):
        self.params = params or {}
        self.messages = []

    def note(self, message):
        self.messages.append(("note", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class FakeConfig:
    def __init__(self):
        self.environment = {}
        self.lldb_src_root = lldbsuite.lldb_root
        self.lldb_platform_url = ""


class TestNVGPUGating(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.addCleanup(
            setattr, nvgpu_decorators, "_gpu_info", nvgpu_decorators._gpu_info
        )

    def remote_platform(self, platform):
        """Make the checks see `platform` as the remote platform, or none if
        it is None, whatever platform this test runs on."""
        return mock.patch.object(lldb, "remote_platform", platform)

    def dotest_nvcc(self, nvcc):
        """Make the checks see `nvcc` as the nvcc that dotest builds with, as
        --nvcc-path sets it."""
        return mock.patch.object(configuration, "nvcc_path", nvcc)

    def old_nvcc(self):
        """Return an nvcc that cannot build for sm_120."""
        return self.script("old-nvcc", "echo compute_86; echo compute_90")

    def script(self, name, body):
        """Write an executable shell script that runs `body`, and return its
        path."""
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write("#!/bin/sh\n" + body + "\n")
        os.chmod(path, 0o755)
        return path

    def test_device_reason(self):
        reason = nvgpu_host.device_reason
        self.assertIsNone(reason(gpu_info(TWO_GPUS, 13), 13))
        # If either CUDA major is unknown, the tests run.
        self.assertIsNone(reason(gpu_info(TWO_GPUS, None), 13))
        self.assertIsNone(reason(gpu_info(TWO_GPUS, 13), None))
        self.assertEqual(
            reason(gpu_info(TWO_GPUS, 12), 13),
            "the driver's CUDA debugger API is major 12 but this lldb targets "
            "CUDA 13; use a CUDA 13 driver or an lldb built for CUDA 12",
        )
        self.assertIn("no CUDA device is visible", reason(gpu_info([], 13), 13))
        self.assertIn(
            "--param nvgpu-no-device",
            reason(gpu_info([], 13, no_device_param=True), 13),
        )
        self.assertEqual(reason(gpu_info(error="cuInit failed"), 13), "cuInit failed")

    def test_decide(self):
        def levels(info, device_nodes=True):
            messages = nvgpu_host.decide(info, 13, device_nodes)
            return [level for level, _ in messages]

        gpus = gpu_info(TWO_GPUS, 13)
        self.assertEqual(levels(gpus), ["note"])
        # A machine with GPUs that will skip the tests that need one gets a
        # warning, so that the run does not look like a normal one on a machine
        # without GPUs.
        self.assertEqual(levels(gpu_info(TWO_GPUS, 12)), ["note", "warning"])
        self.assertEqual(levels(gpu_info(error="cuInit failed")), ["note", "warning"])
        # Not when the skip was asked for, or when there is no GPU.
        self.assertEqual(
            levels(gpu_info([], 13, no_device_param=True)), ["note", "note"]
        )
        self.assertEqual(
            levels(gpu_info(error="no libcuda.so.1"), device_nodes=False),
            ["note", "note"],
        )
        self.assertEqual(
            nvgpu_host.decide(gpus, 13, True)[0][1],
            "nvgpu: GPUs: sm_120, sm_86; driver debugger API: 13; "
            "this lldb targets CUDA 13",
        )
        level, warning = nvgpu_host.decide(gpu_info(TWO_GPUS, 12), 13, True)[1]
        self.assertTrue(
            warning.startswith(
                "nvgpu: tests that need a GPU will be UNSUPPORTED: the driver's"
            )
        )
        # With a remote platform, the tests that need a GPU skip, so lit warns
        # about that and not about this machine's GPUs.
        remote_warning = (
            "warning",
            "nvgpu: tests that need a GPU will be UNSUPPORTED: "
            + nvgpu_host.REMOTE_REASON,
        )
        for info in [gpus, gpu_info([], 13)]:
            messages = nvgpu_host.decide(info, 13, False, "connect://gpu:1234")
            self.assertEqual(messages[1:], [remote_warning])

    def test_nvcc_problem(self):
        problem = nvgpu_host.nvcc_problem
        nvcc = self.script(
            "nvcc",
            '[ "$1" = --list-gpu-arch ] || exit 2\n'
            "echo compute_86; echo compute_90; echo compute_120",
        )
        self.assertIsNone(problem(nvcc, TWO_GPUS))
        old = self.old_nvcc()
        self.assertEqual(problem(old, TWO_GPUS), f"{old} cannot build for compute_120")
        broken = self.script("broken-nvcc", "echo broken >&2; exit 1")
        self.assertIn("exited with 1: 'broken'", problem(broken, TWO_GPUS))
        missing = os.path.join(self.tmp, "missing-nvcc")
        self.assertIn("cannot run", problem(missing, TWO_GPUS))

    def test_probe(self):
        # Stand-ins for the Python interpreter that probe() starts.
        result = nvgpu_host.RESULT_PREFIX + json.dumps(gpu_info(TWO_GPUS, 13))
        good = self.script("python", f"echo stray output; echo '{result}'")
        self.assertEqual(nvgpu_host.probe(python=good), gpu_info(TWO_GPUS, 13))
        mute = self.script("mute-python", "echo crashed >&2; exit 3")
        self.assertIn(
            "exited with 3 and no result: 'crashed'",
            nvgpu_host.probe(python=mute)["error"],
        )
        slow = self.script("slow-python", "exec sleep 10")
        self.assertIn(
            "did not finish in 0.2 s",
            nvgpu_host.probe(python=slow, timeout=0.2)["error"],
        )

    def test_probe_child(self):
        child = nvgpu_host.probe_child
        self.assertEqual(child(lambda name: FakeLibcuda()), gpu_info(TWO_GPUS, 13))
        # A machine with no visible GPU is not an error.
        self.assertEqual(
            child(lambda name: FakeLibcuda(init_result=100)), gpu_info([], 13)
        )
        self.assertEqual(
            child(lambda name: FakeLibcuda(init_result=999))["error"],
            "cuInit failed with CUDA_ERROR_UNKNOWN (999); check that nvidia-smi "
            "works (a driver updated without a reboot fails this way)",
        )
        # A failed query is an error, not a GPU of compute capability 0.0.
        self.assertEqual(
            child(lambda name: FakeLibcuda(attribute_result=999))["error"],
            "listing the GPUs failed with CUDA_ERROR_UNKNOWN (999)",
        )

        def no_driver(name):
            raise OSError(f"{name}: cannot open shared object file")

        self.assertIn("libcuda.so.1 cannot be loaded", child(no_driver)["error"])

    def test_build_major(self):
        header_dir = os.path.join(self.tmp, "third-party", "cuda")
        os.makedirs(header_dir)
        self.assertIsNone(nvgpu_host.build_major(self.tmp))
        for text, major in [
            ("#define CUDBG_API_VERSION_MAJOR 14\n", 14),
            ("#define CUDBG_API_VERSION_MINOR 4\n", None),
        ]:
            with open(os.path.join(header_dir, "cudadebugger.h"), "w") as f:
                f.write(text)
            self.assertEqual(nvgpu_host.build_major(self.tmp), major)
        # The header in the source tree has a major, whatever its value.
        self.assertIsInstance(MAJOR, int)

    def test_lit_local_config_probes_once(self):
        probes = []

        def probe(env):
            probes.append(env)
            return gpu_info(TWO_GPUS, MAJOR)

        lit_config = FakeLitConfig()
        configs = [FakeConfig(), FakeConfig()]
        with mock.patch.object(nvgpu_host, "probe", probe):
            for config in configs:
                nvgpu_host.lit_local_config(config, lit_config)
        # Both NVIDIA directories share lit_config: one probe and one note, and
        # each directory's tests get the GPU info.
        self.assertEqual(len(probes), 1)
        self.assertEqual([level for level, _ in lit_config.messages], ["note"])
        for config in configs:
            published = config.environment[nvgpu_host.GPU_INFO_ENV]
            self.assertEqual(json.loads(published), gpu_info(TWO_GPUS, MAJOR))

    def test_lit_local_config_no_device_param(self):
        seen = []

        def probe(env):
            seen.append(env.get("CUDA_VISIBLE_DEVICES"))
            return gpu_info([], MAJOR)

        config = FakeConfig()
        lit_config = FakeLitConfig({"nvgpu-no-device": ""})
        with mock.patch.object(nvgpu_host, "probe", probe):
            nvgpu_host.lit_local_config(config, lit_config)
        # The GPUs are hidden from the probe and from the tests alike.
        self.assertEqual(seen, [""])
        self.assertEqual(config.environment["CUDA_VISIBLE_DEVICES"], "")
        published = config.environment[nvgpu_host.GPU_INFO_ENV]
        self.assertTrue(json.loads(published)["no_device_param"])
        self.assertEqual(
            lit_config.messages[-1],
            (
                "note",
                "nvgpu: tests that need a GPU will be UNSUPPORTED: --param "
                "nvgpu-no-device hides the GPUs",
            ),
        )

    def test_lit_local_config_remote_platform(self):
        config = FakeConfig()
        config.lldb_platform_url = "connect://gpu:1234"
        lit_config = FakeLitConfig()
        with mock.patch.object(
            nvgpu_host, "probe", return_value=gpu_info(TWO_GPUS, MAJOR)
        ):
            nvgpu_host.lit_local_config(config, lit_config)
        level, message = lit_config.messages[-1]
        self.assertEqual(level, "warning")
        self.assertIn(nvgpu_host.REMOTE_REASON, message)

    def test_which_tests_need_a_gpu(self):
        # A method's mark decides, then NEEDS_GPU on its class or base class:
        # live tests need a GPU, core-file tests do not.
        mark = nvgpu_decorators.skipUnlessNVGPUDevice

        class Live(NVGPUTestCaseBase):
            def test_method(self):
                pass

        class GPUFreeLive(NVGPUTestCaseBase):
            NEEDS_GPU = False

            def test_method(self):
                pass

            @mark
            def test_live(self):
                pass

        class Core(NVGPUCoreTestBase):
            def test_method(self):
                pass

            @mark
            def test_live(self):
                pass

        def needs(cls, name):
            return nvgpu_decorators.needs_gpu(cls(name))

        self.assertTrue(needs(Live, "test_method"))
        self.assertFalse(needs(GPUFreeLive, "test_method"))
        self.assertTrue(needs(GPUFreeLive, "test_live"))
        self.assertFalse(needs(Core, "test_method"))
        self.assertTrue(needs(Core, "test_live"))
        # A class says it needs a GPU with NEEDS_GPU, not with the method mark.
        with self.assertRaises(TypeError):
            mark(Live)

    def test_setup_skips_without_a_gpu(self):
        nvgpu_decorators._gpu_info = gpu_info([], MAJOR)

        class Live(NVGPUTestCaseBase):
            def test_method(self):
                pass

        # The usual LLDB setUp calls TestBase.setUp directly, which bypasses
        # NVGPUTestCaseBase.setUp. The check runs anyway.
        class LiveWithSetUp(NVGPUTestCaseBase):
            def setUp(self):
                TestBase.setUp(self)

            def test_method(self):
                pass

        class Core(NVGPUCoreTestBase):
            def test_method(self):
                pass

        # TestBase.setUp would start a second debugger inside this test, and
        # the skip must come before it, which deletes the build directory.
        with self.remote_platform(None), mock.patch.object(
            TestBase, "setUp"
        ) as base_setup:
            for cls in [Live, LiveWithSetUp]:
                with self.assertRaisesRegex(unittest.SkipTest, "no CUDA device"):
                    cls("test_method").setUp()
            self.assertFalse(base_setup.called)
            Core("test_method").setUp()

    def test_setup_fails_on_unusable_nvcc(self):
        # setUp checks the nvcc that dotest builds with. lit passes its
        # --nvcc-path first, so a later dotest argument can replace it.
        nvgpu_decorators._gpu_info = gpu_info(TWO_GPUS, MAJOR)
        old = self.old_nvcc()
        new = self.script("new-nvcc", "echo compute_86; echo compute_120")

        class Live(NVGPUTestCaseBase):
            def test_method(self):
                pass

        class Core(NVGPUCoreTestBase):
            def test_method(self):
                pass

        with self.remote_platform(None), mock.patch.object(TestBase, "setUp"):
            with self.dotest_nvcc(old):
                with self.assertRaisesRegex(
                    AssertionError,
                    "old-nvcc cannot build for compute_120; set NVGPU_NVCC_PATH",
                ):
                    Live("test_method").setUp()
                # A test that needs no GPU builds nothing for one.
                Core("test_method").setUp()
            with self.dotest_nvcc(new):
                Live("test_method").setUp()
            # Without an nvcc, the build fails and says so.
            with self.dotest_nvcc(None):
                Live("test_method").setUp()

    def test_bare_dotest_checks_nvcc(self):
        # A bare lldb-dotest gets no GPU info from lit, so the tests run the
        # probe themselves, and they check nvcc as they do under lit.
        nvgpu_decorators._gpu_info = None
        old = self.old_nvcc()

        class Live(NVGPUTestCaseBase):
            def test_method(self):
                pass

        probe = mock.patch.object(
            nvgpu_host, "probe", return_value=gpu_info(TWO_GPUS, MAJOR)
        )
        base_setup = mock.patch.object(TestBase, "setUp")
        with self.remote_platform(None), self.dotest_nvcc(old), probe, base_setup:
            with mock.patch.dict(os.environ):
                os.environ.pop(nvgpu_host.GPU_INFO_ENV, None)
                with self.assertRaisesRegex(
                    AssertionError, "old-nvcc cannot build for compute_120"
                ):
                    Live("test_method").setUp()

    def test_generate_core_needs_a_gpu(self):
        # A core-file test needs no GPU until it makes a core by running a
        # program on one. Then, before building anything, it skips on a
        # machine without a GPU and fails with an nvcc that cannot build.
        class Core(NVGPUCoreTestBase):
            def test_method(self):
                pass

        with self.remote_platform(None), self.dotest_nvcc(self.old_nvcc()):
            nvgpu_decorators._gpu_info = gpu_info([], MAJOR)
            with self.assertRaisesRegex(unittest.SkipTest, "no CUDA device"):
                Core("test_method").generate_core()
            nvgpu_decorators._gpu_info = gpu_info(TWO_GPUS, MAJOR)
            with self.assertRaisesRegex(
                AssertionError, "old-nvcc cannot build for compute_120"
            ):
                Core("test_method").generate_core()

    def test_remote_platform_skips(self):
        # The GPU info describes the machine that runs lit, not the remote
        # platform, so the tests that need a GPU skip there, before an nvcc
        # that cannot build for this machine's GPUs would fail them.
        nvgpu_decorators._gpu_info = gpu_info(TWO_GPUS, MAJOR)
        old = self.old_nvcc()

        class Live(NVGPUTestCaseBase):
            def test_method(self):
                pass

        class Core(NVGPUCoreTestBase):
            def test_method(self):
                pass

        remote = lldb.SBPlatform("remote-linux")
        base_setup = mock.patch.object(TestBase, "setUp")
        with self.remote_platform(remote), self.dotest_nvcc(old), base_setup:
            with self.assertRaisesRegex(unittest.SkipTest, "remote platforms"):
                Live("test_method").setUp()
            # A test that needs no GPU runs until it makes a core.
            core = Core("test_method")
            core.setUp()
            with self.assertRaisesRegex(unittest.SkipTest, "remote platforms"):
                core.generate_core()
