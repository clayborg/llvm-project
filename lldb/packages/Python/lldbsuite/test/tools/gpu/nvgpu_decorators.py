"""Whether an NVIDIA GPU API test needs a GPU, and whether this machine has one.

API tests ignore lit's REQUIRES lines, so whether a test needs a GPU is decided
in Python, by the test's base class. A test derived from NVGPUTestCaseBase
debugs a live CUDA program, so it needs a GPU; a core-file test derived from
NVGPUCoreTestCaseBase does not. Each base class says so in its NEEDS_GPU
attribute, which a test class can set too. NVGPUTestCaseBase skips a test that
needs a GPU on a machine that cannot run one, with a reason that says what
would let it run, and lit reports a test file whose tests were all skipped as
UNSUPPORTED. NVGPUCoreTestCaseBase.generate_core skips the same way, because it
runs the test program on the GPU to make a core.

In a class that needs no GPU, a method that runs a CUDA program on the GPU
carries @skipUnlessNVGPUDevice. A method that needs more of the GPU, such as a
minimum compute capability, carries @skipUnlessNVGPU with the requirements
defined here, which implies a GPU too:

    class TestNVGPUDisass(NVGPUTestCaseBase):
        NEEDS_GPU = False

        @skipUnlessNVGPUDevice
        def test_disass_live_program(self):
            ...

        @skipUnlessNVGPU(SM(at_least=90))
        def test_disass_clusters(self):
            ...

The device checks use the GPU info that lit's probe passed to the tests (see
nvgpu_host.py); under a bare lldb-dotest, this module runs the probe itself.
Either way, the nvcc check runs here, on the nvcc that dotest builds with.

A test declares what it needs, as NEEDS_GPU and the marks do, and setUp checks
it before the test starts. Add the next requirement, such as a number of GPUs,
as a class like SM, with its first test. A class decorator would decide when
the file is imported and override the marks on the class's methods. Tests do
not read the GPU info: it describes the machine that runs lit, not the one
that runs the program on a remote platform.
"""

import json
import os

import lldb
import lldbsuite
from lldbsuite.test import configuration
from lldbsuite.test.tools.gpu import nvgpu_host

_gpu_info = None


def _get_gpu_info():
    """Return the GPU info that lit passed to the tests or, under a bare
    lldb-dotest, the result of running the probe here."""
    global _gpu_info
    if _gpu_info is None:
        published = os.environ.get(nvgpu_host.GPU_INFO_ENV)
        if published is not None:
            _gpu_info = json.loads(published)
        else:
            _gpu_info = nvgpu_host.probe()
    return _gpu_info


# The oldest GPU architecture that CUDA 13 builds for.
_OLDEST_SM = 75


class SM:
    """A minimum compute capability of device 0, the GPU a program uses unless
    it selects another, written as the digits after sm_: SM(at_least=90) needs
    sm_90 or newer."""

    def __init__(self, *, at_least):
        # A smaller number is a mistake, such as 9 for sm_90, or a minimum that
        # every GPU meets; either way the test would never skip.
        if at_least < _OLDEST_SM:
            raise ValueError(
                f"SM(at_least={at_least}): every GPU that CUDA 13 supports is "
                f"sm_{_OLDEST_SM} or newer. Write the digits after sm_, such as "
                "90 for sm_90, or drop a requirement that every GPU meets"
            )
        self.at_least = at_least

    def reason(self, devices):
        """Return None if `devices`, the compute capability of each GPU in
        CUDA's device order, meet this requirement, otherwise why not."""
        major, minor = devices[0]
        # A minor version is a single digit, so sm_XY is X * 10 + Y.
        if major * 10 + minor >= self.at_least:
            return None
        return (
            f"device 0 is sm_{major}{minor}; the test needs sm_{self.at_least} or newer"
        )


def skipUnlessNVGPU(*requirements):
    """Mark a test method that runs a CUDA program on a GPU that meets
    `requirements`, such as SM(at_least=90), so that it is skipped on a machine
    without one. A method has one mark, which holds all its requirements."""

    def mark(func):
        if isinstance(func, type):
            raise TypeError(
                "@skipUnlessNVGPU marks a test method; set NEEDS_GPU on a class"
            )
        if hasattr(func, "NVGPU_REQUIRES"):
            raise TypeError(
                f"{func.__name__} has more than one @skipUnlessNVGPU or "
                "@skipUnlessNVGPUDevice mark; pass every requirement to one "
                "@skipUnlessNVGPU"
            )
        func.NEEDS_GPU = True
        func.NVGPU_REQUIRES = requirements
        return func

    return mark


def skipUnlessNVGPUDevice(func):
    """Mark a test method that runs a CUDA program on a GPU, in a class that
    needs no GPU, so that it is skipped on a machine that cannot run the
    program. A class says the same by setting NEEDS_GPU = True."""
    return skipUnlessNVGPU()(func)


def needs_gpu(test):
    """Whether the method that `test` runs needs a GPU: True if it carries
    @skipUnlessNVGPUDevice, otherwise NEEDS_GPU on its class or base class."""
    method = getattr(test, test._testMethodName)
    return getattr(method, "NEEDS_GPU", getattr(type(test), "NEEDS_GPU", True))


def device_skip_reason():
    """Return why a CUDA program cannot run on a GPU here, starting with
    "nvgpu: " to show where the reason came from, or None if it can."""
    if lldb.remote_platform:
        return "nvgpu: " + nvgpu_host.REMOTE_REASON
    reason = nvgpu_host.device_reason(_get_gpu_info(), _build_major())
    return reason and "nvgpu: " + reason


def skip_reason(test):
    """Return why the method that `test` runs, which needs a GPU, cannot run
    here, starting with "nvgpu: ", or None if it can: device_skip_reason(),
    or a requirement of its @skipUnlessNVGPU marks that device 0 does not
    meet."""
    reason = device_skip_reason()
    if reason:
        return reason
    devices = _get_gpu_info()["devices"]
    method = getattr(test, test._testMethodName)
    for requirement in getattr(method, "NVGPU_REQUIRES", ()):
        reason = requirement.reason(devices)
        if reason:
            return "nvgpu: " + reason
    return None


def nvcc_error():
    """Return why the nvcc that dotest builds the test programs with cannot
    build them for this machine's GPUs, or None. That is dotest's --nvcc-path,
    which lit sets but a later dotest argument can change, so the check runs
    here and not in lit. Without an nvcc, the build fails and says so."""
    nvcc = configuration.nvcc_path
    if not nvcc:
        return None
    error = nvgpu_host.nvcc_problem(nvcc, _get_gpu_info()["devices"])
    return error and f"nvgpu: {error}; {nvgpu_host.NVCC_REMEDY}"


def _build_major():
    return nvgpu_host.build_major(lldbsuite.lldb_root)
