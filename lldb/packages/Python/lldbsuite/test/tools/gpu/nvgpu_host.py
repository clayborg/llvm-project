"""Probe this machine's NVIDIA GPUs once per lit run, for the GPU tests.

The lit.local.cfg files in gpu/nvidia and gpu/comparison/nvidia call
lit_local_config() while lit collects the tests. The first call runs the probe,
which asks the CUDA driver which GPUs it can see, and then:

- prints what the probe found as a lit note;
- passes the result, the GPU info, to the tests in the NVGPU_TEST_GPU_INFO
  environment variable. A test that needs a GPU skips when the machine cannot
  run one, and fails when the nvcc that dotest builds with cannot build for it
  (nvgpu_testcase.py).

lit loads this file by path, before lldb's Python module can be imported, so it
imports nothing from lldb or lldbsuite. The probe sees only the machine that
runs lit, and the tests rely on lit's environment, which does not reach a
remote platform, so the tests that need a GPU skip when they run on one.
"""

import ctypes
import glob
import json
import os
import re
import subprocess
import sys

GPU_INFO_ENV = "NVGPU_TEST_GPU_INFO"
NO_DEVICE_PARAM = "nvgpu-no-device"
NVCC_REMEDY = (
    "set NVGPU_NVCC_PATH to an nvcc that can and reconfigure, or pass "
    f"--param {NO_DEVICE_PARAM} to run only the GPU-free tests"
)
REMOTE_REASON = "remote platforms are not supported; run lit on the GPU machine"
PROBE_TIMEOUT = 20
# The probe prints its result after this prefix, so that anything else printed
# in the probe's process cannot be mistaken for the result.
RESULT_PREFIX = "NVGPU_TEST_GPU_INFO="

# Enum values from the driver API's cuda.h (CUresult, CUdevice_attribute) and
# from cudadebugger.h (CUDBGResult). ctypes cannot read a header, so they are
# copied here. Compiled programs pass these numbers to the driver, so they are
# part of its ABI and do not change between drivers.
_CUDA_SUCCESS = 0
_CUDA_ERROR_NO_DEVICE = 100
_CUDBG_SUCCESS = 0
# CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR.
_CC_MAJOR_ATTRIBUTE = 75
_CC_MINOR_ATTRIBUTE = 76


def gpu_info(devices=(), debugger_major=None, error=None, no_device_param=False):
    """Return the GPU info: a dictionary of what the probe found. The probe,
    lit and the tests all pass this dictionary around.

    devices: the compute capability of each GPU the tests can see, as
        [major, minor], in CUDA's device order.
    debugger_major: the major version of the driver's CUDA debugger API.
        lldb-server refuses a driver whose major is not the CUDA major lldb
        was built for.
    error: None, or why the probe could not find out.
    no_device_param: True when --param nvgpu-no-device hid the GPUs.
    """
    return {
        "devices": [list(cc) for cc in devices],
        "debugger_major": debugger_major,
        "error": error,
        "no_device_param": no_device_param,
    }


def probe_child(load_library=None):
    """Ask the CUDA driver about this machine's GPUs and return the GPU info.

    This runs in the separate Python process that probe() starts.
    `load_library` replaces ctypes.CDLL in the unit tests.
    """
    try:
        cuda = (load_library or ctypes.CDLL)("libcuda.so.1")
    except OSError as e:
        return gpu_info(
            error=f"libcuda.so.1 cannot be loaded ({e}); run on a machine with an "
            "NVIDIA GPU and driver"
        )

    # lldb-server refuses a driver whose debugger API major is not the one it
    # was built for (CUDADebuggerAPI.cpp), so ask the driver the same question.
    version = [ctypes.c_uint32() for _ in range(3)]
    result = cuda.cudbgGetAPIVersion(*(ctypes.byref(v) for v in version))
    if result != _CUDBG_SUCCESS:
        return gpu_info(error=f"cudbgGetAPIVersion failed with CUDBGResult {result}")
    debugger_major = version[0].value

    result = cuda.cuInit(0)
    if result == _CUDA_ERROR_NO_DEVICE:
        # Not a failure: the machine has no GPU that the tests can see.
        return gpu_info(debugger_major=debugger_major)
    if result != _CUDA_SUCCESS:
        return gpu_info(
            debugger_major=debugger_major,
            error=f"cuInit failed with {_error_name(cuda, result)}; check that "
            "nvidia-smi works (a driver updated without a reboot fails this way)",
        )
    count = ctypes.c_int()
    result = cuda.cuDeviceGetCount(ctypes.byref(count))
    devices = []
    for ordinal in range(count.value if result == _CUDA_SUCCESS else 0):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        result = cuda.cuDeviceGet(ctypes.byref(device), ordinal)
        if result == _CUDA_SUCCESS:
            result = cuda.cuDeviceGetAttribute(ctypes.byref(major), _CC_MAJOR_ATTRIBUTE, device)
        if result == _CUDA_SUCCESS:
            result = cuda.cuDeviceGetAttribute(ctypes.byref(minor), _CC_MINOR_ATTRIBUTE, device)
        if result != _CUDA_SUCCESS:
            break
        devices.append([major.value, minor.value])
    if result != _CUDA_SUCCESS:
        return gpu_info(
            debugger_major=debugger_major,
            error=f"listing the GPUs failed with {_error_name(cuda, result)}",
        )
    return gpu_info(devices, debugger_major)


def _error_name(cuda, result):
    """The name of CUDA error `result`, such as
    "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (803)"."""
    name = ctypes.c_char_p()
    if cuda.cuGetErrorName(result, ctypes.byref(name)) == _CUDA_SUCCESS and name.value:
        return f"{name.value.decode()} ({result})"
    return f"CUresult {result}"


def probe(env=None, timeout=PROBE_TIMEOUT, python=None):
    """Run probe_child() in a separate Python process and return the GPU info.

    The separate process keeps the CUDA driver out of lit's own process, and a
    driver that hangs costs at most `timeout` seconds. `env` should be the
    environment the test programs get. If the probe fails, the GPU info's error
    says why. `python` replaces this Python in the unit tests.
    """
    command = [python or sys.executable, "-I", os.path.abspath(__file__)]
    try:
        done = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return gpu_info(error=f"the GPU probe did not finish in {timeout} s")
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return gpu_info(
        error=f"the GPU probe exited with {done.returncode} and no result: "
        f"{done.stderr.strip()[-300:]!r}"
    )


def build_major(lldb_src_root):
    """Return the CUDA major version this lldb was built for, or None if it
    cannot be read.

    It is CUDBG_API_VERSION_MAJOR in the copy of cudadebugger.h under
    lldb/third-party/cuda. A build can compile against another copy of the
    header, but a static_assert in CUDADebuggerAPIVersion.cpp requires that
    copy to have the same major as this one.
    """
    path = os.path.join(lldb_src_root, "third-party", "cuda", "cudadebugger.h")
    try:
        with open(path, errors="replace") as f:
            match = re.search(
                r"^\s*#\s*define\s+CUDBG_API_VERSION_MAJOR\s+\(?(\d+)", f.read(), re.M
            )
    except OSError:
        return None
    return int(match.group(1)) if match else None


def device_reason(info, build_major):
    """Return None if the tests that need a GPU can run here. Otherwise return
    why not and what would fix it, which becomes the tests' skip reason.

    If either CUDA major is unknown, the tests run.
    """
    if info["error"]:
        return info["error"]
    if not info["devices"]:
        if info["no_device_param"]:
            return f"--param {NO_DEVICE_PARAM} hides the GPUs"
        return "no CUDA device is visible; run on a machine with an NVIDIA GPU"
    driver = info["debugger_major"]
    if driver is not None and build_major is not None and driver != build_major:
        return (
            f"the driver's CUDA debugger API is major {driver} but this lldb "
            f"targets CUDA {build_major}; use a CUDA {build_major} driver or an "
            f"lldb built for CUDA {driver}"
        )
    return None


def nvcc_problem(nvcc, devices):
    """Return None if `nvcc` can build programs for every GPU in `devices`,
    otherwise what is wrong. The test Makefiles build for every visible GPU
    (-arch=native), so nvcc must support all of them."""
    try:
        done = subprocess.run(
            [nvcc, "--list-gpu-arch"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"{nvcc} cannot run: {e}"
    if done.returncode != 0:
        return (
            f"'{nvcc} --list-gpu-arch' exited with {done.returncode}: "
            f"{done.stderr.strip()[-300:]!r}"
        )
    missing = sorted(
        {"compute_{}{}".format(*cc) for cc in devices} - set(done.stdout.split())
    )
    if missing:
        return f"{nvcc} cannot build for {', '.join(missing)}"
    return None


def decide(info, build_major, has_gpu_device_nodes, platform_url=None):
    """Return what lit should print about the GPUs, as (level, message) pairs,
    where level is "note" or "warning".

    has_gpu_device_nodes: whether /dev/nvidia0, /dev/nvidia1 and so on exist,
        even if the probe found no GPU.
    platform_url: the remote platform the tests run on, or None.

    Nothing here stops the run: lit would stop every test suite in it, not
    only the GPU tests.
    """
    messages = [("note", "nvgpu: " + _summary(info, build_major))]
    if platform_url:
        messages.append(
            (
                "warning",
                f"nvgpu: tests that need a GPU will be UNSUPPORTED: {REMOTE_REASON}",
            )
        )
        return messages
    reason = device_reason(info, build_major)
    if reason is None:
        return messages
    # Warn when a machine that has GPUs will skip every test that needs one,
    # unless --param nvgpu-no-device asked for that. Otherwise the run would
    # look like a normal run on a machine without GPUs.
    has_gpus = bool(info["devices"]) or has_gpu_device_nodes
    level = "warning" if has_gpus and not info["no_device_param"] else "note"
    messages.append(
        (level, f"nvgpu: tests that need a GPU will be UNSUPPORTED: {reason}")
    )
    return messages


def _summary(info, build_major):
    """One line describing the GPUs and driver, for lit's note."""
    devices = ", ".join("sm_{}{}".format(*cc) for cc in info["devices"])
    driver = info["debugger_major"]
    return (
        f"GPUs: {devices or 'none'}; driver debugger API: "
        f"{'unknown' if driver is None else driver}; "
        f"this lldb targets CUDA {build_major}"
    )


def lit_local_config(config, lit_config):
    """Set up the GPU checks for the test directory that `config` describes.

    The probe runs in `config.environment` as it is when this is called.
    Both NVIDIA directories share `lit_config`, which keeps the GPU info, so
    the probe runs once per lit run.
    """
    no_device = NO_DEVICE_PARAM in lit_config.params
    if no_device:
        # Hide the GPUs from the probe and from every test program.
        config.environment["CUDA_VISIBLE_DEVICES"] = ""
    info = getattr(lit_config, "nvgpu_test_gpu_info", None)
    if info is None:
        major = build_major(config.lldb_src_root)
        info = probe(dict(config.environment))
        info["no_device_param"] = no_device
        has_nodes = bool(glob.glob("/dev/nvidia[0-9]*"))
        url = config.lldb_platform_url
        for level, message in decide(info, major, has_nodes, url):
            getattr(lit_config, level)(message)
        lit_config.nvgpu_test_gpu_info = info
    config.environment[GPU_INFO_ENV] = json.dumps(info)


# probe() runs this file as a script in a separate process.
if __name__ == "__main__":
    print(RESULT_PREFIX + json.dumps(probe_child()))
