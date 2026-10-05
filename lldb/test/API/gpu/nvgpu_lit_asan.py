"""Run the NVIDIA GPU API tests on an AddressSanitizer build.

The gpu/nvidia and gpu/comparison/nvidia lit.local.cfg files call configure()
when the build uses LLVM_USE_SANITIZER=Address; nothing else is needed. A test
file then fails when ASan reports in any process it started, and the report is
added to the test output.
"""

import glob
import os
import platform
import re
import shutil
import subprocess

import lit.Test
import lldbtest  # lit.cfg.py puts lldb/test/API on sys.path.


def configure(config):
    compiler = config.cmake_cxx_compiler
    # dotest's Python is not instrumented but loads the instrumented liblldb,
    # so the ASan runtime has to be loaded first.
    config.environment["LD_PRELOAD"] = _ask(
        compiler, f"-print-file-name=libclang_rt.asan-{platform.machine()}.so"
    )
    # detect_leaks=0: CPython and the lldb driver that dotest runs leak at exit.
    # protect_shadow_gap=0: the CUDA driver's debugger back-end maps memory into
    #   ASan's shadow gap, and lldb-server cannot initialize the debug API
    #   without it.
    # symbolize=0: lldb stops lldb-server soon after it starts to exit, and a
    #   report from its shutdown keeps its stack more often when ASan does not
    #   stop to symbolize it. The test format symbolizes the frames afterward.
    config.environment["ASAN_OPTIONS"] = (
        config.environment.get("ASAN_OPTIONS", "")
        + ":detect_leaks=0:protect_shadow_gap=0:symbolize=0"
    )
    # dotest drops the preload before it starts anything: lldb-server carries
    # its own runtime, and nvcc crashes under it.
    config.test_format = ASanLLDBTest(
        config.test_format.dotest_cmd + ["-u", "LD_PRELOAD"],
        _ask(compiler, "-print-prog-name=llvm-symbolizer"),
    )


def _ask(compiler, flag):
    return subprocess.check_output([compiler, flag], text=True).strip()


# A raw frame: "    #3 0x5634135fe222  (/path/lldb-server+0x2578221) (BuildId: 9f...)"
_RAW_FRAME = re.compile(
    r"^( *#\d+ 0x[0-9a-f]+) +\(([^()\s]+)\+(0x[0-9a-f]+)\).*$", re.M
)


class ASanLLDBTest(lldbtest.LLDBTest):
    """lldb's API test format, failing a test file that produced an ASan report.

    lldb starts lldb-server with its stdio on /dev/null, so a report from it
    would be lost, and one written while it shuts down, after the test's last
    assertion, would leave the test green. Each test file gets its own report
    directory, <test exec path>.asan/, with reports named report.<process>.<pid>.
    """

    def __init__(self, dotest_cmd, symbolizer):
        super().__init__(dotest_cmd)
        self.symbolizer = symbolizer

    def execute(self, test, litConfig):
        report_dir = test.getExecPath() + ".asan"
        shutil.rmtree(report_dir, ignore_errors=True)
        os.makedirs(report_dir)
        log_path = os.path.join(report_dir, "report")
        # Each test reaches its lit worker as its own copy, so this does not
        # leak into other test files.
        test.config.environment["ASAN_OPTIONS"] += (
            ":log_exe_name=1:log_path=" + log_path
        )
        code, output = super().execute(test, litConfig)
        reports = sorted(glob.glob(log_path + ".*"))
        if not reports:
            return code, output
        text = ""
        for path in reports:
            with open(path, errors="replace") as report:
                text += f"\nAddressSanitizer report {path}:\n{report.read()}"
        return lit.Test.FAIL, output + self._symbolize(text)

    def _symbolize(self, text):
        """Replace raw frames with function and source line, now, while the
        binaries are still the ones that wrote the reports."""
        frames = list(_RAW_FRAME.finditer(text))
        found = {}
        for binary in {frame[2] for frame in frames}:
            offsets = sorted({frame[3] for frame in frames if frame[2] == binary})
            # Without --no-debuginfod, a DEBUGINFOD_URLS server, which Ubuntu
            # sets by default, is asked about every binary: about 10 s each.
            command = [self.symbolizer, "--no-debuginfod", "--pretty-print"]
            try:
                result = subprocess.run(
                    command + ["--obj=" + binary, *offsets],
                    capture_output=True,
                    text=True,
                )
            except OSError:
                return text
            blocks = result.stdout.strip().split("\n\n")
            found.update(((binary, offset), b) for offset, b in zip(offsets, blocks))

        def symbolized(frame):
            where = found.get((frame[2], frame[3]), "??")
            if not where or where.startswith("??"):
                return frame[0]
            indent = "\n" + " " * len(frame[1])
            return f"{frame[1]} in {where}".replace("\n", indent)

        return _RAW_FRAME.sub(symbolized, text)
