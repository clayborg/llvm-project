"""nvdisasm discovery uses mocked tools and requires no GPU or CUDA toolkit."""

import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from lldbsuite.test import lldbtest_config
from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.gpu.nvgpu_testcase import NVGPUTestCaseBase


class TestNVDisasmDiscovery(NVGPUTestCaseBase):
    NEEDS_GPU = False

    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.calls = self.tmp / "calls"

    def tool(self, directory, opcode, architectures):
        path = self.tmp / directory / "nvdisasm"
        path.parent.mkdir(parents=True, exist_ok=True)
        output = [{}, [{"start": 0, "sass-instructions": [{"opcode": opcode}]}]]
        help_text = (
            ",".join(f"'{sm}'" for sm in architectures)
            if architectures is not None
            else "nvdisasm help without an architecture list"
        )
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"with open({str(self.calls)!r}, 'a') as log:\n"
            f"    log.write({opcode!r} + (':help\\n' if '--help' in sys.argv "
            "else ':run\\n'))\n"
            f"print({help_text!r} if '--help' in sys.argv else "
            f"{json.dumps(output)!r})\n"
        )
        path.chmod(0o755)
        return path

    def disassemble(self, tools, cubins, override=None, allow_failure=False):
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(str(tool.parent) for tool in tools)
        env.pop("CUDA_HOME", None)
        env["LLDB_NVDISASM_SEARCH_ROOT"] = str(self.tmp)
        args = [lldbtest_config.lldbExec, "--no-lldbinit", "-b"]
        args += ["-o", 'settings set platform.plugin.nvgpu.nvdisasm-path ""']
        if override:
            args += [
                "-o",
                "settings set platform.plugin.nvgpu.nvdisasm-path "
                + shlex.quote(str(override)),
            ]
        for cubin, address in cubins:
            command = f"disassemble -s {address:#x} -c 1"
            if allow_failure:
                # Keep running after a command fails so the next attempt tests
                # the cache rather than starting a fresh debugger.
                command = (
                    "script r = lldb.SBCommandReturnObject(); "
                    "lldb.debugger.GetCommandInterpreter().HandleCommand("
                    f"{command!r}, r); print((r.GetOutput() or '') + (r.GetError() or ''))"
                )
            args += [
                "-o",
                "file " + shlex.quote(str(cubin)),
                "-o",
                command,
            ]
        # Start a fresh LLDB process for each check: the SASS disassembler cache
        # is process-wide, and dotest runs all methods in this file in one
        # Python process.
        result = subprocess.run(
            args, env=env, cwd=self.tmp, capture_output=True, text=True, timeout=30
        )
        self.output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, self.output)
        return re.findall(r"<\+0>:\s+(OLD_TOOL|NEW_TOOL)", result.stdout)

    def elf_v8(self):
        return (self.getSourcePath("elfv8.cubin"), 0x7FFFCF280300)

    @skipIfWindows
    def test_selects_compatible_tool_and_caches_it(self):
        old = self.tool("old", "OLD_TOOL", ["SM90"])
        new = self.tool("new", "NEW_TOOL", ["SM86"])
        self.assertEqual(
            self.disassemble([old, new], [self.elf_v8(), self.elf_v8()]),
            ["NEW_TOOL", "NEW_TOOL"],
        )
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls.count("OLD_TOOL:help"), 1)
        self.assertEqual(calls.count("NEW_TOOL:help"), 1)

    @skipIfWindows
    def test_caches_selection_per_sm(self):
        old = self.tool("old", "OLD_TOOL", ["SM86"])
        new = self.tool("new", "NEW_TOOL", ["SM90"])
        # Change only the SM metadata in the existing ELFv7 fixture. The mocked
        # tools do not decode the bytes, so no SM90 code or GPU is needed.
        data = bytearray(Path(self.getSourcePath("elfv7.cubin")).read_bytes())
        flags = struct.unpack_from("<I", data, 48)[0]
        struct.pack_into("<I", data, 48, (flags & ~0xFF) | 90)
        sm90 = self.tmp / "sm90.cubin"
        sm90.write_bytes(data)
        self.assertEqual(
            self.disassemble(
                [old, new],
                [self.elf_v8(), (sm90, 0x7FFFD7243B00), self.elf_v8()],
            ),
            ["OLD_TOOL", "NEW_TOOL", "OLD_TOOL"],
        )
        calls = self.calls.read_text().splitlines()
        self.assertEqual(calls.count("OLD_TOOL:help"), 2)
        self.assertEqual(calls.count("NEW_TOOL:help"), 1)

    @skipIfWindows
    def test_explicit_path_is_preserved(self):
        old = self.tool("old", "OLD_TOOL", ["SM90"])
        new = self.tool("new", "NEW_TOOL", ["SM86"])
        self.assertEqual(
            self.disassemble([old, new], [self.elf_v8()], override=old),
            ["OLD_TOOL"],
        )
        self.assertEqual(self.calls.read_text().splitlines(), ["OLD_TOOL:run"])

    @skipIfWindows
    def test_failed_search_is_cached(self):
        old = self.tool("old", "OLD_TOOL", ["SM90"])
        new = self.tool("new", "NEW_TOOL", ["SM86a"])
        self.disassemble([old, new], [self.elf_v8(), self.elf_v8()], allow_failure=True)
        self.assertIn("No installed nvdisasm advertises support for SM86", self.output)
        self.assertIn(str(old), self.output)
        self.assertIn(str(new), self.output)
        self.assertEqual(
            self.calls.read_text().splitlines(), ["OLD_TOOL:help", "NEW_TOOL:help"]
        )

    @skipIfWindows
    def test_unknown_help_is_a_fallback(self):
        unknown = self.tool("unknown", "OLD_TOOL", None)
        compatible = self.tool("compatible", "NEW_TOOL", ["SM86"])
        self.assertEqual(
            self.disassemble([unknown, compatible], [self.elf_v8()]), ["NEW_TOOL"]
        )
        # No advertised support: try the tool with unknown capabilities.
        self.assertEqual(self.disassemble([unknown], [self.elf_v8()]), ["OLD_TOOL"])

    @skipIfWindows
    def test_versioned_directories_and_symlink_deduplication(self):
        self.tool("usr/local/cuda-9.2/bin", "OLDEST_TOOL", ["SM90"])
        old = self.tool("usr/local/cuda-12.8/bin", "OLD_TOOL", ["SM90"])
        self.tool("usr/local/cuda-13.0/bin", "MIDDLE_TOOL", ["SM90"])
        self.tool("usr/local/cuda-14.0/bin", "NEW_TOOL", ["SM90"])
        (self.tmp / "usr/local/cuda").symlink_to(old.parent.parent)
        self.disassemble([], [self.elf_v8(), self.elf_v8()], allow_failure=True)
        self.assertEqual(
            self.calls.read_text().splitlines(),
            ["OLD_TOOL:help", "NEW_TOOL:help", "MIDDLE_TOOL:help", "OLDEST_TOOL:help"],
        )

    @skipIfWindows
    def test_missing_tool_and_empty_path(self):
        self.tool("", "OLD_TOOL", ["SM86"])
        self.disassemble([], [self.elf_v8()], allow_failure=True)
        self.assertIn("Searched directories:", self.output)
        self.assertIn(str(self.tmp / "usr/local/cuda/bin"), self.output)
        self.assertFalse(self.calls.exists(), "empty PATH entries must not search cwd")
