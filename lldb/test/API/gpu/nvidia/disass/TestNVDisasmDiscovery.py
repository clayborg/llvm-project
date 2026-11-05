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
from lldbsuite.test.lldbtest import TestBase


class TestNVDisasmDiscovery(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.calls = self.tmp / "calls"

    def tool(self, directory, opcode, architectures):
        path = self.tmp / directory / "nvdisasm"
        path.parent.mkdir()
        output = [{}, [{"start": 0, "sass-instructions": [{"opcode": opcode}]}]]
        help_text = ",".join(f"'{sm}'" for sm in architectures)
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

    def disassemble(self, tools, cubins, override=None):
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(str(tool.parent) for tool in tools)
        env.pop("CUDA_HOME", None)
        args = [lldbtest_config.lldbExec, "--no-lldbinit", "-b"]
        args += ["-o", "settings clear platform.plugin.nvgpu.nvdisasm-path"]
        if override:
            args += [
                "-o",
                "settings set platform.plugin.nvgpu.nvdisasm-path "
                + shlex.quote(str(override)),
            ]
        for cubin, address in cubins:
            args += [
                "-o",
                "file " + shlex.quote(str(cubin)),
                "-o",
                f"disassemble -s {address:#x} -c 1",
            ]
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
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
