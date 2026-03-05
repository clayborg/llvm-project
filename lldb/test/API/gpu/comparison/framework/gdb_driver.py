"""
GDB Driver - Maintains a persistent GDB subprocess for automation.

NOTE: Unlike LLDB, GDB's Python API (the `gdb` module) is ONLY available when
running inside GDB itself. You cannot import the `gdb` module in a standalone
Python environment. This is a fundamental difference from LLDB's approach.

LLDB provides the `lldb` module that can be imported in any Python environment
(when LLDB is built with Python support), allowing direct in-process debugging.
GDB, however, embeds Python as a scripting language that runs within GDB's
process space - the `gdb` module simply doesn't exist outside of GDB.

Therefore, the GDB driver maintains a persistent GDB subprocess and communicates
with it via stdin/stdout using Python scripting. This allows state (like thread
selection) to persist across method calls.
"""

import subprocess
import json
import tempfile
import os
import re
import threading
import queue
from typing import List, Optional, Dict, Any

from .debugger_interface import (
    DebuggerInterface,
    DebuggerResult,
    ThreadInfo,
    FrameInfo,
    RegisterValue,
    VariableValue,
    ModuleInfo,
    StopReason,
)


class GdbDriver(DebuggerInterface):
    """
    GDB driver that maintains a persistent GDB subprocess and executes Python
    scripts inside GDB's Python environment to extract structured data.

    State is maintained across method calls since the same GDB process is reused.

    Environment configuration (optional):
        ld_preload: LD_PRELOAD value for ROCgdb Python support.
                    Can be set via ROCGDB_LD_PRELOAD environment variable.
        ld_library_path: LD_LIBRARY_PATH for ROCm libraries.
                         Can be set via ROCGDB_LD_LIBRARY_PATH environment variable.
    """

    # Unique marker to detect end of script output
    END_MARKER = "<<<GDB_SCRIPT_COMPLETE_12345>>>"

    def __init__(
        self, gdb_path: str = "gdb", ld_preload: str = None, ld_library_path: str = None
    ):
        self.gdb_path = gdb_path
        self.ld_preload = ld_preload
        self.ld_library_path = ld_library_path
        self._process: Optional[subprocess.Popen] = None
        self._core_loaded = False
        self._core_path = None
        self._executable_path = None
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()

    def _start_gdb(self):
        """Start the persistent GDB subprocess if not already running."""
        if self._process is not None and self._process.poll() is None:
            return

        # Set up environment with LD_PRELOAD for Python and LD_LIBRARY_PATH for ROCm
        env = os.environ.copy()
        if self.ld_preload:
            env["LD_PRELOAD"] = self.ld_preload
        if self.ld_library_path:
            existing = env.get("LD_LIBRARY_PATH", "")
            if existing:
                env["LD_LIBRARY_PATH"] = f"{self.ld_library_path}:{existing}"
            else:
                env["LD_LIBRARY_PATH"] = self.ld_library_path

        # Start GDB in quiet mode
        cmd = [self.gdb_path, "-q", "-nx"]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )

        # Reset the stop event
        self._stop_reader.clear()

        # Start background thread to read output
        self._reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self._reader_thread.start()

        # Wait for initial (gdb) prompt
        self._wait_for_prompt(timeout=30.0)

    def _reader_worker(self):
        """Background thread that reads from GDB stdout character by character."""
        import io

        buffer = ""
        try:
            while not self._stop_reader.is_set():
                if self._process is None or self._process.poll() is not None:
                    break
                try:
                    # Read one character at a time to handle prompts without newlines
                    char = self._process.stdout.read(1)
                    if not char:
                        if self._process.poll() is not None:
                            break
                        continue
                    buffer += char

                    # When we see a newline or a complete prompt, emit the line
                    if char == "\n" or buffer.endswith("(gdb) "):
                        self._output_queue.put(buffer)
                        buffer = ""
                except Exception:
                    break
            # Emit any remaining buffer content
            if buffer:
                self._output_queue.put(buffer)
        except Exception:
            pass

    def _wait_for_prompt(self, timeout: float = 60.0) -> str:
        """Wait for (gdb) prompt and return accumulated output."""
        lines = []
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Use short polling interval for responsiveness
                line = self._output_queue.get(timeout=0.05)
                lines.append(line)

                # Check if this line contains the (gdb) prompt
                if "(gdb)" in line:
                    # Drain any immediately available output
                    while True:
                        try:
                            extra = self._output_queue.get_nowait()
                            lines.append(extra)
                        except queue.Empty:
                            break
                    break
            except queue.Empty:
                continue

        return "".join(lines)

    def _wait_for_marker(self, timeout: float = 60.0) -> str:
        """Wait for our end marker and return accumulated output."""
        lines = []
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Use short polling interval for responsiveness
                line = self._output_queue.get(timeout=0.05)
                lines.append(line)

                # Check if this line contains our end marker
                if self.END_MARKER in line:
                    # Also wait briefly for the (gdb) prompt that follows
                    try:
                        prompt_deadline = time.time() + 1.0
                        while time.time() < prompt_deadline:
                            extra = self._output_queue.get(timeout=0.05)
                            lines.append(extra)
                            if "(gdb)" in extra:
                                break
                    except queue.Empty:
                        pass
                    break
            except queue.Empty:
                continue

        return "".join(lines)

    def _run_python_script(self, script: str, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Run a Python script inside the persistent GDB process.

        The script should print a JSON object with the key RESULT_JSON:.
        """
        if self._process is None or self._process.poll() is not None:
            self._start_gdb()

        # Wrap script to print our end marker when done
        wrapped_script = f"""
{script}
print("{self.END_MARKER}")
"""

        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_script)
            script_path = f.name

        try:
            # Clear any pending output
            while True:
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    break

            # Send source command
            self._process.stdin.write(f"source {script_path}\n")
            self._process.stdin.flush()

            # Wait for the end marker
            output = self._wait_for_marker(timeout)

            # Parse output for JSON result
            json_match = re.search(r"RESULT_JSON:(.+?)(?:\n|$)", output, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"JSON decode error: {e}",
                        "raw_output": output,
                    }

            return {"success": True, "raw_output": output}

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _send_command(self, command: str, timeout: float = 60.0) -> str:
        """Send a raw GDB command and return output."""
        if self._process is None or self._process.poll() is not None:
            self._start_gdb()

        # Clear pending output
        while True:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

        # Send command
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

        # Wait for prompt
        return self._wait_for_prompt(timeout)

    def load_core(
        self, core_path: str, executable_path: Optional[str] = None
    ) -> DebuggerResult:
        """Load a core file into the persistent GDB process."""
        self._start_gdb()

        self._core_path = core_path
        self._executable_path = executable_path

        # Load executable if specified
        if executable_path:
            self._send_command(f"file {executable_path}")

        # Load the core file
        self._send_command(f"core-file {core_path}")
        self._core_loaded = True

        # Get thread info to verify core loaded
        script = """
import gdb
import json

result = {"success": True, "threads": [], "error": ""}

try:
    output = gdb.execute("info threads", to_string=True)
    result["raw_output"] = output
    result["thread_count"] = len(gdb.selected_inferior().threads())
except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            raw_output=data.get("raw_output", ""),
            extra_data={"thread_count": data.get("thread_count", 0)},
        )

    def get_all_threads(self) -> DebuggerResult:
        """Get list of all threads (CPU + GPU in flat view).

        In GDB/ROCgdb, all threads (both CPU and GPU) are visible in a single
        flat list. This is different from LLDB which has separate CPU and GPU targets.
        """
        script = """
import gdb
import json

result = {"success": True, "threads": [], "error": ""}

try:
    gdb.execute("info threads", to_string=True)
    
    inferior = gdb.selected_inferior()
    for thread in inferior.threads():
        thread_info = {
            "id": thread.global_num,
            "name": thread.name or f"Thread {thread.global_num}",
            "ptid": str(thread.ptid),
            "is_stopped": thread.is_stopped(),
            "is_running": thread.is_running(),
        }
        
        if thread.is_stopped():
            try:
                thread.switch()
                frame = gdb.selected_frame()
                thread_info["pc"] = frame.pc()
                thread_info["function"] = frame.name() or "<unknown>"
            except:
                pass
        
        result["threads"].append(thread_info)
        
except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        threads = []
        for t in data.get("threads", []):
            frames = []
            if "pc" in t:
                frames.append(
                    FrameInfo(
                        index=0,
                        pc=t.get("pc", 0),
                        function=t.get("function", "<unknown>"),
                    )
                )

            threads.append(ThreadInfo(id=t["id"], name=t.get("name"), frames=frames))

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            threads=threads,
            raw_output=data.get("raw_output", ""),
        )

    def select_thread(self, thread_id: int) -> DebuggerResult:
        """Select a thread by ID. State persists across calls."""
        script = f"""
import gdb
import json

result = {{"success": True, "error": ""}}

try:
    gdb.execute("thread {thread_id}")
    thread = gdb.selected_thread()
    result["selected_thread"] = thread.global_num
except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        return DebuggerResult(
            success=data.get("success", False), error_message=data.get("error", "")
        )

    def get_backtrace(self, thread_id: Optional[int] = None) -> DebuggerResult:
        """Get backtrace for current or specified thread."""
        thread_select = f'gdb.execute("thread {thread_id}")' if thread_id else ""

        script = f"""
import gdb
import json

result = {{"success": True, "frames": [], "error": ""}}

try:
    {thread_select}
    
    frame = gdb.newest_frame()
    frame_idx = 0
    
    while frame is not None:
        frame_info = {{
            "index": frame_idx,
            "pc": frame.pc(),
            "function": frame.name() or "<unknown>",
        }}
        
        try:
            sal = frame.find_sal()
            if sal.symtab:
                frame_info["file"] = sal.symtab.filename
                frame_info["line"] = sal.line
        except:
            pass
        
        result["frames"].append(frame_info)
        frame = frame.older()
        frame_idx += 1
        
        if frame_idx > 100:
            break

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        frames = []
        for f in data.get("frames", []):
            frames.append(
                FrameInfo(
                    index=f["index"],
                    pc=f["pc"],
                    function=f["function"],
                    file=f.get("file"),
                    line=f.get("line"),
                )
            )

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            backtrace=frames,
        )

    def get_registers(
        self, register_names: Optional[List[str]] = None
    ) -> DebuggerResult:
        """Get register values for current frame."""
        if register_names:
            names_str = json.dumps(register_names)
            filter_code = f"names_to_get = {names_str}"
        else:
            filter_code = "names_to_get = None"

        script = f"""
import gdb
import json

result = {{"success": True, "registers": {{}}, "error": ""}}

try:
    {filter_code}
    
    frame = gdb.selected_frame()
    arch = frame.architecture()
    
    for reg in arch.registers():
        if names_to_get is not None and reg.name not in names_to_get:
            continue
        
        try:
            val = frame.read_register(reg)
            try:
                int_val = int(val)
                result["registers"][reg.name] = {{
                    "name": reg.name,
                    "value": int_val
                }}
            except:
                result["registers"][reg.name] = {{
                    "name": reg.name,
                    "value_str": str(val)
                }}
        except:
            pass

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        registers = {}
        for name, reg_data in data.get("registers", {}).items():
            if "value" in reg_data:
                registers[name] = RegisterValue(name=name, value=reg_data["value"])

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            registers=registers,
        )

    def get_local_variables(self) -> DebuggerResult:
        """Get local variables in current frame."""
        script = """
import gdb
import json

result = {"success": True, "variables": [], "error": ""}

try:
    frame = gdb.selected_frame()
    block = frame.block()
    
    while block is not None:
        for symbol in block:
            if symbol.is_variable or symbol.is_argument:
                try:
                    val = frame.read_var(symbol)
                    result["variables"].append({
                        "name": symbol.name,
                        "value": str(val),
                        "type": str(val.type)
                    })
                except:
                    pass
        
        if block.function:
            break
        block = block.superblock

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        variables = []
        for v in data.get("variables", []):
            variables.append(
                VariableValue(name=v["name"], value=v["value"], type_name=v["type"])
            )

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            variables=variables,
        )

    def evaluate_expression(self, expression: str) -> DebuggerResult:
        """Evaluate an expression."""
        escaped_expr = expression.replace('"', '\\"')

        script = f"""
import gdb
import json

result = {{"success": True, "value": None, "error": ""}}

try:
    val = gdb.parse_and_eval("{escaped_expr}")
    result["value"] = str(val)
    result["type"] = str(val.type)
    
    try:
        result["int_value"] = int(val)
    except:
        pass

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            raw_output=data.get("value", ""),
            extra_data={"type": data.get("type"), "int_value": data.get("int_value")},
        )

    def get_modules(self) -> DebuggerResult:
        """Get list of loaded modules."""
        script = """
import gdb
import json

result = {"success": True, "modules": [], "error": ""}

try:
    output = gdb.execute("info sharedlibrary", to_string=True)
    result["raw_output"] = output
    
    for objfile in gdb.objfiles():
        module = {
            "name": objfile.filename or "<unknown>",
            "path": objfile.filename or "",
        }
        result["modules"].append(module)

except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        modules = []
        for m in data.get("modules", []):
            modules.append(
                ModuleInfo(
                    name=os.path.basename(m["name"]),
                    path=m["path"],
                    load_address=m.get("load_address", 0),
                )
            )

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            modules=modules,
            raw_output=data.get("raw_output", ""),
        )

    def select_frame(self, frame_index: int) -> DebuggerResult:
        """Select a frame by index."""
        script = f"""
import gdb
import json

result = {{"success": True, "error": ""}}

try:
    gdb.execute("frame {frame_index}")
    frame = gdb.selected_frame()
    result["pc"] = frame.pc()
    result["function"] = frame.name() or "<unknown>"
except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            extra_data={"pc": data.get("pc"), "function": data.get("function")},
        )

    def execute_command(self, command: str) -> DebuggerResult:
        """Execute a raw debugger command and return output."""
        escaped_cmd = command.replace('"', '\\"').replace("\n", "\\n")

        script = f"""
import gdb
import json

result = {{"success": True, "output": "", "error": ""}}

try:
    output = gdb.execute("{escaped_cmd}", to_string=True)
    result["output"] = output
except Exception as e:
    result["success"] = False
    result["error"] = str(e)

print("RESULT_JSON:" + json.dumps(result))
"""
        data = self._run_python_script(script)

        return DebuggerResult(
            success=data.get("success", False),
            error_message=data.get("error", ""),
            raw_output=data.get("output", ""),
        )

    def cleanup(self):
        """Clean up resources - terminate the persistent GDB process."""
        self._stop_reader.set()

        if self._process is not None:
            try:
                self._process.stdin.write("quit\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

        # Clear the queue
        while True:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

        self._core_loaded = False
        self._core_path = None
        self._executable_path = None
