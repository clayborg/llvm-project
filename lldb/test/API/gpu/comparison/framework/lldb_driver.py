"""
LLDB Driver - Uses the in-process LLDB Python API for automation.

This driver uses the native LLDB Python bindings (lldb module) instead of
launching LLDB as a subprocess. This is more efficient and provides
direct access to LLDB's SB API.
"""

import os

import lldb
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
from lldbsuite.test.tools.gpu.gpu_testcase import (
    CPU_TRIPLE_PATTERNS,
    GPU_TRIPLE_PATTERNS,
    find_target_by_triple,
)


class LldbDriver(DebuggerInterface):
    """
    LLDB driver that uses the in-process Python API (SB API) to interact
    with the debugger directly.

    When used with the LLDB test framework, pass the debugger instance
    from self.dbg. When used standalone, a new debugger instance is created.
    """

    def __init__(self, debugger: lldb.SBDebugger = None):
        """
        Initialize the LLDB driver.

        Args:
            debugger: An existing SBDebugger instance (e.g., self.dbg from test).
                     If None, a new debugger instance is created.
        """
        if debugger is not None:
            self._debugger = debugger
            self._owns_debugger = False
        else:
            # Create a new debugger instance for standalone use
            self._debugger = lldb.SBDebugger.Create()
            self._debugger.SetAsync(False)
            self._owns_debugger = True

        self._target = None
        self._process = None
        self._core_path = None

    @property
    def debugger(self) -> lldb.SBDebugger:
        """Get the underlying SBDebugger instance."""
        return self._debugger

    @property
    def target(self) -> Optional[lldb.SBTarget]:
        """Get the current target."""
        return self._target

    @property
    def process(self) -> Optional[lldb.SBProcess]:
        """Get the current process."""
        return self._process

    def set_target(self, target: lldb.SBTarget):
        """Update the cached target and process."""
        self._target = target
        self._process = target.GetProcess() if target and target.IsValid() else None

    def _get_selected_frame(self) -> Optional[lldb.SBFrame]:
        if self._process.GetNumThreads() == 0:
            return None

        thread = self._process.GetSelectedThread()
        if not thread.IsValid():
            thread = self._process.GetThreadAtIndex(0)
            self._process.SetSelectedThread(thread)

        frame = thread.GetSelectedFrame()
        if frame.IsValid():
            return frame

        if thread.GetNumFrames() > 0:
            thread.SetSelectedFrame(0)
            return thread.GetFrameAtIndex(0)

        return None

    def _select_target_by_triple(self, patterns, target_name) -> DebuggerResult:
        target = find_target_by_triple(self._debugger, patterns)
        if not target or not target.IsValid():
            return DebuggerResult(
                success=False, error_message=f"LLDB {target_name} target not found"
            )

        self._debugger.SetSelectedTarget(target)
        self.set_target(target)
        if not self._process or not self._process.IsValid():
            return DebuggerResult(
                success=False, error_message=f"LLDB {target_name} process not found"
            )

        self._get_selected_frame()
        return DebuggerResult(
            success=True,
            extra_data={"target": target},
        )

    def select_cpu(self) -> DebuggerResult:
        """Select the CPU target."""
        return self._select_target_by_triple(CPU_TRIPLE_PATTERNS, "CPU")

    def select_gpu(self) -> DebuggerResult:
        """Select the GPU target."""
        return self._select_target_by_triple(GPU_TRIPLE_PATTERNS, "GPU")

    def get_thread_count(self) -> int:
        """Get the thread count for the selected target/process."""
        return self._process.GetNumThreads()

    def load_core(
        self,
        core_path: str,
        executable_path: Optional[str] = None,
        auto_load_debuginfo: bool = False,
    ) -> DebuggerResult:
        """Load a core file using the in-process LLDB API."""
        if auto_load_debuginfo:
            return self._load_core_auto_debuginfo(core_path)

        self._core_path = core_path

        try:
            # Create target (similar to TestAmdGpuCoreFile.py pattern)
            if executable_path:
                self._target = self._debugger.CreateTarget(executable_path)
            else:
                self._target = self._debugger.CreateTarget(None)

            if not self._target.IsValid():
                return DebuggerResult(
                    success=False, error_message="Failed to create target"
                )

            # Load the core file
            error = lldb.SBError()
            self._process = self._target.LoadCore(core_path, error)

            if not self._process.IsValid():
                return DebuggerResult(
                    success=False,
                    error_message=f"Failed to load core: {error.GetCString()}",
                )

            # Collect target information
            targets_info = []
            for i in range(self._debugger.GetNumTargets()):
                target = self._debugger.GetTargetAtIndex(i)
                target_info = {
                    "index": i,
                    "triple": target.GetTriple(),
                    "num_modules": target.GetNumModules(),
                }

                process = target.GetProcess()
                if process.IsValid():
                    target_info["num_threads"] = process.GetNumThreads()
                    target_info["process_id"] = process.GetProcessID()
                    target_info["state"] = str(process.GetState())

                targets_info.append(target_info)

            return DebuggerResult(
                success=True,
                extra_data={
                    "thread_count": self._process.GetNumThreads(),
                    "target_triple": self._target.GetTriple(),
                    "targets": targets_info,
                },
            )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def _load_core_auto_debuginfo(self, core_path: str) -> DebuggerResult:
        self._core_path = os.path.realpath(core_path)
        output = []

        fbcode_path = os.environ.get("GPU_COMPARISON_FBCODE_PATH")
        if fbcode_path:
            result = self.execute_command(
                f"script import sys; sys.path.insert(0, {fbcode_path!r})"
            )
            output.append(result.raw_output or result.error_message)

        result = self.execute_command(
            "script import os; os.environ['LD_LIBRARY_PATH'] = "
            "os.environ.get('LLDB_SYMBOL_STORAGE_LD_LIBRARY_PATH', '')"
        )
        output.append(result.raw_output or result.error_message)

        result = self.execute_command("command script import fblldb")
        output.append(result.raw_output or result.error_message)

        result = self.execute_command(f"auto-load-debuginfo {self._core_path}")
        output.append(result.raw_output or result.error_message)
        if not result.success:
            return DebuggerResult(
                success=False,
                error_message=result.error_message,
                raw_output=result.raw_output,
                extra_data={"auto_load_output": "\n".join(output)},
            )

        self._target = self._debugger.GetSelectedTarget()
        if not self._target or not self._target.IsValid():
            return DebuggerResult(
                success=False,
                error_message="auto-load-debuginfo did not create a valid target",
                extra_data={"auto_load_output": "\n".join(output)},
            )

        self._process = self._target.GetProcess()
        if not self._process or not self._process.IsValid():
            return DebuggerResult(
                success=False,
                error_message="auto-load-debuginfo did not create a valid process",
                extra_data={"auto_load_output": "\n".join(output)},
            )

        return DebuggerResult(
            success=True,
            extra_data={
                "thread_count": self._process.GetNumThreads(),
                "target_triple": self._target.GetTriple(),
                "auto_load_output": "\n".join(output),
            },
        )

    def get_all_threads(self) -> DebuggerResult:
        """Get list of all threads from all targets (CPU + GPU).

        Note: In LLDB, threads are organized by target. This method iterates
        through all targets to provide a unified view similar to GDB's flat view.
        """
        threads = []
        all_targets_info = []

        try:
            # Iterate through all targets (CPU and GPU)
            for target_idx in range(self._debugger.GetNumTargets()):
                target = self._debugger.GetTargetAtIndex(target_idx)
                process = target.GetProcess()

                target_info = {
                    "target_index": target_idx,
                    "triple": target.GetTriple(),
                    "threads": [],
                }

                if process.IsValid():
                    for i in range(process.GetNumThreads()):
                        thread = process.GetThreadAtIndex(i)

                        # Get top frame info
                        frames = []
                        frame = thread.GetFrameAtIndex(0)
                        if frame.IsValid():
                            pc = frame.GetPC()
                            func = frame.GetFunction()
                            if func.IsValid():
                                function_name = func.GetName()
                            else:
                                symbol = frame.GetSymbol()
                                if symbol.IsValid():
                                    function_name = symbol.GetName()
                                else:
                                    function_name = "<unknown>"

                            frames.append(
                                FrameInfo(index=0, pc=pc, function=function_name)
                            )

                        thread_info = ThreadInfo(
                            id=thread.GetThreadID(),
                            name=thread.GetName() or f"Thread {thread.GetIndexID()}",
                            frames=frames,
                        )

                        threads.append(thread_info)
                        target_info["threads"].append(
                            {
                                "id": thread.GetThreadID(),
                                "index_id": thread.GetIndexID(),
                                "name": thread.GetName()
                                or f"Thread {thread.GetIndexID()}",
                            }
                        )

                all_targets_info.append(target_info)

            return DebuggerResult(
                success=True,
                threads=threads,
                extra_data={"all_targets": all_targets_info},
            )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def select_thread(self, thread_id: int) -> DebuggerResult:
        """Select a thread by ID."""
        try:
            if not self._process or not self._process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            # Search in all targets
            for target_idx in range(self._debugger.GetNumTargets()):
                target = self._debugger.GetTargetAtIndex(target_idx)
                process = target.GetProcess()

                if process.IsValid():
                    for i in range(process.GetNumThreads()):
                        thread = process.GetThreadAtIndex(i)
                        if (
                            thread.GetThreadID() == thread_id
                            or thread.GetIndexID() == thread_id
                        ):
                            process.SetSelectedThread(thread)
                            # Update current target/process reference
                            self._target = target
                            self._process = process
                            return DebuggerResult(
                                success=True,
                                extra_data={"selected_thread": thread.GetThreadID()},
                            )

            return DebuggerResult(
                success=False, error_message=f"Thread {thread_id} not found"
            )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def get_backtrace(self, thread_id: Optional[int] = None) -> DebuggerResult:
        """Get backtrace for current or specified thread."""
        try:
            process = self._process

            # If thread_id specified, find and select it
            if thread_id is not None:
                for target_idx in range(self._debugger.GetNumTargets()):
                    target = self._debugger.GetTargetAtIndex(target_idx)
                    proc = target.GetProcess()
                    if proc.IsValid():
                        for i in range(proc.GetNumThreads()):
                            thread = proc.GetThreadAtIndex(i)
                            if (
                                thread.GetThreadID() == thread_id
                                or thread.GetIndexID() == thread_id
                            ):
                                proc.SetSelectedThread(thread)
                                process = proc
                                break

            if not process or not process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            thread = process.GetSelectedThread()
            if not thread.IsValid():
                return DebuggerResult(
                    success=False, error_message="No valid thread selected"
                )

            frames = []
            for i in range(min(thread.GetNumFrames(), 100)):  # Limit depth
                frame = thread.GetFrameAtIndex(i)

                # Get function name
                func = frame.GetFunction()
                if func.IsValid():
                    function_name = func.GetName()
                else:
                    symbol = frame.GetSymbol()
                    if symbol.IsValid():
                        function_name = symbol.GetName()
                    else:
                        function_name = "<unknown>"

                # Get source info
                file_name = None
                line_num = None
                line_entry = frame.GetLineEntry()
                if line_entry.IsValid():
                    file_spec = line_entry.GetFileSpec()
                    if file_spec.IsValid():
                        file_name = file_spec.GetFilename()
                        line_num = line_entry.GetLine()

                # Get module info
                module_name = None
                module = frame.GetModule()
                if module.IsValid():
                    module_name = module.GetFileSpec().GetFilename()

                frames.append(
                    FrameInfo(
                        index=i,
                        pc=frame.GetPC(),
                        function=function_name,
                        file=file_name,
                        line=line_num,
                        module=module_name,
                    )
                )

            return DebuggerResult(success=True, backtrace=frames)

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def get_registers(
        self, register_names: Optional[List[str]] = None
    ) -> DebuggerResult:
        """Get register values for current frame."""
        try:
            if not self._process or not self._process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            frame = self._get_selected_frame()
            if frame is None or not frame.IsValid():
                return DebuggerResult(
                    success=False, error_message="No valid frame selected"
                )

            return self.get_registers_from_frame(frame, register_names)

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def get_registers_from_frame(
        self,
        frame: lldb.SBFrame,
        register_names: Optional[List[str]] = None,
    ) -> DebuggerResult:
        """Get register values from a specific LLDB frame."""
        try:
            registers = {}
            reg_sets = frame.GetRegisters()

            for reg_set in reg_sets:
                for reg in reg_set:
                    name = reg.GetName()

                    if register_names is not None and name not in register_names:
                        continue

                    error = reg.GetError()
                    if error.Fail():
                        continue

                    # Vector registers do not have a scalar value string here;
                    # GetValueAsUnsigned() can still return a misleading zero.
                    if reg.GetValue() is None:
                        continue

                    byte_size = reg.GetByteSize()
                    try:
                        value = reg.GetValueAsUnsigned()
                    except:
                        continue

                    registers[name] = RegisterValue(
                        name=name,
                        value=value,
                        size=byte_size,
                    )

            return DebuggerResult(success=True, registers=registers)

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def get_local_variables(self) -> DebuggerResult:
        """Get local variables in current frame."""
        try:
            if not self._process or not self._process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            frame = self._get_selected_frame()
            if frame is None or not frame.IsValid():
                return DebuggerResult(
                    success=False, error_message="No valid frame selected"
                )

            # Get locals and arguments
            variables = []
            vars_list = frame.GetVariables(True, True, False, True)

            for var in vars_list:
                variables.append(
                    VariableValue(
                        name=var.GetName(),
                        value=var.GetValue() or str(var),
                        type_name=var.GetTypeName(),
                    )
                )

            return DebuggerResult(success=True, variables=variables)

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def evaluate_expression(self, expression: str) -> DebuggerResult:
        """Evaluate an expression."""
        try:
            if not self._process or not self._process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            frame = self._get_selected_frame()
            if frame is None or not frame.IsValid():
                return DebuggerResult(
                    success=False, error_message="No valid frame selected"
                )

            val = frame.EvaluateExpression(expression)

            if val.GetError().Success():
                extra = {"type": val.GetTypeName()}
                try:
                    extra["int_value"] = val.GetValueAsUnsigned()
                except:
                    pass

                return DebuggerResult(
                    success=True,
                    raw_output=val.GetValue() or str(val),
                    extra_data=extra,
                )
            else:
                return DebuggerResult(
                    success=False, error_message=val.GetError().GetCString()
                )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def get_modules(self) -> DebuggerResult:
        """Get list of loaded modules."""
        try:
            if not self._target or not self._target.IsValid():
                return DebuggerResult(success=False, error_message="No valid target")

            modules = []
            for i in range(self._target.GetNumModules()):
                module = self._target.GetModuleAtIndex(i)
                file_spec = module.GetFileSpec()

                uuid = module.GetUUIDString()
                modules.append(
                    ModuleInfo(
                        name=file_spec.GetFilename() or "<unknown>",
                        path=str(file_spec),
                        load_address=0,  # Would need to iterate sections
                        uuid=uuid if uuid else None,
                    )
                )

            return DebuggerResult(success=True, modules=modules)

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def select_frame(self, frame_index: int) -> DebuggerResult:
        """Select a frame by index."""
        try:
            if not self._process or not self._process.IsValid():
                return DebuggerResult(success=False, error_message="No valid process")

            thread = self._process.GetSelectedThread()
            if not thread.IsValid():
                return DebuggerResult(
                    success=False, error_message="No valid thread selected"
                )

            frame = thread.GetFrameAtIndex(frame_index)
            if frame.IsValid():
                thread.SetSelectedFrame(frame_index)

                func = frame.GetFunction()
                if func.IsValid():
                    function_name = func.GetName()
                else:
                    symbol = frame.GetSymbol()
                    if symbol.IsValid():
                        function_name = symbol.GetName()
                    else:
                        function_name = "<unknown>"

                return DebuggerResult(
                    success=True,
                    extra_data={"pc": frame.GetPC(), "function": function_name},
                )
            else:
                return DebuggerResult(
                    success=False, error_message=f"Frame {frame_index} not valid"
                )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def execute_command(self, command: str) -> DebuggerResult:
        """Execute a raw debugger command and return output."""
        try:
            ci = self._debugger.GetCommandInterpreter()
            ret = lldb.SBCommandReturnObject()
            ci.HandleCommand(command, ret)

            if ret.Succeeded():
                return DebuggerResult(success=True, raw_output=ret.GetOutput() or "")
            else:
                return DebuggerResult(
                    success=False,
                    error_message=ret.GetError() or "Command failed",
                    raw_output=ret.GetOutput() or "",
                )

        except Exception as e:
            return DebuggerResult(success=False, error_message=str(e))

    def cleanup(self):
        """Clean up resources."""
        self._target = None
        self._process = None
        self._core_path = None

        # Only destroy the debugger if we created it
        if self._owns_debugger and self._debugger:
            lldb.SBDebugger.Destroy(self._debugger)
            self._debugger = None
