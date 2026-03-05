"""
Abstract interface and common data structures for debugger comparison testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class StopReason(Enum):
    """Reason why the process stopped."""

    NONE = "none"
    BREAKPOINT = "breakpoint"
    SIGNAL = "signal"
    EXCEPTION = "exception"
    EXEC = "exec"
    EXITED = "exited"
    UNKNOWN = "unknown"


@dataclass
class RegisterValue:
    """Represents a register and its value."""

    name: str
    value: int
    size: int = 0

    def __eq__(self, other):
        if not isinstance(other, RegisterValue):
            return False
        return self.name == other.name and self.value == other.value


@dataclass
class VariableValue:
    """Represents a variable and its value."""

    name: str
    value: str
    type_name: str

    def __eq__(self, other):
        if not isinstance(other, VariableValue):
            return False
        return (
            self.name == other.name
            and self.value == other.value
            and self.type_name == other.type_name
        )


@dataclass
class FrameInfo:
    """Represents a stack frame."""

    index: int
    pc: int
    function: str
    file: Optional[str] = None
    line: Optional[int] = None
    module: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, FrameInfo):
            return False
        # Compare key fields, allowing for minor differences
        return (
            self.index == other.index
            and self.pc == other.pc
            and self.function == other.function
        )


@dataclass
class ThreadInfo:
    """Represents a thread."""

    id: int
    name: Optional[str] = None
    stop_reason: StopReason = StopReason.NONE
    frames: List[FrameInfo] = field(default_factory=list)

    def __eq__(self, other):
        if not isinstance(other, ThreadInfo):
            return False
        return self.id == other.id and self.name == other.name


@dataclass
class ModuleInfo:
    """Represents a loaded module/shared library."""

    name: str
    path: str
    load_address: int
    uuid: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, ModuleInfo):
            return False
        return self.name == other.name and self.load_address == other.load_address


@dataclass
class DebuggerResult:
    """Container for results from debugger operations."""

    success: bool = True
    error_message: str = ""
    threads: List[ThreadInfo] = field(default_factory=list)
    registers: Dict[str, RegisterValue] = field(default_factory=dict)
    variables: List[VariableValue] = field(default_factory=list)
    modules: List[ModuleInfo] = field(default_factory=list)
    backtrace: List[FrameInfo] = field(default_factory=list)
    raw_output: str = ""
    extra_data: Dict[str, Any] = field(default_factory=dict)


class DebuggerInterface(ABC):
    """
    Abstract interface for debugger operations.
    Both GDB and LLDB drivers implement this interface.
    """

    @abstractmethod
    def load_core(
        self, core_path: str, executable_path: Optional[str] = None
    ) -> DebuggerResult:
        """Load a core file for debugging."""
        pass

    @abstractmethod
    def get_all_threads(self) -> DebuggerResult:
        """Get list of all threads.

        For GDB/ROCgdb, this returns all threads (CPU + GPU) in a flat view.
        For LLDB, this returns threads for the currently selected target only.
        """
        pass

    @abstractmethod
    def select_thread(self, thread_id: int) -> DebuggerResult:
        """Select a thread by ID."""
        pass

    @abstractmethod
    def get_backtrace(self, thread_id: Optional[int] = None) -> DebuggerResult:
        """Get backtrace for current or specified thread."""
        pass

    @abstractmethod
    def get_registers(
        self, register_names: Optional[List[str]] = None
    ) -> DebuggerResult:
        """Get register values for current frame."""
        pass

    @abstractmethod
    def get_local_variables(self) -> DebuggerResult:
        """Get local variables in current frame."""
        pass

    @abstractmethod
    def evaluate_expression(self, expression: str) -> DebuggerResult:
        """Evaluate an expression."""
        pass

    @abstractmethod
    def get_modules(self) -> DebuggerResult:
        """Get list of loaded modules."""
        pass

    @abstractmethod
    def select_frame(self, frame_index: int) -> DebuggerResult:
        """Select a frame by index."""
        pass

    @abstractmethod
    def execute_command(self, command: str) -> DebuggerResult:
        """Execute a raw debugger command and return output."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass
