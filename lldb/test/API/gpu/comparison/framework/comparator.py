"""
Result comparator for comparing GDB and LLDB debugging outputs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from .debugger_interface import (
    DebuggerResult,
    ThreadInfo,
    FrameInfo,
    RegisterValue,
    VariableValue,
    ModuleInfo,
)


@dataclass
class ComparisonDifference:
    """Represents a single difference between GDB and LLDB results."""

    category: str  # e.g., "thread", "register", "frame", "variable"
    field: str  # e.g., "name", "value", "pc"
    gdb_value: Any
    lldb_value: Any
    description: str


@dataclass
class ComparisonResult:
    """Result of comparing GDB and LLDB outputs."""

    is_equivalent: bool = True
    differences: List[ComparisonDifference] = field(default_factory=list)
    gdb_only: Dict[str, List[Any]] = field(default_factory=dict)
    lldb_only: Dict[str, List[Any]] = field(default_factory=dict)
    summary: str = ""

    def add_difference(
        self,
        category: str,
        field: str,
        gdb_value: Any,
        lldb_value: Any,
        description: str = "",
    ):
        self.is_equivalent = False
        self.differences.append(
            ComparisonDifference(
                category=category,
                field=field,
                gdb_value=gdb_value,
                lldb_value=lldb_value,
                description=description
                or f"{category}.{field}: GDB={gdb_value}, LLDB={lldb_value}",
            )
        )

    def add_gdb_only(self, category: str, item: Any):
        if category not in self.gdb_only:
            self.gdb_only[category] = []
        self.gdb_only[category].append(item)

    def add_lldb_only(self, category: str, item: Any):
        if category not in self.lldb_only:
            self.lldb_only[category] = []
        self.lldb_only[category].append(item)

    def get_summary(self) -> str:
        if self.is_equivalent:
            return "Results are equivalent"

        lines = ["Results differ:"]

        for diff in self.differences:
            lines.append(f"  - {diff.description}")

        if self.gdb_only:
            for category, items in self.gdb_only.items():
                lines.append(f"  - GDB only {category}: {len(items)} items")

        if self.lldb_only:
            for category, items in self.lldb_only.items():
                lines.append(f"  - LLDB only {category}: {len(items)} items")

        return "\n".join(lines)


class ResultComparator:
    """
    Compares results from GDB and LLDB, normalizing for known differences.
    """

    def __init__(
        self,
        ignore_thread_names: bool = True,
        ignore_thread_ids: bool = True,
        normalize_function_names: bool = True,
        pc_tolerance: int = 0,
    ):
        """
        Initialize comparator with options.

        Args:
            ignore_thread_names: Don't compare thread name strings
            ignore_thread_ids: Don't compare thread numeric IDs (different numbering schemes)
            normalize_function_names: Strip demangling differences
            pc_tolerance: Allow PC values to differ by this amount
        """
        self.ignore_thread_names = ignore_thread_names
        self.ignore_thread_ids = ignore_thread_ids
        self.normalize_function_names = normalize_function_names
        self.pc_tolerance = pc_tolerance

    def compare_threads(
        self, gdb_result: DebuggerResult, lldb_result: DebuggerResult
    ) -> ComparisonResult:
        """Compare thread lists from GDB and LLDB."""
        result = ComparisonResult()

        gdb_threads = gdb_result.threads
        lldb_threads = lldb_result.threads

        # Compare counts
        if len(gdb_threads) != len(lldb_threads):
            result.add_difference(
                "threads",
                "count",
                len(gdb_threads),
                len(lldb_threads),
                f"Thread count differs: GDB={len(gdb_threads)}, LLDB={len(lldb_threads)}",
            )

        # For GPU debugging, thread counts should match
        # We compare by index rather than ID since IDs may differ
        min_count = min(len(gdb_threads), len(lldb_threads))

        for i in range(min_count):
            gdb_thread = gdb_threads[i]
            lldb_thread = lldb_threads[i]

            # Compare top frame if available
            if gdb_thread.frames and lldb_thread.frames:
                gdb_frame = gdb_thread.frames[0]
                lldb_frame = lldb_thread.frames[0]

                # Compare PC
                if abs(gdb_frame.pc - lldb_frame.pc) > self.pc_tolerance:
                    result.add_difference(
                        f"thread[{i}]",
                        "pc",
                        hex(gdb_frame.pc),
                        hex(lldb_frame.pc),
                        f"Thread {i} PC differs",
                    )

                # Compare function name
                if self.normalize_function_names:
                    gdb_func = self._normalize_function_name(gdb_frame.function)
                    lldb_func = self._normalize_function_name(lldb_frame.function)
                else:
                    gdb_func = gdb_frame.function
                    lldb_func = lldb_frame.function

                if gdb_func != lldb_func:
                    result.add_difference(
                        f"thread[{i}]",
                        "function",
                        gdb_func,
                        lldb_func,
                        f"Thread {i} function differs",
                    )

        result.summary = result.get_summary()
        return result

    def compare_backtrace(
        self, gdb_result: DebuggerResult, lldb_result: DebuggerResult
    ) -> ComparisonResult:
        """Compare backtraces from GDB and LLDB."""
        result = ComparisonResult()

        gdb_frames = gdb_result.backtrace
        lldb_frames = lldb_result.backtrace

        # Compare frame counts
        if len(gdb_frames) != len(lldb_frames):
            result.add_difference(
                "backtrace",
                "depth",
                len(gdb_frames),
                len(lldb_frames),
                f"Backtrace depth differs: GDB={len(gdb_frames)}, LLDB={len(lldb_frames)}",
            )

        min_depth = min(len(gdb_frames), len(lldb_frames))

        for i in range(min_depth):
            gdb_frame = gdb_frames[i]
            lldb_frame = lldb_frames[i]

            # Compare PC
            if abs(gdb_frame.pc - lldb_frame.pc) > self.pc_tolerance:
                result.add_difference(
                    f"frame[{i}]",
                    "pc",
                    hex(gdb_frame.pc),
                    hex(lldb_frame.pc),
                    f"Frame {i} PC differs",
                )

            # Compare function name
            if self.normalize_function_names:
                gdb_func = self._normalize_function_name(gdb_frame.function)
                lldb_func = self._normalize_function_name(lldb_frame.function)
            else:
                gdb_func = gdb_frame.function
                lldb_func = lldb_frame.function

            if gdb_func != lldb_func:
                result.add_difference(
                    f"frame[{i}]",
                    "function",
                    gdb_func,
                    lldb_func,
                    f"Frame {i} function differs: GDB='{gdb_func}', LLDB='{lldb_func}'",
                )

        result.summary = result.get_summary()
        return result

    def compare_registers(
        self,
        gdb_result: DebuggerResult,
        lldb_result: DebuggerResult,
        register_names: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """Compare register values from GDB and LLDB."""
        result = ComparisonResult()

        gdb_regs = gdb_result.registers
        lldb_regs = lldb_result.registers

        # Determine which registers to compare
        if register_names:
            names_to_compare = set(register_names)
        else:
            names_to_compare = set(gdb_regs.keys()) | set(lldb_regs.keys())

        for name in sorted(names_to_compare):
            gdb_reg = gdb_regs.get(name)
            lldb_reg = lldb_regs.get(name)

            if gdb_reg is None:
                result.add_lldb_only("registers", name)
                continue

            if lldb_reg is None:
                result.add_gdb_only("registers", name)
                continue

            # Compare values
            if gdb_reg.value != lldb_reg.value:
                result.add_difference(
                    "registers",
                    name,
                    hex(gdb_reg.value),
                    hex(lldb_reg.value),
                    f"Register {name}: GDB={hex(gdb_reg.value)}, LLDB={hex(lldb_reg.value)}",
                )

        result.summary = result.get_summary()
        return result

    def compare_variables(
        self, gdb_result: DebuggerResult, lldb_result: DebuggerResult
    ) -> ComparisonResult:
        """Compare local variables from GDB and LLDB."""
        result = ComparisonResult()

        gdb_vars = {v.name: v for v in gdb_result.variables}
        lldb_vars = {v.name: v for v in lldb_result.variables}

        all_names = set(gdb_vars.keys()) | set(lldb_vars.keys())

        for name in sorted(all_names):
            gdb_var = gdb_vars.get(name)
            lldb_var = lldb_vars.get(name)

            if gdb_var is None:
                result.add_lldb_only("variables", name)
                continue

            if lldb_var is None:
                result.add_gdb_only("variables", name)
                continue

            # Compare values (as strings since formatting may differ)
            if gdb_var.value != lldb_var.value:
                result.add_difference(
                    "variables",
                    name,
                    gdb_var.value,
                    lldb_var.value,
                    f"Variable {name}: values differ",
                )

            # Compare types
            if gdb_var.type_name != lldb_var.type_name:
                result.add_difference(
                    "variables",
                    f"{name}.type",
                    gdb_var.type_name,
                    lldb_var.type_name,
                    f"Variable {name}: types differ",
                )

        result.summary = result.get_summary()
        return result

    def compare_modules(
        self, gdb_result: DebuggerResult, lldb_result: DebuggerResult
    ) -> ComparisonResult:
        """Compare loaded modules from GDB and LLDB."""
        result = ComparisonResult()

        gdb_modules = {m.name: m for m in gdb_result.modules}
        lldb_modules = {m.name: m for m in lldb_result.modules}

        all_names = set(gdb_modules.keys()) | set(lldb_modules.keys())

        for name in sorted(all_names):
            gdb_mod = gdb_modules.get(name)
            lldb_mod = lldb_modules.get(name)

            if gdb_mod is None:
                result.add_lldb_only("modules", name)
                continue

            if lldb_mod is None:
                result.add_gdb_only("modules", name)
                continue

        result.summary = result.get_summary()
        return result

    def _normalize_function_name(self, name: str) -> str:
        """Normalize function name for comparison."""
        if not name:
            return "<unknown>"

        # Strip common prefixes/suffixes that may differ
        name = name.strip()

        # Handle anonymous namespaces differently between GDB/LLDB
        name = name.replace("(anonymous namespace)", "{anonymous}")
        name = name.replace("<anonymous>", "{anonymous}")

        return name

    def normalize_pointer_value(self, value):
        """Normalize pointer value format for comparison.

        GDB shows '0x0' while LLDB shows '0x0000000000000000'.
        Both represent the same value, just different formatting.
        """
        if value is None:
            return None
        # Check if it looks like a hex pointer
        if isinstance(value, str) and value.startswith("0x"):
            try:
                # Parse as int and back to hex to normalize
                int_val = int(value, 16)
                return hex(int_val)
            except ValueError:
                pass
        return value

    def compare_all(
        self, gdb_result: DebuggerResult, lldb_result: DebuggerResult
    ) -> ComparisonResult:
        """Perform all comparisons and return combined result."""
        combined = ComparisonResult()

        # Thread comparison
        if gdb_result.threads or lldb_result.threads:
            thread_result = self.compare_threads(gdb_result, lldb_result)
            combined.differences.extend(thread_result.differences)
            combined.gdb_only.update(thread_result.gdb_only)
            combined.lldb_only.update(thread_result.lldb_only)

        # Backtrace comparison
        if gdb_result.backtrace or lldb_result.backtrace:
            bt_result = self.compare_backtrace(gdb_result, lldb_result)
            combined.differences.extend(bt_result.differences)
            combined.gdb_only.update(bt_result.gdb_only)
            combined.lldb_only.update(bt_result.lldb_only)

        # Register comparison
        if gdb_result.registers or lldb_result.registers:
            reg_result = self.compare_registers(gdb_result, lldb_result)
            combined.differences.extend(reg_result.differences)
            combined.gdb_only.update(reg_result.gdb_only)
            combined.lldb_only.update(reg_result.lldb_only)

        # Variable comparison
        if gdb_result.variables or lldb_result.variables:
            var_result = self.compare_variables(gdb_result, lldb_result)
            combined.differences.extend(var_result.differences)
            combined.gdb_only.update(var_result.gdb_only)
            combined.lldb_only.update(var_result.lldb_only)

        combined.is_equivalent = len(combined.differences) == 0
        combined.summary = combined.get_summary()

        return combined
