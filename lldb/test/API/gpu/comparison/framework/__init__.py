# GPU Debugger Comparison Framework
#
# This is a shared framework for comparing LLDB with vendor-specific debuggers
# (e.g., ROCgdb for AMD, cuda-gdb for NVIDIA) on GPU debugging scenarios.
#
# Directory structure:
#   gpu/comparison/
#   ├── framework/          # Shared comparison framework
#   │   ├── debugger_interface.py  # Abstract interface and data classes
#   │   ├── gdb_driver.py          # Generic GDB driver (can be extended)
#   │   ├── lldb_driver.py         # LLDB driver
#   │   └── comparator.py          # Result comparison utilities
#   ├── amd/                # AMD-specific tests
#   │   └── TestAmdGpuCoreFileComparison.py
#   └── nvidia/             # Future: NVIDIA-specific tests
#

from .debugger_interface import DebuggerInterface, DebuggerResult
from .gdb_driver import GdbDriver
from .lldb_driver import LldbDriver
from .comparator import ResultComparator

__all__ = [
    "DebuggerInterface",
    "DebuggerResult",
    "GdbDriver",
    "LldbDriver",
    "ResultComparator",
]
