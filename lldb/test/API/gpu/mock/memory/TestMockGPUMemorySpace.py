"""
Test Mock GPU memory space reading functionality via CLI and API.

Tests both the CLI command "memory read --space <space>" and the API using SBAddressSpec.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.tools.gpu.gpu_testcase import GpuTestCaseBase

GPU_INITIALIZE_BREAKPOINT = "// CPU BREAKPOINT"
SOURCE_FILE = "main.c"


class TestMockGPUMemorySpace(GpuTestCaseBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        """Test Mock GPU memory space reading"""
        super().setUp()
        self.build()
        self.source_spec = lldb.SBFileSpec(SOURCE_FILE, False)
        lldbutil.run_to_source_breakpoint(
            self, GPU_INITIALIZE_BREAKPOINT, self.source_spec
        )
        self.assertEqual(self.dbg.GetNumTargets(), 2, "There are two targets")
        self.assertTrue(self.cpu_process.IsValid(), "CPU process is valid")
        self.assertTrue(self.gpu_process.IsValid(), "GPU process is valid")
        self.mock_gpu_thread = self.gpu_process.GetThreadAtIndex(0)
        self.assertTrue(self.mock_gpu_thread.IsValid(), "Thread should be valid")

    def test_global_memory_space_read_succeeds(self):
        """Test reading from Global memory space returns expected data."""
        addr_spec = lldb.SBAddressSpec(0x1000, "Global")
        error = lldb.SBError()
        bytes_data = self.gpu_process.ReadMemoryFromSpec(addr_spec, 16, error)
        self.assertTrue(
            error.Success(),
            f"Global space API read should succeed: {error.GetCString()}",
        )
        self.assertEqual(
            bytes_data, b"1111111111111111", "Global space should return '1' bytes"
        )

    def test_thread_memory_space_fails_without_thread_context(self):
        """Test Thread memory space access fails when no thread is specified."""
        addr_spec = lldb.SBAddressSpec(0x1000, "Thread")
        error = lldb.SBError()
        self.gpu_process.ReadMemoryFromSpec(addr_spec, 16, error)
        self.assertFalse(error.Success(), "Thread space without thread should fail")
        self.assertEqual(
            'reading from address space "Thread" requires a thread', error.GetCString()
        )

    def test_thread_memory_space_read_succeeds_with_valid_thread(self):
        """Test reading from Thread memory space works when thread context is provided."""
        addr_spec = lldb.SBAddressSpec(0x1000, "Thread", self.mock_gpu_thread)
        error = lldb.SBError()
        bytes_data = self.gpu_process.ReadMemoryFromSpec(addr_spec, 16, error)
        self.assertTrue(
            error.Success(),
            f"Thread space with thread should succeed: {error.GetCString()}",
        )
        self.assertEqual(
            bytes_data, b"2222222222222222", "Thread space should return '2' bytes"
        )

    def test_invalid_memory_space_name_is_rejected(self):
        """Test that invalid memory space names are properly rejected."""
        addr_spec = lldb.SBAddressSpec(0x1000, "Invalid")
        error = lldb.SBError()
        self.gpu_process.ReadMemoryFromSpec(addr_spec, 16, error)
        self.assertFalse(error.Success(), "Invalid address space should fail")
        self.assertIn('invalid address space "Invalid"', error.GetCString())

    def test_cpu_process_rejects_gpu_memory_space_operations(self):
        """Test that CPU processes reject GPU memory space API calls."""
        addr_spec = lldb.SBAddressSpec(0x1000, "Global")
        error = lldb.SBError()
        self.cpu_process.ReadMemoryFromSpec(addr_spec, 16, error)
        self.assertFalse(
            error.Success(), "CPU process should reject address space API calls"
        )
        self.assertIn("process doesn't support address spaces", error.GetCString())
