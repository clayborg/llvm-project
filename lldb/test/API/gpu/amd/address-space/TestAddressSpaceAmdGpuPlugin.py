"""
Address space tests for the AMDGPU plugin.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *

SOURCE = "aspace.hip"

# The following addresses are offsets from the start of the lane private stack
# for the location where these kernel local variables are stored.  Note that
# these addresses depend on the stack layout by the compiler.  They seem to be
# pretty consistent across rocm versions, but be aware that these private stack
# addresses could change.
SIZE_STACK_OFFSET = 0x20
IDX_STACK_OFFSET = 0x24

# The wave size is the number of lanes in a wave.
WAVE_SIZE = 64


class Location:
    """Helper class to describe a location in memory with an expected value"""

    def __init__(self, name, expected_value, size_in_bytes, address):
        self.name = name
        self.address = address
        if isinstance(expected_value, list):
            self.expected_bytes = b"".join(
                [n.to_bytes(size_in_bytes, "little") for n in expected_value]
            )
            self.size_in_bytes = size_in_bytes * len(expected_value)
        else:
            self.expected_bytes = expected_value.to_bytes(size_in_bytes, "little")
            self.size_in_bytes = size_in_bytes


class AddressSpaceAmdGpuTestCase(AmdGpuTestCaseBase):
    def validate_memory_read(
        self, address_space: str, loc: Location, thread: lldb.SBThread
    ):
        """Helper function to validate memory read from an address space"""
        addr_spec = lldb.SBAddressSpec(loc.address, address_space, thread)
        error = lldb.SBError()
        data = self.gpu_process.ReadMemoryFromSpec(addr_spec, loc.size_in_bytes, error)
        self.assertTrue(
            error.Success(),
            f"{loc.name} reading from address space '{address_space}' failed: {str(error)}",
        )
        self.assertEqual(
            data,
            loc.expected_bytes,
            f"Data for Location {loc.name} does not match expected value",
        )

    def validate_read_address_from_global_variable(
        self, address_space: str, location: Location
    ):
        """Helper function to check we can read from address space using an address from a global variable.
        The global variable lookup is done on the cpu side and its value is added to the current location
        address to allow easy offsetting from a global variable that represents an array.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// CPU BREAKPOINT - BEFORE LAUNCH", lldb.SBFileSpec(SOURCE)
        )

        # Find the variable in the CPU target.
        var = self.cpu_target.FindFirstGlobalVariable(location.name)
        self.assertTrue(
            var.IsValid(),
            f"{location.name} variable should be valid in the CPU target",
        )

        # Get the address stored in the variable (it's a pointer memory).
        addr = var.GetValueAsUnsigned()
        self.assertNotEqual(addr, 0, f"{location.name} address should not be null")
        location.address += addr

        # Continue executing to the gpu breakpoint.
        gpu_threads = self.continue_to_gpu_source_breakpoint(
            SOURCE,
            "// GPU BREAKPOINT",
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")

        # Switch back to GPU to read from generic address space
        self.select_gpu()

        self.validate_memory_read(address_space, location, gpu_threads[0])

    def run_to_first_gpu_breakpoint(self):
        """Helper to run to common gpu breakpoint"""
        self.build()

        gpu_threads = self.run_to_gpu_breakpoint(
            SOURCE, "// GPU BREAKPOINT", "// CPU BREAKPOINT - BEFORE LAUNCH"
        )
        self.assertNotEqual(None, gpu_threads, "GPU should be stopped at breakpoint")
        return gpu_threads

    def test_generic(self):
        """Test reading from the generic address space."""
        # Read from generic address space using the address stored in the device_output pointer.
        # The device_output variable is a pointer to device visible memory.
        # We expect the element at index 1 to be 2 (output[1] = shared_mem[1] = 1 * 2 = 2).
        location = Location(
            "device_output",
            expected_value=2,
            size_in_bytes=4,
            address=4,  # Offset address by 4 bytes to get index 1.
        )
        self.validate_read_address_from_global_variable("generic", location)

    def test_region(self):
        """Test that we fail to read from the region address space. It is not supported on this architecture (MI300/MI350)."""
        self.run_to_first_gpu_breakpoint()

        addr_spec = lldb.SBAddressSpec(0, "region")
        error = lldb.SBError()
        self.gpu_process.ReadMemoryFromSpec(addr_spec, 1, error)

        self.assertFalse(error.Success(), "Read from region address space should fail")
        self.assertEqual(
            "AMD_DBGAPI_STATUS_ERROR: AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY",
            error.GetCString(),
        )

    def test_local(self):
        """Test that we can read from local memory."""
        gpu_threads = self.run_to_first_gpu_breakpoint()

        # Check that we can read local memory locations.
        # We expect the element at index 3 to be 6 (shared_mem[3] = 3 * 2 = 6).
        location = Location("shared_mem", expected_value=6, size_in_bytes=4, address=12)
        self.validate_memory_read("local", location, gpu_threads[0])

    def test_private_lane(self):
        """Test that we can read from the private_lane address space."""
        gpu_threads = self.run_to_first_gpu_breakpoint()

        # Check that we can read private_lane memory Locations.
        # These locations map to local variables on the private stack memory for
        # each lane. The `size` value is the input parameter to the kernel and
        # the `idx` value is the local variable in the kernel that stores the
        # threadIdx.x value. We check the `idx` variable twice since it should
        # have different values for lane 0 and lane 1.
        checks = [
            (
                Location(
                    "size",
                    expected_value=WAVE_SIZE,
                    size_in_bytes=4,
                    address=SIZE_STACK_OFFSET,
                ),
                gpu_threads[0],
            ),
            (
                Location(
                    "idx", expected_value=0, size_in_bytes=4, address=IDX_STACK_OFFSET
                ),
                gpu_threads[0],
            ),
            (
                Location(
                    "idx", expected_value=1, size_in_bytes=4, address=IDX_STACK_OFFSET
                ),
                gpu_threads[1],
            ),
        ]

        for location, thread in checks:
            self.validate_memory_read("private_lane", location, thread)

    def test_private_wave(self):
        """Test that we can read from the private_wave address space."""

        # The private_wave address space has the unswizzled values for each lane.
        # This makes it easy for the debugger to read the value of one variable
        # for each lane as consecutive memory locations.
        #
        # The location below describes the first 3 lane values for the idx variable.
        # The memory offset is calculated finding the offset of the idx variable
        # for a lane and then multiplying by the wave size since the unswizzled
        # memory is laid out with the values for each lane in consecutive memory.
        addr = IDX_STACK_OFFSET * WAVE_SIZE

        gpu_threads = self.run_to_first_gpu_breakpoint()
        location = Location(
            "idx[0:3]", expected_value=[0, 1, 2], size_in_bytes=4, address=addr
        )
        self.validate_memory_read("private_wave", location, gpu_threads[0])
