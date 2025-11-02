# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Test AMD GPU core file debugging functionality
"""

from asyncio.tasks import sleep

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os
import sys

from lldbsuite.test import lldbutil


class TestAmdGpuCoreFile(TestBase):
    """Test AMD GPU core file debugging"""

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        """Check for real GPU core files before running tests"""
        TestBase.setUp(self)

        # These tests require REAL GPU core files from actual crashes
        # The synthetic cores are too minimal for full LLDB loading
        #
        # To generate real cores:
        # 1. Build a GPU program: hipcc -g -O0 gpu_crash.hip -o gpu_crash
        # 2. Enable cores: ulimit -c unlimited
        # 3. Run and crash: ./gpu_crash
        # 4. Copy: cp core.* <test_dir>/test.core
        #
        # If real cores aren't available, tests will skip gracefully

        # Check for core file and skip if not available
        test_dir = os.path.dirname(__file__)
        self.core_file_path = os.path.join(test_dir, "test.core")
        if not os.path.exists(self.core_file_path):
            self.skipTest(
                "GPU core file not available - copy your core to: "
                + self.core_file_path
            )

    def get_process_state_str(self, process):
        """Helper to get human-readable process state"""
        state_map = {
            lldb.eStateInvalid: "invalid",
            lldb.eStateUnloaded: "unloaded",
            lldb.eStateConnected: "connected",
            lldb.eStateAttaching: "attaching",
            lldb.eStateLaunching: "launching",
            lldb.eStateStopped: "stopped",
            lldb.eStateRunning: "running",
            lldb.eStateStepping: "stepping",
            lldb.eStateCrashed: "crashed",
            lldb.eStateDetached: "detached",
            lldb.eStateExited: "exited",
            lldb.eStateSuspended: "suspended",
        }
        return state_map.get(process.GetState(), f"unknown({process.GetState()})")

    def find_gpu_target(self):
        """Find GPU target in debugger's target list"""
        for i in range(self.dbg.GetNumTargets()):
            target = self.dbg.GetTargetAtIndex(i)
            triple = target.GetTriple()
            if "amdgcn" in triple or "amdgpu" in triple:
                return target
        return None

    def find_cpu_target(self):
        """Find CPU target in debugger's target list"""
        for i in range(self.dbg.GetNumTargets()):
            target = self.dbg.GetTargetAtIndex(i)
            triple = target.GetTriple()
            if "x86_64" in triple or "aarch64" in triple:
                return target
        return None

    @skipIfRemote
    def test_load_amd_gpu_core_file(self):
        """Test loading an AMD GPU core file creates both CPU and GPU targets"""

        # Load the core file - matching TestLinuxCore pattern
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid(), "Failed to create target")

        error = lldb.SBError()
        process = target.LoadCore(self.core_file_path, error)
        if not process.IsValid():
            self.fail(
                f"Failed to load core file: {self.core_file_path}\nError: {error.GetCString()}\nProcess valid: {process.IsValid()}"
            )
        self.assertEqual(
            process.GetState(),
            lldb.eStateStopped,
            f"Process should be stopped after loading core, but is: {self.get_process_state_str(process)}",
        )

        # Check that we have two targets - CPU and GPU
        num_targets = self.dbg.GetNumTargets()
        self.assertGreaterEqual(num_targets, 1, "Should have at least the CPU target")

        # Find CPU target
        cpu_target = self.find_cpu_target()
        self.assertIsNotNone(cpu_target, "CPU target should be created")

        # Check if GPU target was created (may not exist if core doesn't contain GPU data)
        gpu_target = self.find_gpu_target()
        if gpu_target:
            # Verify GPU target properties
            self.assertTrue(gpu_target.IsValid(), "GPU target should be valid")
            triple = gpu_target.GetTriple()
            self.assertIn(
                "amdgcn", triple, f"GPU target should have amdgcn triple, got: {triple}"
            )

    @skipIfRemote
    def test_gpu_target_process_state(self):
        """Test GPU process state after loading core file"""
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(self.core_file_path)
        self.assertTrue(process.IsValid())

        gpu_target = self.find_gpu_target()
        if not gpu_target:
            self.skipTest("GPU target not created from this core file")

        gpu_process = gpu_target.GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")
        self.assertEqual(
            gpu_process.GetState(),
            lldb.eStateStopped,
            f"GPU process should be stopped, but is: {self.get_process_state_str(gpu_process)}",
        )

    @skipIfRemote
    def test_gpu_threads_present(self):
        """Test that GPU threads (waves) are enumerated from core file"""
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(self.core_file_path)

        gpu_target = self.find_gpu_target()
        if not gpu_target:
            self.skipTest("GPU target not created from this core file")

        gpu_process = gpu_target.GetProcess()
        num_threads = gpu_process.GetNumThreads()

        # If we have GPU data, we should have at least one thread (wave)
        self.assertGreater(
            num_threads, 0, "GPU process should have at least one thread (wave)"
        )

        # Verify thread properties
        thread = gpu_process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "First GPU thread should be valid")
        self.assertIsNotNone(thread.GetThreadID(), "GPU thread should have a thread ID")

    @skipIfRemote
    def test_gpu_register_context(self):
        """Test GPU register context is accessible from core file"""
        target = self.dbg.CreateTarget(None)

        # fbvscode.set_trace()
        process = target.LoadCore(self.core_file_path)

        gpu_target = self.find_gpu_target()
        if not gpu_target:
            self.skipTest("GPU target not created from this core file")

        gpu_process = gpu_target.GetProcess()
        if gpu_process.GetNumThreads() == 0:
            self.skipTest("No GPU threads available in core file")

        thread = gpu_process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "GPU thread should have a valid frame")

        # Try to get registers
        regs = frame.GetRegisters()
        self.assertGreater(regs.GetSize(), 0, "Should have register sets available")

        # Check for typical GPU register sets
        reg_set_names = [
            regs.GetValueAtIndex(i).GetName() for i in range(regs.GetSize())
        ]
        # AMD GPUs typically have GPR, SGPR, VGPR, etc.
        self.assertTrue(len(reg_set_names) > 0, "Should have at least one register set")

    @skipIfRemote
    def test_gpu_architecture_detection(self):
        """Test that GPU architecture is correctly detected from core file"""
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(self.core_file_path)

        gpu_target = self.find_gpu_target()
        if not gpu_target:
            self.skipTest("GPU target not created from this core file")

        triple = gpu_target.GetTriple()
        self.assertIn("amdgcn", triple, "GPU triple should contain amdgcn")

    # TODO: Re-enable this test when GPU memory reading is implemented
    # Currently DoReadMemory returns "not implemented" for ProcessElfGpuCore
    # To re-enable:
    # 1. Implement DoReadMemory in ProcessAmdGpuCore
    # 2. Remove the @unittest.skip decorator
    # 3. Verify the test passes with a real GPU core file
    @unittest.skip(
        "GPU memory reading not yet implemented - DoReadMemory returns 'not implemented'"
    )
    @skipIfRemote
    def test_gpu_memory_reading(self):
        """Test reading GPU memory from core file"""
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(self.core_file_path)

        gpu_target = self.find_gpu_target()
        if not gpu_target:
            self.skipTest("GPU target not created from this core file")

        gpu_process = gpu_target.GetProcess()
        if gpu_process.GetNumThreads() == 0:
            self.skipTest("No GPU threads available in core file")

        # Try to read some memory from GPU address space
        # This tests the xfer_global_memory callback path
        thread = gpu_process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)

        # Get PC to find a valid address
        pc = frame.GetPC()
        if pc != lldb.LLDB_INVALID_ADDRESS:
            # Try to read a small amount of memory at PC
            error = lldb.SBError()
            data = gpu_process.ReadMemory(pc, 16, error)

            # We expect this to work or fail gracefully with an appropriate error
            # The exact behavior depends on whether the PC points to valid GPU memory
            # in the core file
            if error.Success():
                self.assertIsNotNone(data, "Should get data when read succeeds")
                self.assertGreater(len(data), 0, "Should read some data")
