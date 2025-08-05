from lldbsuite.test.lldbtest import TestBase
import lldb


class GpuTestCaseBase(TestBase):
    """
    Class that should be used by all GPU tests.
    """
    NO_DEBUG_INFO_TESTCASE = True

    @property
    def cpu_target(self):
        """Return the CPU target. Assumes the CPU target is the first target."""
        return self.dbg.GetTargetAtIndex(0)

    @property
    def gpu_target(self):
        """Return the GPU target. Assumes the GPU target is the second target."""
        return self.dbg.GetTargetAtIndex(1)

    @property
    def cpu_process(self):
        """Return the GPU target. Assumes the GPU target is the second target."""
        return self.cpu_target.GetProcess()

    @property
    def gpu_process(self):
        """Return the GPU target. Assumes the GPU target is the second target."""
        return self.gpu_target.GetProcess()

    def select_cpu(self):
        """Select the CPU target."""
        self.dbg.SetSelectedTarget(self.cpu_target)

    def select_gpu(self):
        """Select the GPU target."""
        self.dbg.SetSelectedTarget(self.gpu_target)
