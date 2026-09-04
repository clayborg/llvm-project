from lldbsuite.test.lldbtest import TestBase
import lldb

# Triple substrings that identify GPU targets.
GPU_TRIPLE_PATTERNS = ("amdgcn", "r600", "nvptx", "mockgpu", "spirv")

# Triple substrings that identify CPU targets.
CPU_TRIPLE_PATTERNS = ("x86_64", "aarch64", "arm")


def find_target_by_triple(debugger, patterns):
    """Find the first target whose triple contains one of the given patterns."""
    for i in range(debugger.GetNumTargets()):
        target = debugger.GetTargetAtIndex(i)
        triple = target.GetTriple()
        if any(p in triple for p in patterns):
            return target
    return None


class GpuTestCaseBase(TestBase):
    """
    Class that should be used by all GPU tests.
    """
    NO_DEBUG_INFO_TESTCASE = True

    def _find_target_by_triple(self, patterns):
        """Find the first target whose triple contains one of the given patterns."""
        return find_target_by_triple(self.dbg, patterns)

    @property
    def cpu_target(self):
        """Return the CPU target by searching for a CPU triple."""
        return self._find_target_by_triple(CPU_TRIPLE_PATTERNS)

    @property
    def gpu_target(self):
        """Return the GPU target by searching for a GPU triple."""
        return self._find_target_by_triple(GPU_TRIPLE_PATTERNS)

    @property
    def cpu_process(self):
        """Return the CPU process."""
        target = self.cpu_target
        return target.GetProcess() if target else None

    @property
    def gpu_process(self):
        """Return the GPU process."""
        target = self.gpu_target
        return target.GetProcess() if target else None

    def select_cpu(self):
        """Select the CPU target."""
        self.dbg.SetSelectedTarget(self.cpu_target)

    def select_gpu(self):
        """Select the GPU target."""
        self.dbg.SetSelectedTarget(self.gpu_target)
