"""
Test that AMDGPU registers with an amd-dbgapi flags type print their fields.
"""

import lldb
from lldbsuite.test.lldbtest import *
from amdgpu_testcase import *


class AmdGpuRegisterFlagsTestCase(AmdGpuTestCaseBase):
    def test_register_fields(self):
        """Test that mode, trapsts and status print decoded fields."""
        self.build()
        self.run_to_gpu_breakpoint("register_flags.hip", "// GPU BREAKPOINT")
        self.select_gpu()
        # Padding fields count towards the child limit (default 24), and these
        # registers have more children than that, so raise it to see the low
        # bits.
        self.runCmd("settings set target.max-children-count 32")

        # The kernel set the rounding modes with s_setreg.
        self.expect(
            "register read mode",
            substrs=["FP_ROUND.64_16 = MINUS_INF", "FP_ROUND.32 = PLUS_INF"],
        )
        # The kernel also wrote TRAPSTS. EXCP.INVALID (bit 0) prints last, so
        # this also checks that the limit was raised. Each pattern matches
        # through the end of the line so "= 5" does not match "= 50".
        self.expect(
            "register read trapsts",
            patterns=[
                r"EXCP_CYCLE = 5\n",
                r"EXCP\.DIV0 = 1\n",
                r"EXCP\.DENORM = 0\n",
                r"EXCP\.INVALID = 1\n",
            ],
            substrs=["DP_RATE = "],
        )
        self.expect("register read status", substrs=["VALID = 1", "SCC = "])
