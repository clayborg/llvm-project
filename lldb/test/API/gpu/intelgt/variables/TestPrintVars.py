"""
Test per-lane variable printing in IntelGT GPU debugging.

GPU SIMD lanes are modeled as separate threads (lane index in the low 8 bits
of the TID). A local scalar such as `int gid = id[0]` has a distinct value in
each lane. This test verifies that switching between lane-threads yields the
correct per-lane value, and that a uniform variable stays constant.

At -O0 locals are spilled to per-lane scratch memory, so this exercises the
memory-resident variable path (DW_AT_frame_base + DW_OP_fbreg, possibly with a
DW_OP_INTEL_push_simd_lane term). Whether the lane actually reaches the address
computation is exactly what we are verifying here.
"""

import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.tools.gpu.intelgt_testcase import IntelGtTestCaseBase


class IntelGtPrintVarsTestCase(IntelGtTestCaseBase):
    """Verify per-lane local variable values."""

    def _launch_and_collect_lanes(self):
        """Build, launch, stop at the kernel breakpoint, and return the list
        of GPU threads (one per active lane) stopped there."""
        self.build()
        exe = self.getBuildArtifact("print_vars_kernel")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Target should be valid")

        bp = target.BreakpointCreateBySourceRegex(
            "inside-kernel", lldb.SBFileSpec("print_vars_kernel.cpp"))
        self.assertTrue(bp.IsValid(), "Breakpoint should be valid")
        self.assertGreater(bp.GetNumLocations(), 0,
                           "Breakpoint should have a location")

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        env_list = [f"{k}={v}" for k, v in os.environ.items()]
        launch_info.SetEnvironmentEntries(env_list, True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(),
                        f"Launch should succeed: {error.GetCString()}")

        listener = self.dbg.GetListener()
        event = lldb.SBEvent()
        state = process.GetState()
        while state == lldb.eStateRunning:
            if not listener.WaitForEvent(30, event):
                self.fail("Timeout waiting for process to stop")
            state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped, "Host process should stop")

        self.assertEqual(self.dbg.GetNumTargets(), 2,
                         "GPU target should be created")
        gpu_process = self.dbg.GetTargetAtIndex(1).GetProcess()
        self.assertTrue(gpu_process.IsValid(), "GPU process should be valid")

        # Enumerate ALL GPU threads. Only the first lane of an EU thread
        # reports eStopReasonBreakpoint; sibling lanes report other reasons
        # but are still valid lane-threads we can read per-lane values from.
        # Exclude the shadow thread (TID == 1, no EU-thread parent).
        lane_threads = [
            t for t in gpu_process.threads if t.GetThreadID() != 1
        ]
        self.assertGreater(len(lane_threads), 0,
                           "Should have at least one GPU lane")
        return lane_threads

    @staticmethod
    def _lane_of(thread):
        """Extract the SIMD lane index from the thread TID (low 8 bits)."""
        return thread.GetThreadID() & 0xFF

    @staticmethod
    def _read_int(thread, name):
        """Read an int local from frame 0 of *thread*; return None if absent."""
        frame = thread.GetFrameAtIndex(0)
        var = frame.FindVariable(name)
        if not var.IsValid():
            return None
        return var.GetValueAsSigned()

    @staticmethod
    def _read_member(thread, var_name, member_name):
        """Read an int struct member (var.member) from frame 0; None if absent."""
        frame = thread.GetFrameAtIndex(0)
        var = frame.FindVariable(var_name)
        if not var.IsValid():
            return None
        member = var.GetChildMemberWithName(member_name)
        if not member.IsValid():
            return None
        return member.GetValueAsSigned()

    def test_dump_lane_values(self):
        """Diagnostic: print each lane's view of gid/doubled/konst.

        This always passes; it surfaces the raw per-lane data (and the DWARF
        location of `gid`) so we can see what the hardware/compiler actually
        produce before asserting on it.
        """
        lane_threads = self._launch_and_collect_lanes()

        # Dump the DWARF location of gid via the SB API (works for locals,
        # unlike `image lookup`). Shows whether a lane term is in the location.
        frame0 = lane_threads[0].GetFrameAtIndex(0)
        gid_var = frame0.FindVariable("gid")
        if gid_var.IsValid():
            stream = lldb.SBStream()
            gid_var.GetDescription(stream)
            print(f"gid SBValue description:\n{stream.GetData()}")
            print(f"gid location: {gid_var.GetLocation()}")

        # Group by EU thread (TID with the low 8 lane bits cleared) so we can
        # see how many distinct EU threads actually stopped at the breakpoint.
        by_eu = {}
        for t in lane_threads:
            eu = t.GetThreadID() & ~0xFF
            by_eu.setdefault(eu, []).append(t)
        print(f"total GPU lane-threads: {len(lane_threads)} across "
              f"{len(by_eu)} EU thread(s)")

        for eu in sorted(by_eu):
            print(f"EU thread base=0x{eu:x}: {len(by_eu[eu])} lanes")
            for t in sorted(by_eu[eu], key=self._lane_of):
                lane = self._lane_of(t)
                gid = self._read_int(t, "gid")
                doubled = self._read_int(t, "doubled")
                konst = self._read_int(t, "konst")
                print(f"  lane {lane:2d} (tid=0x{t.GetThreadID():x}): "
                      f"gid={gid} doubled={doubled} konst={konst} "
                      f"stop={t.GetStopReason()}")

    def test_per_lane_values_differ(self):
        """Each lane sees its own gid; doubled == 2*gid; konst is uniform.

        Keyed by global TID (not lane index) so lanes from different EU threads
        do not collide; with all EU threads presented, every lane's gid must be
        distinct.
        """
        lane_threads = self._launch_and_collect_lanes()

        gid_by_tid = {}
        for t in lane_threads:
            gid = self._read_int(t, "gid")
            doubled = self._read_int(t, "doubled")
            konst = self._read_int(t, "konst")
            tid = t.GetThreadID()

            self.assertIsNotNone(gid, f"tid 0x{tid:x}: gid should be readable")
            self.assertIsNotNone(doubled,
                                 f"tid 0x{tid:x}: doubled should be readable")
            self.assertIsNotNone(konst,
                                 f"tid 0x{tid:x}: konst should be readable")

            gid_by_tid[tid] = gid
            self.assertEqual(doubled, 2 * gid,
                             f"tid 0x{tid:x}: doubled ({doubled}) should be "
                             f"2*gid ({2 * gid})")
            self.assertEqual(konst, 7,
                             f"tid 0x{tid:x}: konst should be 7 (uniform)")

        # The core hypothesis: per-lane values are actually distinct.
        distinct = set(gid_by_tid.values())
        self.assertEqual(
            len(distinct), len(gid_by_tid),
            f"each lane should see a distinct gid, got {sorted(distinct)}")

    def test_gid_tracks_lane_index(self):
        """gid increases 1:1 with the lane index WITHIN each EU thread.

        The work-item id is base + lane, where the base is per-EU-thread (the
        lane index is NOT the global id). With all EU threads of the workgroup
        presented, there is one base per EU thread, so we assert gid - lane is
        constant within each EU thread (grouped by EU-thread TID), not globally.
        """
        lane_threads = self._launch_and_collect_lanes()
        base_by_eu = {}
        for t in lane_threads:
            lane = self._lane_of(t)
            gid = self._read_int(t, "gid")
            self.assertIsNotNone(gid, f"lane {lane}: gid should be readable")
            eu = t.GetThreadID() & ~0xFF
            base = gid - lane
            if eu in base_by_eu:
                self.assertEqual(
                    base_by_eu[eu], base,
                    f"EU 0x{eu:x}: gid-lane should be constant within an EU "
                    f"thread (saw {base_by_eu[eu]} and {base})")
            else:
                base_by_eu[eu] = base
        self.assertGreater(len(base_by_eu), 0, "should have at least one EU")

    def test_per_lane_accessor_read_differs(self):
        """A value read from an accessor into a local (from_acc = in_acc[gid])
        is per-lane and equals that lane's gid.

        Exercises an accessor read resolved into a per-lane scratch local --
        a different DWARF path than a plain scalar literal. Modelled on GDB's
        simd-locations `ain[gid]` comparison.
        """
        lane_threads = self._launch_and_collect_lanes()
        for t in lane_threads:
            tid = t.GetThreadID()
            gid = self._read_int(t, "gid")
            from_acc = self._read_int(t, "from_acc")
            self.assertIsNotNone(gid, f"tid 0x{tid:x}: gid should be readable")
            self.assertIsNotNone(
                from_acc, f"tid 0x{tid:x}: from_acc should be readable")
            self.assertEqual(
                from_acc, gid,
                f"tid 0x{tid:x}: from_acc ({from_acc}) should equal "
                f"gid ({gid}) -- in_acc[gid] == gid by construction")

    def test_per_lane_struct_members_differ(self):
        """Struct member values (p.a, p.b) are per-lane and correctly related.

        p.a == gid and p.b == gid + 100, so each lane sees distinct member
        values. Exercises member-offset resolution on a per-lane struct local
        (GDB simd-locations reads out_s.a / .c the same way).
        """
        lane_threads = self._launch_and_collect_lanes()
        a_by_tid = {}
        for t in lane_threads:
            tid = t.GetThreadID()
            gid = self._read_int(t, "gid")
            pa = self._read_member(t, "p", "a")
            pb = self._read_member(t, "p", "b")
            self.assertIsNotNone(gid, f"tid 0x{tid:x}: gid should be readable")
            self.assertIsNotNone(pa, f"tid 0x{tid:x}: p.a should be readable")
            self.assertIsNotNone(pb, f"tid 0x{tid:x}: p.b should be readable")
            self.assertEqual(pa, gid,
                             f"tid 0x{tid:x}: p.a ({pa}) should equal gid ({gid})")
            self.assertEqual(pb, gid + 100,
                             f"tid 0x{tid:x}: p.b ({pb}) should equal gid+100 "
                             f"({gid + 100})")
            a_by_tid[tid] = pa
        distinct = set(a_by_tid.values())
        self.assertEqual(
            len(distinct), len(a_by_tid),
            f"each lane should see a distinct p.a, got {sorted(distinct)}")

    def test_per_lane_reference_resolves(self):
        """A reference local (int &ref = gid) resolves per lane.

        OBSERVED (verified): LLDB renders the reference as a pointer-like value
        (the address of the referent) rather than auto-dereferencing to gid's
        value -- GDB shows it as `(int &) @addr: <value>`. We therefore assert
        the reference RESOLVES and that dereferencing it yields the lane's gid,
        rather than asserting on the raw rendered value. This pins the current
        behaviour so an improvement (auto-deref) or regression is noticed.
        """
        lane_threads = self._launch_and_collect_lanes()
        for t in lane_threads:
            tid = t.GetThreadID()
            gid = self._read_int(t, "gid")
            frame = t.GetFrameAtIndex(0)
            ref = frame.FindVariable("ref")
            self.assertTrue(ref.IsValid(),
                            f"tid 0x{tid:x}: reference 'ref' should resolve")
            # Dereference the reference and read the pointee's value.
            deref = ref.Dereference()
            deref_val = (deref.GetValueAsSigned()
                         if deref.IsValid() else None)
            self.assertEqual(
                deref_val, gid,
                f"tid 0x{tid:x}: *ref ({deref_val}) should equal gid ({gid})")
