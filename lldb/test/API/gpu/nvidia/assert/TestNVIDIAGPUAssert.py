import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from subprocess import run
import threading
import os


class ProcessListener(threading.Thread):
    """A thread that listens for process events from any processes."""

    def __init__(self, dbg, traceOn, event_timeout=100):
        """
        Args:
          dbg: The debugger instance.
          traceOn: Whether to trace the listener.
          event_timeout: The timeout in second for waiting for an event. If the timeout is hit, an exception is raised.
        """
        super().__init__()
        self.listener = dbg.GetListener()
        self.traceOn = traceOn
        self.on_stop_callbacks = {}
        self.event_timeout = event_timeout

    def onStop(self, pid, cb):
        """Register a callback to be called when the process with the given PID stops.

        The callback should return True if the listener should continue processing events,
        or False if the listener should stop processing events.
        """
        self.on_stop_callbacks[pid] = cb

    def run(self):
        while True:
            if self.traceOn:
                print("Try wait for event...")

            event = lldb.SBEvent()
            if not self.listener.WaitForEvent(self.event_timeout, event):
                raise Exception(
                    "[ProcessStateListener] Timeout occurred waiting for event..."
                )

            if event.GetBroadcasterClass() != lldb.SBProcess.GetBroadcasterClass():
                continue

            if not (event.GetType() & lldb.SBProcess.eBroadcastBitStateChanged):
                continue

            proc = lldb.SBProcess.GetProcessFromEvent(event)

            if self.traceOn:
                print("Event description:", lldbutil.get_description(event))
                print("Process state:", lldbutil.state_type_to_str(proc.GetState()))

            if lldb.SBProcess.GetStateFromEvent(event) == lldb.eStateStopped:
                # We want all stop events to be processed by some callback
                if not self.on_stop_callbacks[proc.GetProcessID()](proc):
                    break
        if self.traceOn:
            print("ProcessListener thread exiting")


class TestNVIDIAGPUAssert(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_gpu_asserting(self):
        """Test that we know when the GPU has asserted."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        target.BreakpointCreateByName("main", "a.out")

        process = target.LaunchSimple(None, None, os.getcwd())
        traceOn = self.TraceOn()
        # We don't want to wait for the targets to get finish get interrupted before disposing them at tearDown,
        # so we set the interrupt timeout to 0.
        # We hit this codepath because we are not using sync mode.
        self.runCmd("settings set target.process.interrupt-timeout 0")
        self.dbg.SetAsync(True)

        process_listener = ProcessListener(self.dbg, traceOn)

        def onCPUStop(_proc):
            # We continue processing on every CPU stop
            return True

        def onGPUStop(proc):
            description = str(proc.thread[0])
            if traceOn:
                print("GPU stop reason:", description)
            if "NVIDIA GPU Thread Stopped by Exception" in description:
                # We stop processing events
                if traceOn:
                    print("will stop processing events")
                return False
            if traceOn:
                print("will continue processing events")
            return True

        process_listener.onStop(process.GetProcessID(), onCPUStop)
        process_listener.onStop(1, onGPUStop)

        process_listener.start()
        process.Continue()
        process_listener.join()
