import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestAddressSpaceMemoryWrite(GDBRemoteTestBase):
    """
    Test that a write to a non-default address space goes out as the standard
    "M" memory packet with the "address_space:<hex-id>;" suffix after its data,
    the suffix that reads carry on "m" and "x". The server advertises
    "address-spaces+" and "address-space-writes+" in qSupported and reports the
    spaces via "jAddressSpacesInfo".
    """

    # In hex, as qSupported reports it. Small, so that one write needs more
    # than one packet.
    PACKET_SIZE = 0x100

    def connect_to_server(
        self, address_space_writes=True, address_space_write_error=None
    ):
        """Connect to a server with address spaces. Without
        `address_space_writes`, the server does not advertise
        "address-space-writes+". With `address_space_write_error`, the server
        fails every write to a non-default address space with that reason, as
        one does that can only read from its address spaces. Return the process
        and the list that the server adds each "M" packet to."""
        features = "address-spaces+"
        if address_space_writes:
            features += ";address-space-writes+"
        address_spaces_json = (
            '[{"name":"global","space_id":1,"is_thread_specific":false},'
            '{"name":"local","space_id":26,"is_thread_specific":true}]'
        )
        packet_size = self.PACKET_SIZE

        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                super().__init__()
                self.writes = []

            def qSupported(self, client_supported):
                return f"PacketSize={packet_size:x};QStartNoAckMode+;{features}"

            def qHostInfo(self):
                return "ptrsize:8;endian:little;"

            def _respond_impl(self, packet):
                # The base dispatcher can't parse the "address_space:<hex-id>;"
                # suffix, so handle every write here.
                if packet and packet[0] == "M":
                    self.writes.append(packet)
                    if address_space_write_error and "address_space:" in packet:
                        return "E01;" + address_space_write_error.encode().hex()
                    return "OK"
                return super()._respond_impl(packet)

            def other(self, packet):
                if packet == "jAddressSpacesInfo":
                    return escape_binary(address_spaces_json)
                return ""

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        # "memory write" needs a stopped process.
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )
        return process, self.server.responder.writes

    def test_write(self):
        process, writes = self.connect_to_server()

        # Same numeric address, two spaces (global == id 1, local == id 26).
        # Ids go out in base 16, like the other numeric values in the packet,
        # and only the thread specific space carries a "thread:" field, which
        # names the selected thread.
        # The values are single bytes, so that the data does not depend on
        # the byte order.
        thread = process.GetSelectedThread()
        self.assertTrue(thread.IsValid())
        self.runCmd("memory write --space global 0x1000 0xaa 0xbb 0xcc 0xdd --size 1")
        self.runCmd("memory write --space local 0x1000 0x11 0x22 0x33 0x44 --size 1")
        self.assertEqual(
            writes,
            [
                "M1000,4:aabbccdd;address_space:1;",
                "M1000,4:11223344;address_space:1a;thread:%x;" % thread.GetThreadID(),
            ],
        )

        # A write to the default address space carries no suffix.
        del writes[:]
        self.runCmd("memory write 0x1000 0x55 0x66 0x77 0x88 --size 1")
        self.assertEqual(writes, ["M1000,4:55667788"])

        # An unknown address space is an error, not a write.
        self.expect(
            "memory write --space nonexistent 0x1000 0 --size 1",
            error=True,
            substrs=["invalid address space"],
        )
        self.assertEqual(writes, ["M1000,4:55667788"])

        # A write too large for one packet is split. Every piece carries the
        # suffix and fits in a packet with it; "$" and "#NN" frame the packet.
        del writes[:]
        text = "0123456789abcdef" * 8
        self.runCmd(f'memory write --space local --format s 0x2000 "{text}"')
        self.assertGreater(len(writes), 1)
        suffix = ";address_space:1a;thread:%x;" % thread.GetThreadID()
        address = 0x2000
        written = b""
        for packet in writes:
            self.assertLessEqual(len("$") + len(packet) + len("#NN"), self.PACKET_SIZE)
            self.assertTrue(packet.endswith(suffix), packet)
            location, _, data = packet[1 : -len(suffix)].partition(":")
            self.assertEqual(
                location, "%x,%x" % (address, len(data) // 2), "no gap or overlap"
            )
            written += bytes.fromhex(data)
            address += len(data) // 2
        self.assertEqual(written, text.encode() + b"\0")

    def test_server_without_address_space_writes(self):
        """A server that advertises "address-spaces+" alone, as one built
        before "M" took the suffix does, would ignore the suffix and write to
        the default address space. lldb sends it no such write."""
        process, writes = self.connect_to_server(address_space_writes=False)

        self.expect(
            "memory write --space global 0x1000 0xaa --size 1",
            error=True,
            substrs=["the remote does not support writing to address spaces"],
        )
        self.assertEqual(writes, [])

        # Writes to the default address space still work.
        self.runCmd("memory write 0x1000 0xaa --size 1")
        self.assertEqual(writes, ["M1000,1:aa"])

    def test_server_that_only_reads_address_spaces(self):
        """A server whose address spaces are read-only fails a write to one,
        and its reason reaches the user. The reason here is the one that
        lldb-server gives for a process that does not implement such writes."""
        reason = "writing to non-default address spaces is not supported"
        process, writes = self.connect_to_server(address_space_write_error=reason)

        self.expect(
            "memory write --space global 0x1000 0xaa --size 1",
            error=True,
            substrs=[reason],
        )
        self.assertEqual(writes, ["M1000,1:aa;address_space:1;"])

        # Writes to the default address space still work.
        del writes[:]
        self.runCmd("memory write 0x1000 0xaa --size 1")
        self.assertEqual(writes, ["M1000,1:aa"])
