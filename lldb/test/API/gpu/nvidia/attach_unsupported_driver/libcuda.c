// Stands in for a libcuda that predates the driver's safe attach procedure: it
// has the debugger handshake globals, but not
// cudbgInitiateDebuggerAttachProcedureFd.
unsigned cudbgApiClientPid;
unsigned cudbgApiClientRevision;
unsigned cudbgSessionId;
unsigned cudbgDebuggerCapabilities;

int libcuda_stand_in(void) { return 0; }
