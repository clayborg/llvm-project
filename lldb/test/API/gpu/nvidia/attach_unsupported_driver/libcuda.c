// Stands in for a libcuda that lacks the symbols of the driver's safe attach
// procedure, such as cudbgInitiateDebuggerAttachProcedureFd.
int libcuda_stand_in(void) { return 0; }
