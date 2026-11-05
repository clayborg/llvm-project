NVCC_SOURCES := shadow_breakpoints.cu twins.cu

include Makefile.rules

# The program without the host debug information that nvcc's -g adds. The
# device code keeps its own.
nodebug.out: shadow_breakpoints.cu twins.cu
	$(NVCC) $(NVCC_ARCHFLAGS) $(NVCC_DEBUGFLAGS) -O0 -o $@ $^
