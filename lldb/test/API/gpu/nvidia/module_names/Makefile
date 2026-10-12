NVCC_SOURCES := main.cu nodebug.cu

include Makefile.rules

# Without -G, the cubin has no compile unit and is named after its kernel.
nodebug.o: NVCCFLAGS := $(filter-out -G,$(NVCCFLAGS))
