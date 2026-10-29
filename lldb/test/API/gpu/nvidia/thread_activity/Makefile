NVCC_SOURCES := activity.cu
LD_EXTRAS := -lcuda

include Makefile.rules

all: loaded.fatbin

loaded.fatbin: loaded.cu
	$(NVCC) $(NVCCFLAGS) -fatbin -o $@ $<
