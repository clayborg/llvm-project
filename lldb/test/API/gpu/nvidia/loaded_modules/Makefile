NVCC_SOURCES := main.cu
LD_EXTRAS := -lcuda

include Makefile.rules

all: fatbin_kernel.fatbin ptx_kernel.ptx

fatbin_kernel.fatbin: fatbin_kernel.cu
	$(NVCC) $(NVCCFLAGS) -fatbin -o $@ $<

# -ptx takes one architecture. Without -arch, nvcc uses its default, and the
# driver can compile PTX for any GPU of that architecture or newer. `override`
# keeps an architecture list on make's command line out of this rule.
ptx_kernel.ptx: override NVCC_ARCHFLAGS :=
ptx_kernel.ptx: ptx_kernel.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<
