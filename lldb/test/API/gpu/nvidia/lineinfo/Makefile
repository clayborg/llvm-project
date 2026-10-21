NVCC_SOURCES := main.cu
# Line tables (-lineinfo) instead of device debug information (-G), so the
# device code is optimized. `override` keeps a NVCC_DEBUGFLAGS on make's command
# line, such as a run configuration's, from changing the level this test is
# about.
override NVCC_DEBUGFLAGS := -lineinfo

include Makefile.rules
