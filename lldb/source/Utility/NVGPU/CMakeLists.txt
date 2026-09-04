include(SetupCUDA)

add_lldb_library(lldbUtilityNVGPU
  CUDAAddressSpaces.cpp
  CUDAException.cpp
  SASSRegisterInfo.cpp
  ThreadName.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    ${LLDB_SYSTEM_LIBS}
  )

lldb_add_cuda_include_dirs(lldbUtilityNVGPU PRIVATE)
