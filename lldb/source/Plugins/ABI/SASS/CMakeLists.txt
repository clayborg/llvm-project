# Include shared CUDA setup
include(SetupCUDA)

add_lldb_library(lldbPluginABISASS PLUGIN
  ABISASS.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
    lldbValueObject
  )

# Include the CUDA debugger headers
lldb_add_cuda_include_dirs(lldbPluginABISASS PRIVATE)

# Include path for SASSRegisterNumbers.h
target_include_directories(lldbPluginABISASS PRIVATE ${LLDB_SOURCE_DIR})
