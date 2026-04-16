include(SetupCUDA)

add_lldb_library(lldbPluginProcessNVGPUCore PLUGIN
  NVGPUCoreData.cpp
  ProcessNVGPUCore.cpp
  ThreadNVGPUCore.cpp
  RegisterContextNVGPUCore.cpp

  LINK_COMPONENTS
    BinaryFormat
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
    lldbUtilityNVGPU
    lldbPluginObjectFileELF
  )

target_include_directories(lldbPluginProcessNVGPUCore PRIVATE
  "${LLDB_SOURCE_DIR}/source"
  )

lldb_add_cuda_include_dirs(lldbPluginProcessNVGPUCore PRIVATE)
