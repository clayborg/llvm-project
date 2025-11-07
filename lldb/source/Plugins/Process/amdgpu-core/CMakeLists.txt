# Find AMD-dbgapi package using the provided CMake config
set(amd-dbgapi_DIR "${ROCM_PATH}/lib/cmake/amd-dbgapi" CACHE PATH "Path to amd-dbgapi cmake config")
find_package(amd-dbgapi REQUIRED CONFIG)

include_directories(${ROCM_PATH}/include)

add_lldb_library(lldbPluginProcessAmdGpuCore PLUGIN
  ProcessAmdGpuCore.cpp
  ThreadAMDGPU.cpp
  RegisterContextAmdGpu.cpp

  LINK_COMPONENTS
    BinaryFormat
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbPluginDynamicLoaderProcessModuleList
    lldbPluginObjectFileELF
    lldbPluginProcessUtility
    lldbPluginProcessElfCore
    amd-dbgapi
  )

set_target_properties(lldbPluginProcessAmdGpuCore PROPERTIES LINK_FLAGS "-Wl,-rpath,${ROCM_PATH}/lib")
target_include_directories(lldbPluginProcessAmdGpuCore PRIVATE "${LLDB_SOURCE_DIR}/source" ${ROCM_PATH}/include "../..")
