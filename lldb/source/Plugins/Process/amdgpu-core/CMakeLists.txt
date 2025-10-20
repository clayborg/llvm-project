# Find AMD-dbgapi package using the provided CMake config
set(amd-dbgapi_DIR "${ROCM_PATH}/lib/cmake/amd-dbgapi" CACHE PATH "Path to amd-dbgapi cmake config")
# set(amd-dbgapi_DIR "/home/jeffreytan/ROCdbgapi/debugInstall/lib64/cmake/amd-dbgapi" CACHE PATH "Path to amd-dbgapi cmake config")
find_package(amd-dbgapi REQUIRED CONFIG)

include_directories(${ROCM_PATH}/include)

add_lldb_library(lldbPluginProcessAmdGpuCore PLUGIN
  ProcessAmdGpuCore.cpp
  ProcessElfGpuCore.cpp

  LINK_COMPONENTS
    BinaryFormat
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbPluginDynamicLoaderPosixDYLD
    lldbPluginObjectFileELF
    lldbPluginProcessUtility
    lldbPluginProcessElfCore
    amd-dbgapi
  )

set_target_properties(lldbPluginProcessAmdGpuCore PROPERTIES LINK_FLAGS "-Wl,-rpath,/home/jeffreytan/ROCdbgapi/debugInstall/lib64/:${ROCM_PATH}/lib")
target_include_directories(lldbPluginProcessAmdGpuCore PRIVATE "${LLDB_SOURCE_DIR}/source" ${ROCM_PATH}/include "../..")
