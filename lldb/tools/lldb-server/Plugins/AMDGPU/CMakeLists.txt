# Find AMD-dbgapi package using the provided CMake config
set(amd-dbgapi_DIR "${ROCM_PATH}/lib/cmake/amd-dbgapi" CACHE PATH "Path to amd-dbgapi cmake config")
find_package(amd-dbgapi REQUIRED CONFIG)

include_directories(${ROCM_PATH}/include)

add_lldb_library(lldbServerPluginAMDGPU
  LLDBServerPluginAMDGPU.cpp
  ProcessAMDGPU.cpp
  RegisterContextAMDGPU.cpp
  ThreadAMDGPU.cpp

  LINK_LIBS
    amd-dbgapi
)
set_target_properties(lldbServerPluginAMDGPU PROPERTIES LINK_FLAGS "-Wl,-rpath,${ROCM_PATH}/lib")
target_include_directories(lldbServerPluginAMDGPU PRIVATE "${LLDB_SOURCE_DIR}/source" ${ROCM_PATH}/include "../..")
