# Find AMD-dbgapi package using the provided CMake config
set(amd-dbgapi_DIR "${ROCM_PATH}/lib/cmake/amd-dbgapi" CACHE PATH "Path to amd-dbgapi cmake config" FORCE)
find_package(amd-dbgapi REQUIRED CONFIG)

include_directories(${ROCM_PATH}/include)

add_lldb_library(lldbServerPluginAMDGPU
  LLDBServerPluginAMDGPU.cpp
  ProcessAMDGPU.cpp
  RegisterContextAmdGpu.cpp
  ThreadAMDGPU.cpp
  WaveAMDGPU.cpp

  LINK_LIBS
    amd-dbgapi
)
set_target_properties(lldbServerPluginAMDGPU PROPERTIES LINK_FLAGS "-Wl,-rpath,${ROCM_PATH}/lib")
target_include_directories(lldbServerPluginAMDGPU PRIVATE "${LLDB_SOURCE_DIR}/source" ${ROCM_PATH}/include "../..")

# Bundle ROCm runtime libraries
file(GLOB _rocm_dbgapi_libs "${ROCM_PATH}/lib/librocm-dbgapi.so*")
file(GLOB _rocm_comgr_libs "${ROCM_PATH}/lib/libamd_comgr.so*")
file(GLOB _rocm_zstd_libs "${ROCM_PATH}/lib/libzstd.so*")
install(FILES ${_rocm_dbgapi_libs} ${_rocm_comgr_libs} ${_rocm_zstd_libs}
  DESTINATION lib
  COMPONENT lldb-server)
