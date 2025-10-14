add_lldb_library(lldbServerPluginNVGPU
  CUDADebuggerAPI.cpp
  DeviceState.cpp
  DeviceState-Decoding.cpp
  LLDBServerPluginNVGPU.cpp
  MainLoopEventNotifier.cpp
  ProcessNVGPU.cpp
  RegisterContextNVGPU.cpp
  ThreadNVGPU.cpp
)

# Include shared CUDA setup
include(SetupCUDA)

# Verify NVCC is available (required for test compilation)
if (LLDB_INCLUDE_TESTS)
  lldb_verify_nvcc_available()
endif()


target_include_directories(lldbServerPluginNVGPU PRIVATE
  "${LLDB_SOURCE_DIR}/source" 
  "../.."
)

# We need to expose these headers so that lldb-gdbserver.cpp can build, as it imports
# LLDBServerPluginNVGPU.h directly. Once the plugins mature, we should be able to
# move to a well-defined interface that wouldn't require exposing these headers.
lldb_add_cuda_include_dirs(lldbServerPluginNVGPU PUBLIC)

# Pass CUDA environment variables as preprocessor definitions
lldb_apply_cuda_env_definitions(lldbServerPluginNVGPU)

# Link against CUDA libraries if needed
target_link_libraries(lldbServerPluginNVGPU PRIVATE ${CUDA_LIBRARIES})
