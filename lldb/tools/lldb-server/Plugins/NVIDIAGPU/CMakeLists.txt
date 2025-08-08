add_lldb_library(lldbServerPluginNVIDIAGPU
  CUDADebuggerAPI.cpp
  DeviceState.cpp
  DeviceState-Decoding.cpp
  LLDBServerPluginNVIDIAGPU.cpp
  MainLoopEventNotifier.cpp
  NVIDIAGPU.cpp
  RegisterContextNVIDIAGPU.cpp
  ThreadNVIDIAGPU.cpp
)

# Include shared CUDA setup
include(SetupCUDA)

# Verify NVCC is available (required for test compilation)
lldb_verify_nvcc_available()

target_include_directories(lldbServerPluginNVIDIAGPU PRIVATE
  "${LLDB_SOURCE_DIR}/source" 
  "../.."
)

# We need to expose these headers so that lldb-gdbserver.cpp can build, as it imports
# LLDBServerPluginNVIDIAGPU.h directly. Once the plugins mature, we should be able to
# move to a well-defined interface that wouldn't require exposing these headers.
lldb_add_cuda_include_dirs(lldbServerPluginNVIDIAGPU PUBLIC)

# Pass CUDA environment variables as preprocessor definitions
lldb_apply_cuda_env_definitions(lldbServerPluginNVIDIAGPU)

# Link against CUDA libraries if needed
target_link_libraries(lldbServerPluginNVIDIAGPU PRIVATE ${CUDA_LIBRARIES})
