
add_lldb_library(lldbServerPluginNVIDIAGPU
  CUDADebuggerAPI.cpp
  LLDBServerPluginNVIDIAGPU.cpp
  MainLoopEventNotifier.cpp
  NVIDIAGPU.cpp
  RegisterContextNVIDIAGPU.cpp
  ThreadNVIDIAGPU.cpp
  Utils.cpp
)

target_include_directories(lldbServerPluginNVIDIAGPU PRIVATE "${LLDB_SOURCE_DIR}/source" "../..")
