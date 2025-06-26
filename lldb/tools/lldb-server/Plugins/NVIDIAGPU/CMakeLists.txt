add_lldb_library(lldbServerPluginNVIDIAGPU
  CUDADebuggerAPI.cpp
  LLDBServerPluginNVIDIAGPU.cpp
  MainLoopEventNotifier.cpp
  NVIDIAGPU.cpp
  RegisterContextNVIDIAGPU.cpp
  ThreadNVIDIAGPU.cpp
  Utils.cpp
)

set(NVIDIAGPU_DEBUGGER_INCLUDE_DIR_DESC 
    "Path to the folder containing the CUDA debugger header file (cudadebugger.h)")
set(NVIDIAGPU_DEBUGGER_INCLUDE_DIR CACHE STRING ${NVIDIAGPU_DEBUGGER_INCLUDE_DIR_DESC})

if(NOT NVIDIAGPU_DEBUGGER_INCLUDE_DIR)
  find_package(CUDAToolkit)

  if (CUDAToolkit_FOUND)
    set(NVIDIAGPU_DEBUGGER_INCLUDE_DIR 
        "${CUDAToolkit_LIBRARY_ROOT}/extras/Debugger/include" 
        CACHE STRING ${NVIDIAGPU_DEBUGGER_INCLUDE_DIR_DESC} FORCE)
  else()
    # Default for ubuntu
    set(NVIDIAGPU_DEBUGGER_INCLUDE_DIR 
        "/usr/local/cuda/extras/Debugger/include/cudadebugger.h" 
        CACHE STRING ${NVIDIAGPU_DEBUGGER_INCLUDE_DIR_DESC} FORCE)
  endif()
endif()

set(TROUBLESHOOTING_MESSAGE 
    "Please (re)install the CUDA Toolkit or set a valid NVIDIAGPU_DEBUGGER_INCLUDE_DIR CMake variable.")

if(NOT NVIDIAGPU_DEBUGGER_INCLUDE_DIR) 
  message(FATAL_ERROR "NVIDIAGPU_DEBUGGER_INCLUDE_DIR not set. ${TROUBLESHOOTING_MESSAGE}")
elseif(NOT EXISTS "${NVIDIAGPU_DEBUGGER_INCLUDE_DIR}")
  message(FATAL_ERROR 
      "NVIDIAGPU_DEBUGGER_INCLUDE_DIR (${NVIDIAGPU_DEBUGGER_INCLUDE_DIR}) not found. ${TROUBLESHOOTING_MESSAGE}")
endif()

target_include_directories(lldbServerPluginNVIDIAGPU PRIVATE
  "${LLDB_SOURCE_DIR}/source" 
  "../.."
)

# We need to expose these headers so that lldb-gdbserver.cpp can build, as it imports
# LLDBServerPluginNVIDIAGPU.h directly. Once the plugins mature, we should be able to
# move to a well-defined interface that wouldn't require exposing these headers.
target_include_directories(lldbServerPluginNVIDIAGPU PUBLIC
  ${NVIDIAGPU_DEBUGGER_INCLUDE_DIR}
)

# Link against CUDA libraries if needed
target_link_libraries(lldbServerPluginNVIDIAGPU PRIVATE ${CUDA_LIBRARIES})
