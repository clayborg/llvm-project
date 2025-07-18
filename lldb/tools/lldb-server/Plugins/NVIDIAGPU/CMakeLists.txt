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

    set(NVIDIAGPU_NVCC_PATH "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE STRING "Path to the NVCC compiler." FORCE)
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

if(NOT NVIDIAGPU_NVCC_PATH)
  message(FATAL_ERROR "NVIDIAGPU_NVCC_PATH not set. Please install the CUDA Toolkit or set a valid NVIDIAGPU_NVCC_PATH CMake variable.")
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

# Pass CUDA environment variables as preprocessor definitions
if(NVIDIAGPU_CUDBG_INJECTION_PATH)
  target_compile_definitions(lldbServerPluginNVIDIAGPU PRIVATE 
    CMAKE_NVIDIAGPU_CUDBG_INJECTION_PATH="${NVIDIAGPU_CUDBG_INJECTION_PATH}")
endif()
if(NVIDIAGPU_CUDA_VISIBLE_DEVICES)
  target_compile_definitions(lldbServerPluginNVIDIAGPU PRIVATE 
    CMAKE_NVIDIAGPU_CUDA_VISIBLE_DEVICES="${NVIDIAGPU_CUDA_VISIBLE_DEVICES}")
endif()
if(NVIDIAGPU_CUDA_DEVICE_ORDER)
  target_compile_definitions(lldbServerPluginNVIDIAGPU PRIVATE 
    CMAKE_NVIDIAGPU_CUDA_DEVICE_ORDER="${NVIDIAGPU_CUDA_DEVICE_ORDER}")
endif()
if(NVIDIAGPU_CUDA_LAUNCH_BLOCKING)
  target_compile_definitions(lldbServerPluginNVIDIAGPU PRIVATE 
    CMAKE_NVIDIAGPU_CUDA_LAUNCH_BLOCKING="${NVIDIAGPU_CUDA_LAUNCH_BLOCKING}")
endif()

# Link against CUDA libraries if needed
target_link_libraries(lldbServerPluginNVIDIAGPU PRIVATE ${CUDA_LIBRARIES})
