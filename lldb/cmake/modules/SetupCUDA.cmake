#===----------------------------------------------------------------------===//
#
# Common CUDA setup logic for LLDB plugins that require CUDA support.
# This module finds the CUDA Toolkit and sets up necessary include paths.
#
#===----------------------------------------------------------------------===//

# Include guard to prevent multiple inclusion
if(LLDB_SETUP_CUDA_INCLUDED)
  return()
endif()
set(LLDB_SETUP_CUDA_INCLUDED TRUE)

# Ensure the NVIDIA GPU plugin is explicitly enabled
if(NOT LLDB_ENABLE_NVGPU_PLUGIN)
  message(FATAL_ERROR "Attempting to use CUDA setup without enabling NVIDIA GPU support. Please set LLDB_ENABLE_NVGPU_PLUGIN=ON in your CMake configuration.")
endif()

# Set up CUDA debugger include directory
set(NVGPU_DEBUGGER_INCLUDE_DIR_DESC 
    "Path to the folder containing the CUDA debugger header file (cudadebugger.h)")
set(NVGPU_DEBUGGER_INCLUDE_DIR CACHE STRING ${NVGPU_DEBUGGER_INCLUDE_DIR_DESC})

# Try to find CUDA Toolkit if include dir not explicitly set
if(NOT NVGPU_DEBUGGER_INCLUDE_DIR)
  find_package(CUDAToolkit)

  if (CUDAToolkit_FOUND)
    set(NVGPU_DEBUGGER_INCLUDE_DIR 
        "${CUDAToolkit_LIBRARY_ROOT}/extras/Debugger/include" 
        CACHE STRING ${NVGPU_DEBUGGER_INCLUDE_DIR_DESC} FORCE)
    
    # Set NVCC path if not already set (useful for some plugins)
    if(NOT NVGPU_NVCC_PATH)
      set(NVGPU_NVCC_PATH "${CUDAToolkit_NVCC_EXECUTABLE}" 
          CACHE STRING "Path to the NVCC compiler." FORCE)
    endif()
  endif()
endif()

# Error handling for missing CUDA debugger headers
set(TROUBLESHOOTING_MESSAGE 
    "Please (re)install the CUDA Toolkit or set a valid NVGPU_DEBUGGER_INCLUDE_DIR CMake variable.")

if(NOT NVGPU_DEBUGGER_INCLUDE_DIR) 
  message(FATAL_ERROR "NVGPU_DEBUGGER_INCLUDE_DIR not set. ${TROUBLESHOOTING_MESSAGE}")
elseif(NOT EXISTS "${NVGPU_DEBUGGER_INCLUDE_DIR}")
  message(FATAL_ERROR 
      "NVGPU_DEBUGGER_INCLUDE_DIR (${NVGPU_DEBUGGER_INCLUDE_DIR}) not found. ${TROUBLESHOOTING_MESSAGE}")
endif()

# Function to verify NVCC compiler is available
# Some CUDA plugins require the NVCC compiler for runtime compilation
function(lldb_verify_nvcc_available)
  if(NOT NVGPU_NVCC_PATH)
    message(FATAL_ERROR "NVGPU_NVCC_PATH not set. Please install the CUDA Toolkit or set a valid NVGPU_NVCC_PATH CMake variable.")
  endif()
endfunction()

# Function to add CUDA debugger include directories to a target
function(lldb_add_cuda_include_dirs target_name)
  # Check if include directories have already been added to this target
  get_target_property(_cuda_includes_applied ${target_name} LLDB_CUDA_INCLUDES_APPLIED)
  if(_cuda_includes_applied)
    return()
  endif()
  
  # Mark that we've added includes to this target
  set_target_properties(${target_name} PROPERTIES LLDB_CUDA_INCLUDES_APPLIED TRUE)
  
  target_include_directories(${target_name} ${ARGN} ${NVGPU_DEBUGGER_INCLUDE_DIR})
endfunction()

# Function to apply CUDA environment variables as compile definitions to a target
# These environment variables control various CUDA runtime behaviors and are
# commonly needed by CUDA-related plugins
function(lldb_apply_cuda_env_definitions target_name)
  # Check if definitions have already been applied to this target
  get_target_property(_cuda_env_applied ${target_name} LLDB_CUDA_ENV_DEFINITIONS_APPLIED)
  if(_cuda_env_applied)
    return()
  endif()
  
  # Mark that we've applied definitions to this target
  set_target_properties(${target_name} PROPERTIES LLDB_CUDA_ENV_DEFINITIONS_APPLIED TRUE)
  
  if(NVGPU_CUDBG_INJECTION_PATH)
    target_compile_definitions(${target_name} PRIVATE 
      CMAKE_NVGPU_CUDBG_INJECTION_PATH="${NVGPU_CUDBG_INJECTION_PATH}")
  endif()
  if(NVGPU_CUDA_VISIBLE_DEVICES)
    target_compile_definitions(${target_name} PRIVATE 
      CMAKE_NVGPU_CUDA_VISIBLE_DEVICES="${NVGPU_CUDA_VISIBLE_DEVICES}")
  endif()
  if(NVGPU_CUDA_DEVICE_ORDER)
    target_compile_definitions(${target_name} PRIVATE 
      CMAKE_NVGPU_CUDA_DEVICE_ORDER="${NVGPU_CUDA_DEVICE_ORDER}")
  endif()
  if(NVGPU_CUDA_LAUNCH_BLOCKING)
    target_compile_definitions(${target_name} PRIVATE
      CMAKE_NVGPU_CUDA_LAUNCH_BLOCKING="${NVGPU_CUDA_LAUNCH_BLOCKING}")
  endif()

  if(NOT NVGPU_INITIALIZATION_SYMBOL)
    set(NVGPU_INITIALIZATION_SYMBOL "cuInit")
  endif()
  target_compile_definitions(${target_name} PRIVATE
    CMAKE_NVGPU_INITIALIZATION_SYMBOL="${NVGPU_INITIALIZATION_SYMBOL}")
endfunction()
