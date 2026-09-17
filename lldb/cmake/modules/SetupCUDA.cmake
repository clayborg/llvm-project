#===----------------------------------------------------------------------===//
#
# Common CUDA setup logic for LLDB plugins that require CUDA support.
#
# Builds against the CUDA debugger API headers vendored in
# lldb/third-party/cuda, or the folder NVGPU_DEBUGGER_INCLUDE_DIR names.
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
    "Path to a folder containing the CUDA debugger header files (cudadebugger.h, cudacoredump.h), used instead of the copies in lldb/third-party/cuda")
set(NVGPU_DEBUGGER_INCLUDE_DIR CACHE STRING "${NVGPU_DEBUGGER_INCLUDE_DIR_DESC}")

set(TROUBLESHOOTING_MESSAGE
    "Set NVGPU_DEBUGGER_INCLUDE_DIR to a folder holding both headers, or clear it to use the copies in lldb/third-party/cuda.")

# Pick the directory to compile against. NVGPU_DEBUGGER_INCLUDE_DIR itself is
# left empty, because that is what marks the in-tree copies as the default.
if(NVGPU_DEBUGGER_INCLUDE_DIR)
  # Resolve against the build directory, which is cmake's working directory.
  # A relative path would otherwise mean something different in each of the
  # directories that consume it.
  get_filename_component(cudbg_include_dir "${NVGPU_DEBUGGER_INCLUDE_DIR}"
      ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
  set(cudbg_include_origin "from NVGPU_DEBUGGER_INCLUDE_DIR")
else()
  # Resolve against this file, so the default does not depend on
  # LLDB_SOURCE_DIR, which a standalone lldb build sets differently.
  get_filename_component(cudbg_include_dir
      "${CMAKE_CURRENT_LIST_DIR}/../../third-party/cuda" ABSOLUTE)
  set(cudbg_include_origin "in the source tree")
endif()

foreach(cudbg_header cudadebugger.h cudacoredump.h)
  if(NOT EXISTS "${cudbg_include_dir}/${cudbg_header}")
    message(FATAL_ERROR
        "${cudbg_include_dir} (${cudbg_include_origin}) does not contain "
        "${cudbg_header}. ${TROUBLESHOOTING_MESSAGE}")
  endif()
endforeach()

# Cached so lldb_add_cuda_include_dirs can read it from the other directories.
# CACHE INTERNAL implies FORCE, so it is rewritten on every configure.
set(LLDB_NVGPU_SELECTED_CUDBG_INCLUDE_DIR "${cudbg_include_dir}" CACHE INTERNAL
    "CUDA debugger include directory LLDB's NVGPU targets compile against.")

# The Toolkit is only used for nvcc now, which the test suite needs to compile
# its CUDA programs (see NVGPU_NVCC_PATH in lldb/test/API/lit.site.cfg.py.in).
if(NOT NVGPU_NVCC_PATH)
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    set(NVGPU_NVCC_PATH "${CUDAToolkit_NVCC_EXECUTABLE}"
        CACHE STRING "Path to the NVCC compiler." FORCE)
  endif()
endif()

# Function to verify NVCC compiler is available
# Some CUDA plugins require the NVCC compiler for runtime compilation
function(lldb_verify_nvcc_available)
  if(NOT NVGPU_NVCC_PATH)
    message(FATAL_ERROR "NVGPU_NVCC_PATH not set. Please install the CUDA Toolkit or set a valid NVGPU_NVCC_PATH CMake variable, or configure with LLDB_INCLUDE_TESTS=OFF.")
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

  # BEFORE puts this ahead of the directory-wide include paths from
  # add_lldb_library and LLDBConfig.cmake. It does not outrank the
  # CMAKE_INCLUDE_CURRENT_DIR entries, which hold no cudadebugger.h.
  target_include_directories(${target_name} BEFORE ${ARGN}
    ${LLDB_NVGPU_SELECTED_CUDBG_INCLUDE_DIR})
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
