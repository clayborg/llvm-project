set(LLVM_TARGET_DEFINITIONS LLGSOptions.td)
tablegen(LLVM LLGSOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLGSOptionsTableGen)

set(LLDB_PLUGINS)

if(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  list(APPEND LLDB_PLUGINS lldbPluginProcessLinux)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND LLDB_PLUGINS lldbPluginProcessAIX)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND LLDB_PLUGINS lldbPluginProcessFreeBSD)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND LLDB_PLUGINS lldbPluginProcessNetBSD)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  list(APPEND LLDB_PLUGINS lldbPluginObjectFileMachO)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(APPEND LLDB_PLUGINS lldbPluginObjectFilePECOFF)
elseif(CMAKE_SYSTEM_NAME MATCHES "AIX")
  list(APPEND LLDB_PLUGINS lldbPluginObjectFileXCOFF)
else()
  list(APPEND LLDB_PLUGINS lldbPluginObjectFileELF)
endif()

if(APPLE_EMBEDDED)
  if(LLDB_CODESIGN_IDENTITY)
    # Use explicit LLDB identity
    set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY})
  else()
    # Use explicit LLVM identity or default to lldb_codesign if empty
    if(NOT LLVM_CODESIGNING_IDENTITY)
      set(LLVM_CODESIGNING_IDENTITY lldb_codesign)
    endif()
  endif()
endif()

option(LLDB_ENABLE_AMDGPU_PLUGIN "Enable support for the AMD GPU plugin" OFF)
option(LLDB_ENABLE_MOCK_GPU_PLUGIN "Enable support for the Mock GPU plugin" OFF)

if(LLDB_ENABLE_AMDGPU_PLUGIN)
  if(NOT DEFINED ROCM_PATH)
    message(FATAL_ERROR "ROCM_PATH is not defined and LLDB_ENABLE_AMDGPU_PLUGIN expects it. Use -DROCM_PATH=/path/to/rocm")
  endif()

  if(NOT EXISTS ${ROCM_PATH})
    message(FATAL_ERROR "ROCM_PATH does not exist: ${ROCM_PATH}. Use -DROCM_PATH=/path/to/rocm")
  endif()
  message(STATUS "ROCM_PATH is set to: '${ROCM_PATH}'")

  add_definitions(-DLLDB_ENABLE_AMDGPU_PLUGIN=1)
  list(APPEND LLDB_PLUGINS lldbServerPluginAMDGPU)
endif()

if(LLDB_ENABLE_MOCK_GPU_PLUGIN)
  add_definitions(-DLLDB_ENABLE_MOCKGPU_PLUGIN=1)
  list(APPEND LLDB_PLUGINS lldbServerPluginMockGPU)
endif()

add_lldb_tool(lldb-server
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp
    SystemInitializerLLGS.cpp

    LINK_COMPONENTS
      Option
      Support
    LINK_LIBS
      lldbHost
      lldbInitialization
      lldbVersion
      ${LLDB_PLUGINS}
      lldbPluginInstructionARM
      lldbPluginInstructionLoongArch
      lldbPluginInstructionMIPS
      lldbPluginInstructionMIPS64
      lldbPluginInstructionRISCV
      ${LLDB_SYSTEM_LIBS}
)

set_target_properties(lldb-server PROPERTIES ENABLE_EXPORTS 1)

add_dependencies(lldb-server
  LLGSOptionsTableGen
  ${tablegen_deps}
)
target_include_directories(lldb-server PRIVATE "${LLDB_SOURCE_DIR}/source")
target_link_libraries(lldb-server PRIVATE ${LLDB_SYSTEM_LIBS})

add_subdirectory(Plugins)
