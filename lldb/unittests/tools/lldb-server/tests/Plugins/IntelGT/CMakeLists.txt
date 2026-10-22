add_lldb_unittest(LLDBServerIntelGTPluginTests
  RegisterContextIntelGTTest.cpp
  # Include the actual source file we're testing for coverage
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/RegisterContextIntelGT.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/IntelGTArch.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/ProcessIntelGT.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/ThreadIntelGT.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/EUThreadIntelGT.cpp
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT/LLDBServerPluginIntelGT.cpp
  )

# Find Level Zero
find_package(PkgConfig REQUIRED)
pkg_search_module(LevelZero REQUIRED IMPORTED_TARGET libze_loader)


target_include_directories(LLDBServerIntelGTPluginTests
  PRIVATE
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/IntelGT
  ${LLDB_SOURCE_DIR}/source
  ${LLDB_SOURCE_DIR}/tools/lldb-server
  ${LevelZero_INCLUDE_DIRS}
  )

target_link_libraries(LLDBServerIntelGTPluginTests
  PRIVATE
  PkgConfig::LevelZero
  lldbPluginProcessGDBRemote
  lldbPluginProcessUtility
  lldbPluginObjectFileELF
  lldbHost
  lldbUtility
  lldbCore
  lldbSymbol
  lldbTarget
  )

# Enable code coverage for IntelGT tests if requested
if(LLVM_BUILD_INSTRUMENTED_COVERAGE OR INTELGT_ENABLE_COVERAGE)
  message(STATUS "Code coverage enabled for IntelGT unit tests")

  # Use compiler-specific coverage flags
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang/LLVM coverage flags
    target_compile_options(LLDBServerIntelGTPluginTests
      PRIVATE
      -fprofile-instr-generate
      -fcoverage-mapping
      )
    target_link_options(LLDBServerIntelGTPluginTests
      PRIVATE
      -fprofile-instr-generate
      )
    message(STATUS "  Using Clang coverage instrumentation")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC coverage flags
    target_compile_options(LLDBServerIntelGTPluginTests
      PRIVATE
      --coverage
      -fprofile-arcs
      -ftest-coverage
      )
    target_link_options(LLDBServerIntelGTPluginTests
      PRIVATE
      --coverage
      )
    message(STATUS "  Using GCC coverage instrumentation")
  else()
    message(WARNING "Code coverage requested but compiler ${CMAKE_CXX_COMPILER_ID} not supported")
  endif()
endif()
