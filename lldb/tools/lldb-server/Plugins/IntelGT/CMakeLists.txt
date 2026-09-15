# Locate Level Zero using PkgConfig
find_package(PkgConfig REQUIRED)
pkg_search_module(LevelZero REQUIRED IMPORTED_TARGET libze_loader)

add_lldb_library(lldbServerPluginIntelGT
  LLDBServerPluginIntelGT.cpp
  ProcessIntelGT.cpp
  ThreadIntelGT.cpp
  EUThreadIntelGT.cpp
  RegisterContextIntelGT.cpp
  IntelGTArch.cpp
  LINK_LIBS
  PkgConfig::LevelZero
)

target_include_directories(lldbServerPluginIntelGT PRIVATE
  "${LLDB_SOURCE_DIR}/source"
  "../.."
  ${LevelZero_INCLUDE_DIRS}
)
