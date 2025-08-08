
add_lldb_library(lldbServerPluginUtils
  Bitset.cpp
  Utils.cpp
)

target_include_directories(lldbServerPluginUtils PRIVATE "${LLDB_SOURCE_DIR}/source" "../..")
