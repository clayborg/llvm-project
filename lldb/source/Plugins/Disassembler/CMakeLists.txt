set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND Disassembler)

add_subdirectory(LLVMC)
if(LLDB_ENABLE_NVGPU_PLUGIN)
  add_subdirectory(SASS)
endif()
