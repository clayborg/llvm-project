set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND StructuredData)

add_subdirectory(DarwinLog)
if(LLDB_ENABLE_NVGPU_PLUGIN)
  add_subdirectory(NVGPUMonitor)
endif()

