set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND Platform)
set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES
  DynamicLoader
  ObjectContainer
  Process
)

add_subdirectory(AIX)
add_subdirectory(AMDGPU)
add_subdirectory(Android)
add_subdirectory(FreeBSD)
add_subdirectory(gdb-server)
add_subdirectory(Linux)
add_subdirectory(MacOSX)
add_subdirectory(NetBSD)
if(LLDB_ENABLE_NVGPU_PLUGIN)
  add_subdirectory(NVGPU)
endif()
add_subdirectory(OpenBSD)
add_subdirectory(POSIX)
add_subdirectory(QemuUser)
add_subdirectory(Windows)
