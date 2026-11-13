set(FREEBSD_SOURCES
  RegisterContextFreeBSDTest.cpp)
set(NETBSD_SOURCES
  RegisterContextNetBSDTest_i386.cpp
  RegisterContextNetBSDTest_x86_64.cpp)
set(AMDGPU_SOURCES
  AmdDbgApiRegisterTypeParserTest.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND PLATFORM_SOURCES ${FREEBSD_SOURCES})
elseif (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND PLATFORM_SOURCES ${NETBSD_SOURCES})
endif()

if (LLDB_ENABLE_AMDGPU_PLUGIN)
  list(APPEND PLATFORM_SOURCES ${AMDGPU_SOURCES})
endif()

set(LLVM_OPTIONAL_SOURCES
  ${FREEBSD_SOURCES}
  ${NETBSD_SOURCES}
  ${AMDGPU_SOURCES})

add_lldb_unittest(ProcessUtilityTests
  LinuxProcMapsTest.cpp
  MemoryTagManagerAArch64MTETest.cpp
  RegisterContextTest.cpp
  CoreFileMemoryRangesTest.cpp
  ${PLATFORM_SOURCES}

  LINK_LIBS
    lldbPluginProcessUtility
    LLVMTestingSupport)
