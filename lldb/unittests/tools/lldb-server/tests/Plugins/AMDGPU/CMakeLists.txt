add_lldb_unittest(LLDBServerAMDGPUPluginTests
  AMDGPUPluginTests.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbCore
    lldbHostHelpers
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformLinux
    lldbPluginProcessGDBRemote
    lldbServerPluginAMDGPU

    LLVMTestingSupport
  )

target_include_directories(LLDBServerAMDGPUPluginTests PRIVATE
  ${LLDB_SOURCE_DIR}/tools/lldb-server/Plugins/AMDGPU)

  add_dependencies(LLDBServerAMDGPUPluginTests lldb-server ${ALL_LLDB_TEST_EXECUTABLES})
