add_lldb_unittest(ObjectContainerTests
  ObjectContainerClangOffloadBundleTest.cpp

  LINK_COMPONENTS
    Object
    Support
  LINK_LIBS
    lldbPluginObjectContainerClangOffloadBundle
    lldbPluginObjectFileELF
    lldbCore
    lldbUtilityHelpers
    LLVMTestingSupport
  )
