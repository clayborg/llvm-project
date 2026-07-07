add_lldb_library(lldbPluginArchitectureIntelGT PLUGIN
  ArchitectureIntelGT.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
  )
