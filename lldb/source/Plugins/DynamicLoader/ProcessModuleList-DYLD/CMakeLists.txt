add_lldb_library(lldbPluginDynamicLoaderProcessModuleList PLUGIN
  DynamicLoaderProcessModuleList.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessAmdGpuCore
  LINK_COMPONENTS
    Support
  )
