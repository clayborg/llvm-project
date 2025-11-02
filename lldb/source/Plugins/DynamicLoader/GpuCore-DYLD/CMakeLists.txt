add_lldb_library(lldbPluginDynamicLoaderGpuCoreDYLD PLUGIN
  DynamicLoaderGpuCoreDYLD.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessAmdGpuCore
  LINK_COMPONENTS
    Support
  )
