add_lldb_library(lldbPluginStructuredDataNVGPUMonitor PLUGIN
  StructuredDataNVGPUMonitor.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
  )
