add_lldb_library(lldbPluginStructuredDataNVIDIAGPUMonitor PLUGIN
  StructuredDataNVIDIAGPUMonitor.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
  )
