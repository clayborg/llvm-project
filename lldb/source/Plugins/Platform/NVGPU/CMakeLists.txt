add_lldb_library(lldbPluginPlatformNVGPU PLUGIN
  PlatformNVGPU.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
