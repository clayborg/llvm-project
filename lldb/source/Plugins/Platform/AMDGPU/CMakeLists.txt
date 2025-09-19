add_lldb_library(lldbPluginPlatformAMDGPU PLUGIN
  PlatformAMDGPU.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
