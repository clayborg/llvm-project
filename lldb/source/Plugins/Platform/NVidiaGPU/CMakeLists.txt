add_lldb_library(lldbPluginPlatformNVidiaGPU PLUGIN
  PlatformNVidiaGPU.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
