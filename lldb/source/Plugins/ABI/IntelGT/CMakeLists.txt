add_lldb_library(lldbPluginABIIntelGT PLUGIN
  ABIIntelGT.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessUtility
  LINK_COMPONENTS
    MC
    Support
    TargetParser
  )
