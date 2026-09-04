add_lldb_library(lldbPluginDisassemblerIntelGT PLUGIN
  DisassemblerIntelGT.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
  LINK_COMPONENTS
    Support
)
