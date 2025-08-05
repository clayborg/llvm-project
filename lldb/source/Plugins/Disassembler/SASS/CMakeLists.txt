add_lldb_library(lldbPluginDisassemblerSASS PLUGIN
  DisassemblerSASS.cpp
  DisassemblerSASSCache.cpp
  InstructionSASS.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  ) 
