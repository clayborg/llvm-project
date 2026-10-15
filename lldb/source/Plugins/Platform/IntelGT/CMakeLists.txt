configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/PlatformIntelGTUserProperties.td.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PlatformIntelGTUserProperties.td"
  @ONLY
)

lldb_tablegen(PlatformIntelGTUserProperties.inc -gen-lldb-property-defs
  SOURCE "${CMAKE_CURRENT_BINARY_DIR}/PlatformIntelGTUserProperties.td"
  TARGET LLDBPluginPlatformIntelGTUserPropertiesGen)

lldb_tablegen(PlatformIntelGTUserPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE "${CMAKE_CURRENT_BINARY_DIR}/PlatformIntelGTUserProperties.td"
  TARGET LLDBPluginPlatformIntelGTUserPropertiesEnumGen)

add_lldb_library(lldbPluginPlatformIntelGT PLUGIN
  PlatformIntelGT.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbTarget
    lldbUtility
  )

add_dependencies(lldbPluginPlatformIntelGT
  LLDBPluginPlatformIntelGTUserPropertiesGen
  LLDBPluginPlatformIntelGTUserPropertiesEnumGen)
