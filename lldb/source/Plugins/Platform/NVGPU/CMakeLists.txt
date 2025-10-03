configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/PlatformNVGPUUserProperties.td.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PlatformNVGPUUserProperties.td"
  @ONLY
)

lldb_tablegen(PlatformNVGPUUserProperties.inc -gen-lldb-property-defs
  SOURCE "${CMAKE_CURRENT_BINARY_DIR}/PlatformNVGPUUserProperties.td"
  TARGET LLDBPluginPlatformNVGPUUserPropertiesGen)

lldb_tablegen(PlatformNVGPUUserPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE "${CMAKE_CURRENT_BINARY_DIR}/PlatformNVGPUUserProperties.td"
  TARGET LLDBPluginPlatformNVGPUUserPropertiesEnumGen)


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

add_dependencies(lldbPluginPlatformNVGPU
  LLDBPluginPlatformNVGPUUserPropertiesGen
  LLDBPluginPlatformNVGPUUserPropertiesEnumGen)
