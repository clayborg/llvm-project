include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "processamdgpucore" in {
  def DbgApiLogLevel: Property<"dbgapi-log-level", "Enum">,
    Global,
    DefaultEnumValue<"eDbgApiLogLevelWarning">,
    EnumValues<"OptionEnumValues(g_dbgapi_log_level_values)">,
    Desc<"Set the AMD debug API (ROCdbgapi) log level. "
         "Levels: none, fatal, warning, info, verbose, trace.">;
}
