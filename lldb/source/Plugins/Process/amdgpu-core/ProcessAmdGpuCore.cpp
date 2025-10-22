//===-- ProcessAmdGpuCore.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include <memory>

#include "Plugins/DynamicLoader/GpuCore-DYLD/DynamicLoaderGpuCoreDYLD.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Threading.h"

#include "ProcessAmdGpuCore.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ProcessAmdGpuCore)

llvm::StringRef ProcessAmdGpuCore::GetPluginDescriptionStatic() {
  return "ELF amd gpu core dump plug-in.";
}

void ProcessAmdGpuCore::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessAmdGpuCore::Terminate() {
  PluginManager::UnregisterPlugin(ProcessAmdGpuCore::CreateInstance);
}

lldb::ProcessSP ProcessAmdGpuCore::CreateInstance(lldb::TargetSP target_sp,
                                                  lldb::ListenerSP listener_sp,
                                                  const FileSpec *crash_file,
                                                  bool can_connect) {
  return std::make_shared<ProcessAmdGpuCore>(target_sp, listener_sp,
                                             *crash_file);
}

bool ProcessAmdGpuCore::CanDebug(lldb::TargetSP target_sp,
                                 bool plugin_specified_by_name) {
  // TODO: check target is a ELF core file.
  return ProcessElfGpuCore::CanDebug(target_sp, plugin_specified_by_name);
}

// Destructor
ProcessAmdGpuCore::~ProcessAmdGpuCore() {
  // TODO: clean up for AMD GPU.
}

static amd_dbgapi_status_t amd_dbgapi_client_process_get_info_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_client_process_info_t query, size_t value_size, void *value) {
  ProcessAmdGpuCore *process =
      reinterpret_cast<ProcessAmdGpuCore *>(client_process_id);
  lldb::pid_t pid = process->GetID();
  LLDB_LOGF(GetLog(LLDBLog::Process),
            "amd_dbgapi_client_process_get_info_callback callback, with query "
            "%d, pid %lu",
            query, (unsigned long)pid);
  switch (query) {
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID: {
    /* AMD debug API expects AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE
            for coredump.*/
    return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
  }
  case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE: {
    amd_dbgapi_core_state_data_t *core_state_data =
        (amd_dbgapi_core_state_data_t *)value;
    std::optional<CoreNote> amd_gpu_note_opt = process->GetAmdGpuNote();

    if (!amd_gpu_note_opt.has_value())
      return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;

    void *core_state = malloc(amd_gpu_note_opt->data.GetByteSize());
    amd_gpu_note_opt->data.CopyData(0, amd_gpu_note_opt->data.GetByteSize(),
                                    core_state);
    core_state_data->size = amd_gpu_note_opt->data.GetByteSize();
    core_state_data->data = core_state;
    core_state_data->endianness =
        amd_gpu_note_opt->data.GetByteOrder() == lldb::eByteOrderLittle
            ? AMD_DBGAPI_ENDIAN_LITTLE
            : AMD_DBGAPI_ENDIAN_BIG;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  }
}

static amd_dbgapi_status_t amd_dbgapi_insert_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t address,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log, "insert_breakpoint callback at address: 0x%" PRIx64, address);
  // TODO: should not be called for coredump.
  return AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED;
}

/* remove_breakpoint callback.  */

static amd_dbgapi_status_t amd_dbgapi_remove_breakpoint_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_breakpoint_id_t breakpoint_id) {
  LLDB_LOGF(GetLog(LLDBLog::Process), "remove_breakpoint callback for %" PRIu64,
            breakpoint_id.handle);
  return AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED;
}

/* xfer_global_memory callback.  */

static amd_dbgapi_status_t amd_dbgapi_xfer_global_memory_callback(
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t global_address, amd_dbgapi_size_t *value_size,
    void *read_buffer, const void *write_buffer) {
  LLDB_LOGF(GetLog(LLDBLog::Process),
            "xfer_global_memory callback for address: 0x%" PRIx64,
            global_address);
  ProcessAmdGpuCore *process =
      reinterpret_cast<ProcessAmdGpuCore *>(client_process_id);
  if (!process)
    return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
  Status error;
  size_t ret = process->GetCpuProcess()->ReadMemory(global_address, read_buffer,
                                                    *value_size, error);
  if (error.Fail() || ret != *value_size)
    return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
  return AMD_DBGAPI_STATUS_SUCCESS;
}

static void amd_dbgapi_log_message_callback(amd_dbgapi_log_level_t level,
                                            const char *message) {
  LLDB_LOGF(GetLog(LLDBLog::Process), "ROCdbgapi [%d]: %s", level, message);
}

static amd_dbgapi_callbacks_t s_dbgapi_callbacks = {
    malloc,
    free,
    amd_dbgapi_client_process_get_info_callback,
    amd_dbgapi_insert_breakpoint_callback,
    amd_dbgapi_remove_breakpoint_callback,
    amd_dbgapi_xfer_global_memory_callback,
    amd_dbgapi_log_message_callback,
};

std::optional<CoreNote> ProcessAmdGpuCore::GetAmdGpuNote() {
  ObjectFileELF *core = (ObjectFileELF *)m_cpu_core_process->GetObjectFile();
  if (core == nullptr)
    return std::nullopt;
  llvm::ArrayRef<elf::ELFProgramHeader> program_headers =
      core->ProgramHeaders();
  if (program_headers.size() == 0)
    return std::nullopt;

  for (const elf::ELFProgramHeader &H : program_headers) {
    if (H.p_type == llvm::ELF::PT_NOTE) {
      DataExtractor segment_data = core->GetSegmentData(H);
      llvm::Expected<std::vector<CoreNote>> notes_or_error =
          parseSegment(segment_data);
      if (!notes_or_error) {
        llvm::consumeError(notes_or_error.takeError());
        continue;
      }
      for (const auto &note : *notes_or_error) {
        if (note.info.n_type == 33)
          return note;
      }
    }
  }
  return std::nullopt;
}

bool ProcessAmdGpuCore::initRocm() {
  // Initialize AMD Debug API with callbacks
  amd_dbgapi_status_t status = amd_dbgapi_initialize(&s_dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Process), "Failed to initialize AMD debug API");
    exit(-1);
  }

  // TODO: for debugging purpose. Move into lldb logging channel.
  amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_VERBOSE);
  // Attach to the process with AMD Debug API
  status = amd_dbgapi_process_attach(
      reinterpret_cast<amd_dbgapi_client_process_id_t>(
          this), // Use pid_ as client_process_id
      &m_gpu_pid);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "Failed to attach to process with AMD debug API: %d", status);
    amd_dbgapi_finalize();
    return false;
  }

  amd_dbgapi_architecture_id_t architecture_id;
  // TODO: do not hardcode the device id
  status = amd_dbgapi_get_architecture(0x04C, &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    // Handle error
    LLDB_LOGF(GetLog(LLDBLog::Process), "amd_dbgapi_get_architecture failed");
    return false;
  }
  m_architecture_id = architecture_id;

  return true;
}

// llvm::Expected<lldb::ProcessSP>
// ProcessAmdGpuCore::LoadGpuCore(Debugger &debugger, const FileSpec &core_file) {
//   Log *log = GetLog(LLDBLog::Process);
//   LLDB_LOG(log, "ProcessAmdGpuCore::LoadGpuCore()");

//   TargetSP gpu_target_sp;
//   llvm::StringRef exe_path;
//   llvm::StringRef triple;
//   OptionGroupPlatform *platform_options = nullptr;
//   // Create an empty target for our GPU.
//   Status error(debugger.GetTargetList().CreateTarget(
//       debugger, exe_path, triple, eLoadDependentsNo, platform_options,
//       gpu_target_sp));
//   if (error.Fail())
//     return error.ToError();
//   if (!gpu_target_sp)
//     return llvm::createStringError(llvm::inconvertibleErrorCode(),
//                                    "failed to create target");

//   ProcessSP gpu_process_sp =
//       CreateInstance(gpu_target_sp, debugger.GetListener(), &core_file, false);
//   if (!gpu_process_sp)
//     return llvm::createStringError(llvm::inconvertibleErrorCode(),
//                                    "failed to create GPU process");

//   error = gpu_process_sp->DoLoadCore();
//   if (error.Fail())
//     return error.ToError();

//   return gpu_process_sp;
// }

// Process Control
Status ProcessAmdGpuCore::DoLoadCore() {
  // Status error = ProcessElfGpuCore::DoLoadCore();
  // if (error.Fail())
  //   return error;
  initRocm();
  return Status{};
}

lldb_private::DynamicLoader *ProcessAmdGpuCore::GetDynamicLoader() {
  if (m_dyld_up.get() == nullptr)
    m_dyld_up.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderGpuCoreDYLD::GetPluginNameStatic()));
  return m_dyld_up.get();
}

llvm::Expected<lldb_private::LoadedModuleInfoList>
ProcessAmdGpuCore::GetLoadedModuleList() {
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log, "ProcessAmdGpuCore::GetLoadedModuleList()");

  amd_dbgapi_code_object_id_t *code_object_list;
  size_t count;

  amd_dbgapi_process_id_t gpu_pid{GetID()};
  amd_dbgapi_status_t status = amd_dbgapi_process_code_object_list(
      gpu_pid, &count, &code_object_list, nullptr);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(log, "Failed to get code object list: %d", status);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to get code object list");
  }
  LoadedModuleInfoList module_list;
  for (size_t i = 0; i < count; ++i) {
    uint64_t l_addr;
    char *uri_bytes;

    status = amd_dbgapi_code_object_get_info(
        code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
        sizeof(l_addr), &l_addr);
    if (status != AMD_DBGAPI_STATUS_SUCCESS)
      continue;

    status = amd_dbgapi_code_object_get_info(
        code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME,
        sizeof(uri_bytes), &uri_bytes);
    if (status != AMD_DBGAPI_STATUS_SUCCESS)
      continue;

    LLDB_LOGF(log, "Code object %zu: %s at address %" PRIu64, i, uri_bytes,
              l_addr);

    LoadedModuleInfoList::LoadedModuleInfo module_info;
    module_info.set_name(uri_bytes);
    module_info.set_base(l_addr);
    module_list.m_list.push_back(module_info);
    s_dbgapi_callbacks.deallocate_memory(uri_bytes);
  }  
  return module_list;
}

bool ProcessAmdGpuCore::DoUpdateThreadList(ThreadList &old_thread_list,
                                           ThreadList &new_thread_list) {
  bool ret = false;
  size_t count;
  amd_dbgapi_wave_id_t *wave_list;
  amd_dbgapi_changed_t changed;
  amd_dbgapi_status_t status =
      amd_dbgapi_process_wave_list(m_gpu_pid, &count, &wave_list, &changed);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "amd_dbgapi_process_wave_list failed with status %d", status);
    return false;
  }

  if (changed == AMD_DBGAPI_CHANGED_NO)
    return ret;

  // TODO: wrap with RAII
  free(wave_list);
  return ret;
}
