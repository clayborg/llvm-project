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
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Threading.h"

#include "ProcessAmdGpuCore.h"
#include "ThreadAMDGPU.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ProcessAmdGpuCore)

llvm::StringRef ProcessAmdGpuCore::GetPluginDescriptionStatic() {
  return "ELF amd gpu core dump plug-in.";
}

void ProcessAmdGpuCore::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    // Register as GPU Process plugin (for merged CPU+GPU cores)
    // AMD only supports merged mode, not standalone GPU-only cores
    ProcessElfGpuCore::RegisterEmbeddedCorePlugin(
        GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessAmdGpuCore::Terminate() {
  ProcessElfGpuCore::UnregisterEmbeddedCorePlugin(
      ProcessAmdGpuCore::CreateInstance);
}

std::shared_ptr<ProcessElfGpuCore> ProcessAmdGpuCore::CreateInstance(
    std::shared_ptr<ProcessElfCore> cpu_core_process,
    lldb::ListenerSP listener_sp, const FileSpec *crash_file) {
  // Check if this core file has AMD GPU notes (type 33 =
  // NT_AMDGPU_KFD_CORE_STATE)
  if (!cpu_core_process->GetAmdGpuNote().has_value()) {
    return nullptr;
  }
  llvm::Expected<lldb::TargetSP> gpu_target_or_err =
      CreateGpuTarget(cpu_core_process->GetTarget().GetDebugger());
  if (!gpu_target_or_err) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "Failed to create GPU target: %s",
              llvm::toString(gpu_target_or_err.takeError()).c_str());
    return nullptr;
  }

  TargetSP gpu_target_sp = *gpu_target_or_err;
  auto gpu_process_sp = std::make_shared<ProcessAmdGpuCore>(
      cpu_core_process, gpu_target_sp, listener_sp, *crash_file);
  if (!gpu_process_sp) {
    LLDB_LOGF(GetLog(LLDBLog::Process), "Failed to create GPU process");
    return nullptr;
  }
  // Associate the GPU process on the GPU target (this is critical!)
  gpu_target_sp->SetProcessSP(gpu_process_sp);

  // Set up the CPU-GPU connection
  cpu_core_process->GetTarget().SetGPUPluginTarget(
      gpu_process_sp->GetPluginName(), gpu_target_sp);
  return gpu_process_sp;
}

// Destructor
ProcessAmdGpuCore::~ProcessAmdGpuCore() {
  if (m_gpu_pid.handle != AMD_DBGAPI_PROCESS_NONE.handle) {
    amd_dbgapi_status_t status = amd_dbgapi_process_detach(m_gpu_pid);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(LLDBLog::Process), "Failed to detach from process: %d",
                status);
    }
    m_gpu_pid = AMD_DBGAPI_PROCESS_NONE;
    status = amd_dbgapi_finalize();
    assert(status == AMD_DBGAPI_STATUS_SUCCESS);
  }
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
    std::optional<CoreNote> amd_gpu_note_opt =
        process->GetCpuProcess()->GetAmdGpuNote();

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

  // Write operations are not supported for core files (which are read-only)
  // Return ERROR_NOT_SUPPORTED to explicitly indicate this is not allowed
  if (write_buffer != nullptr) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "xfer_global_memory callback: write operation not supported for "
              "read-only core file (address=0x%" PRIx64 ", size=%zu)",
              global_address, *value_size);
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  }

  Status error;
  std::shared_ptr<Process> cpu_process_sp = process->GetCpuProcess();
  if (!cpu_process_sp) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "xfer_global_memory callback failed to get cpu process");
    return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
  }
  size_t ret = cpu_process_sp->ReadMemory(global_address, read_buffer,
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

bool ProcessAmdGpuCore::initRocm() {
  // Initialize AMD Debug API with callbacks
  amd_dbgapi_status_t status = amd_dbgapi_initialize(&s_dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Process), "Failed to initialize AMD debug API");
    return false;
  }

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

const ArchSpec &ProcessAmdGpuCore::GetArchitecture() {
  // Query the subtype from the dbgapi.
  uint32_t cpu_subtype = 0;
  amd_dbgapi_status_t status = amd_dbgapi_architecture_get_info(
      m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
      sizeof(cpu_subtype), &cpu_subtype);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(LLDBLog::Process),
              "amd_dbgapi_architecture_get_info failed: %d", status);
  }

  m_arch = ArchSpec(eArchTypeELF, llvm::ELF::EM_AMDGPU, cpu_subtype);
  m_arch.MergeFrom(ArchSpec("amdgcn-amd-amdhsa"));
  return m_arch;
}

// Process Control
Status ProcessAmdGpuCore::DoLoadCore() {
  if (!initRocm()) {
    return Status::FromErrorString(
        "Failed to initialize AMD ROCm debug API. Please ensure ROCm is "
        "properly installed and the core file contains valid GPU state.");
  }

  GetTarget().SetArchitecture(GetArchitecture());
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

  amd_dbgapi_status_t status = amd_dbgapi_process_code_object_list(
      m_gpu_pid, &count, &code_object_list, nullptr);
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
  bool ret = true;
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

  // Ensure wave_list is always freed, even if we return early
  auto wave_list_cleanup =
      llvm::make_scope_exit([wave_list]() { free(wave_list); });

  if (changed == AMD_DBGAPI_CHANGED_NO)
    return ret;

  for (size_t i = 0; i < count; ++i) {
    auto thread = std::make_unique<ThreadAMDGPU>(
        *this, m_architecture_id, wave_list[i].handle, wave_list[i]);
    new_thread_list.AddThread(std::move(thread));
  }

  return ret;
}

void ProcessAmdGpuCore::AddThread(amd_dbgapi_wave_id_t wave_id) {}
