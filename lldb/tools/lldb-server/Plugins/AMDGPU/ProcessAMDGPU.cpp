//===-- ProcessAMDGPU.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessAMDGPU.h"
#include "AmdDbgApiHelpers.h"
#include "ThreadAMDGPU.h"

#include "LLDBServerPluginAMDGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/AmdGpuCoreUtils.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <cinttypes>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessAMDGPU::ProcessAMDGPU(lldb::pid_t pid, NativeDelegate &delegate,
                             LLDBServerPluginAMDGPU *plugin)
    : NativeProcessProtocol(pid, -1, delegate), m_debugger(plugin) {
  m_state = eStateStopped;
}

Status ProcessAMDGPU::Resume(const ResumeActionList &resume_actions) {
  SetState(StateType::eStateRunning, true);
  ThreadAMDGPU *thread = (ThreadAMDGPU *)GetCurrentThread();
  thread->GetRegisterContext().InvalidateAllRegisters();
  return Status();
}

Status ProcessAMDGPU::Halt() {
  SetState(StateType::eStateStopped, true);
  return Status();
}

Status ProcessAMDGPU::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

/// Sends a process a UNIX signal \a signal.
///
/// \return
///     Returns an error object.
Status ProcessAMDGPU::Signal(int signo) {
  return Status::FromErrorString("unimplemented");
}

/// Tells a process to interrupt all operations as if by a Ctrl-C.
///
/// The default implementation will send a local host's equivalent of
/// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
/// operation.
///
/// \return
///     Returns an error object.
Status ProcessAMDGPU::Interrupt() { return Status(); }

Status ProcessAMDGPU::Kill() { return Status(); }

Status ProcessAMDGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                 size_t &bytes_read) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessAMDGPU::WriteMemory(lldb::addr_t addr, const void *buf,
                                  size_t size, size_t &bytes_written) {
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t ProcessAMDGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

ThreadAMDGPU *
ProcessAMDGPU::FindThread(std::function<bool(ThreadAMDGPU &)> pred) {
  ThreadAMDGPU *found = nullptr;
  ForEachThread([&found, &pred](ThreadAMDGPU &thread) {
    if (pred(thread)) {
      found = &thread;
      return IterationAction::Stop;
    }
    return IterationAction::Continue;
  });
  return found;
}

void ProcessAMDGPU::ForEachThread(
    std::function<lldb_private::IterationAction(ThreadAMDGPU &)> const
        &callback) {
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
  for (ThreadAMDGPU &thread : AMDGPUThreadRange(m_threads)) {
    if (callback(thread) == IterationAction::Stop)
      break;
  }
}

// Select which thread should be considered the "current thread" in the process.
lldb::tid_t ProcessAMDGPU::ChooseCurrentThread() {
  // Helper to find the first thread with a valid stop reason.
  auto GetFirstThreadWithValidStopReason = [this]() {
    return FindThread(
        [](ThreadAMDGPU &thread) { return thread.HasValidStopReason(); });
  };

  // Return an invalid id when there are no threads.
  if (m_threads.empty())
    return LLDB_INVALID_THREAD_ID;

  // If the current thread has a valid stop reason then use that thread.
  ThreadAMDGPU *current_thread = GetCurrentThreadAMDGPU();
  if (current_thread && current_thread->HasValidStopReason())
    return current_thread->GetID();

  // Otherwise, look for a thread with a valid stop reason.
  if (ThreadAMDGPU *stopped_thread = GetFirstThreadWithValidStopReason())
    return stopped_thread->GetID();

  // If there are no stopped threads then just return the
  // current thread if it exists otherwise choose the first stopped thread.
  return current_thread ? current_thread->GetID() : m_threads.front()->GetID();
}

// Choose which thread is the current thread and update the current thread ID to
// match.
void ProcessAMDGPU::UpdateCurrentThread() {
  lldb::tid_t tid = ChooseCurrentThread();
  SetCurrentThreadID(tid);
}

size_t ProcessAMDGPU::UpdateThreads() {
  UpdateThreadListFromWaves();
  if (m_threads.empty()) {
    m_threads.push_back(ThreadAMDGPU::CreateGPUShadowThread(*this));
  }
  UpdateCurrentThread();
  return m_threads.size();
}

const ArchSpec &ProcessAMDGPU::GetArchitecture() const {
  // Query the subtype from the dbgapi.
  uint32_t cpu_subtype = 0;
  amd_dbgapi_status_t status = amd_dbgapi_architecture_get_info(
      m_debugger->m_architecture_id,
      AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE, sizeof(cpu_subtype),
      &cpu_subtype);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "amd_dbgapi_architecture_get_info failed: %d", status);
  }

  m_arch = ArchSpec(eArchTypeELF, llvm::ELF::EM_AMDGPU, cpu_subtype);
  m_arch.MergeFrom(ArchSpec("amdgcn-amd-amdhsa"));
  return m_arch;
}

// Breakpoint functions
Status ProcessAMDGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                    bool hardware) {
  bool success = m_debugger->CreateGPUBreakpoint(addr);
  if (!success) {
    return Status::FromErrorString("CreateGPUBreakpoint failed");
  }
  return Status();
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessAMDGPU::GetAuxvData() const {
  return nullptr; // TODO: try to return
                  // llvm::make_error<UnimplementedError>();
}

Status ProcessAMDGPU::GetLoadedModuleFileSpec(const char *module_path,
                                              FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessAMDGPU::GetFileLoadAddress(const llvm::StringRef &file_name,
                                         lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}

void ProcessAMDGPU::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<ProcessInfo &>(launch_info);
}

void ProcessAMDGPU::HandleNativeProcessExit(const WaitStatus &exit_status) {
  // Set our exit status to match the native process and notify delegates
  SetExitStatus(exit_status, true);
}

bool ProcessAMDGPU::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "ProcessAMDGPU::%s() entered", __FUNCTION__);
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

std::optional<GPUDynamicLoaderResponse>
ProcessAMDGPU::GetGPUDynamicLoaderLibraryInfos(
    const GPUDynamicLoaderArgs &args) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "ProcessAMDGPU::%s() entered", __FUNCTION__);

  GPUDynamicLoaderResponse response;

  llvm::iterator_range<GpuModuleManager::CodeObjectList::const_iterator>
      code_objects = args.full ? m_gpu_module_manager.GetLoadedCodeObjects()
                               : m_gpu_module_manager.GetChangedCodeObjects();

  LLDB_LOGF(log, "ProcessAMDGPU::%s() found %zu GPU modules", __FUNCTION__,
            std::distance(code_objects.begin(), code_objects.end()));

  for (const GpuModuleManager::CodeObject &code_object : code_objects) {
    // Use the shared ParseLibraryInfo function
    AmdGpuCodeObject amd_code_object(code_object.uri, code_object.load_address,
                                     code_object.IsLoaded());
    if (auto lib_info = ParseLibraryInfo(amd_code_object)) {
      LLDB_LOGF(
          log,
          "ProcessAMDGPU::%s() %s library: path=%s, load_addr=0x%" PRIx64
          ", native_memory_address=%" PRIu64 ", native_memory_size=%" PRIu64
          ", file_offset=%" PRIu64 ", file_size=%" PRIu64,
          __FUNCTION__, lib_info->load ? "load" : "unload",
          lib_info->pathname.c_str(), lib_info->load_address.value_or(0),
          lib_info->native_memory_address.value_or(0),
          lib_info->native_memory_size.value_or(0),
          lib_info->file_offset.value_or(0), lib_info->file_size.value_or(0));
      response.library_infos.push_back(*lib_info);
    } else {
      LLDB_LOGF(log, "ProcessAMDGPU::%s() failed to parse module path \"%s\"",
                __FUNCTION__, code_object.uri.c_str());
    }
  }

  // We have reported all changes to lldb so clear the list
  // to accumulate only the new changes.
  m_gpu_module_manager.ClearChangedObjectList();

  return response;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerAMDGPU::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = launch_info.GetProcessID();
  auto proc_up =
      std::make_unique<ProcessAMDGPU>(pid, native_delegate, m_debugger);
  proc_up->SetLaunchInfo(launch_info);
  return proc_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerAMDGPU::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}

bool ProcessAMDGPU::handleWaveStop(amd_dbgapi_event_id_t eventId) {
  amd_dbgapi_wave_id_t wave_id;
  auto status = amd_dbgapi_event_get_info(eventId, AMD_DBGAPI_EVENT_INFO_WAVE,
                                          sizeof(wave_id), &wave_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_event_get_info failed: %d",
              status);
    return false;
  }
  amd_dbgapi_wave_stop_reasons_t stop_reason;
  status = amd_dbgapi_wave_get_info(wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                                    sizeof(stop_reason), &stop_reason);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_wave_get_info failed: %d",
              status);
    return false;
  }
  if ((stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT) != 0) {
    uint64_t pc;
    status = amd_dbgapi_wave_get_info(wave_id, AMD_DBGAPI_WAVE_INFO_PC,
                                      sizeof(pc), &pc);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_wave_get_info failed: %d",
                status);
      exit(-1);
    }
    pc -= 4;
    amd_dbgapi_register_id_t pc_register_id;
    status = amd_dbgapi_architecture_get_info(
        m_debugger->m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
        sizeof(pc_register_id), &pc_register_id);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "amd_dbgapi_architecture_get_info failed: %d", status);
      exit(-1);
    }
    status =
        amd_dbgapi_write_register(wave_id, pc_register_id, 0, sizeof(pc), &pc);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "amd_dbgapi_write_register failed: %d",
                status);
      exit(-1);
    }

    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Wave stopped due to breakpoint at: 0x%" PRIx64
              " with wave id: %" PRIu64 " "
              "event id: %" PRIu64,
              pc, wave_id.handle, eventId.handle);
    return true;
  } else {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Wave stopped due to unknown reason: %d",
              stop_reason);
  }
  return false;
}

bool ProcessAMDGPU::handleDebugEvent(amd_dbgapi_event_id_t eventId,
                                     amd_dbgapi_event_kind_t eventKind) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "handleDebugEvent(%" PRIu64 ", %s)",
            eventId.handle, AmdDbgApiEventKindToString(eventKind));
  bool result = false;

  // Handle event kinds
  switch (eventKind) {
  case AMD_DBGAPI_EVENT_KIND_NONE:
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "No event received");
    break;

  case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Breakpoint resume event received");
    break;

  case AMD_DBGAPI_EVENT_KIND_WAVE_STOP: {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Wave stop event received");

    // Handle wave stop
    result = handleWaveStop(eventId);
    m_gpu_state = State::GPUStopped;
    break;
  }

  case AMD_DBGAPI_EVENT_KIND_RUNTIME: {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime event received.");
    // Get runtime state for the event
    amd_dbgapi_runtime_state_t runtimeState = AMD_DBGAPI_RUNTIME_STATE_UNLOADED;
    amd_dbgapi_status_t status =
        amd_dbgapi_event_get_info(eventId, AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
                                  sizeof(runtimeState), &runtimeState);

    if (status == AMD_DBGAPI_STATUS_SUCCESS) {
      // Handle different runtime states
      switch (runtimeState) {
      case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
        LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime loaded successfully");
        m_debugger->GpuRuntimeDidLoad();
        break;
      case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
        LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime load restricted");
        break;
      case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
        LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime unloaded");
        break;
      }
    }
    break;
  }

  case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED: {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Code object event received");

    amd_dbgapi_code_object_id_t *code_object_list;
    size_t count;

    amd_dbgapi_process_id_t gpu_pid{GetID()};
    amd_dbgapi_status_t status = amd_dbgapi_process_code_object_list(
        gpu_pid, &count, &code_object_list, nullptr);
    if (status != AMD_DBGAPI_STATUS_SUCCESS) {
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "Failed to get code object list: %d",
                status);
      return result;
    }

    DbgApiClientMemoryPtr<amd_dbgapi_code_object_id_t>
        code_objects_list_deleter(code_object_list);
    m_gpu_module_manager.BeginCodeObjectListUpdate();
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

      DbgApiClientMemoryPtr<char> uri_bytes_deleter(uri_bytes);
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "Code object %zu: %s at address %" PRIu64, i, uri_bytes,
                l_addr);

      m_gpu_module_manager.CodeObjectIsLoaded(uri_bytes, l_addr);
    }
    m_gpu_module_manager.EndCodeObjectListUpdate();
    break;
  }

  default:
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Unknown event kind: %d", eventKind);
    break;
  }

  if (eventKind != AMD_DBGAPI_EVENT_KIND_NONE)
    amd_dbgapi_event_processed(eventId);

  return result;
}

void ProcessAMDGPU::UpdateThreadListFromWaves() {
  std::vector<amd_dbgapi_wave_id_t> new_waves = UpdateWavesAndReturnNew();

  // Remove dead threads.
  m_threads.erase(
      std::remove_if(m_threads.begin(), m_threads.end(),
                     [this](const auto &t) {
                       ThreadAMDGPU &thread = static_cast<ThreadAMDGPU &>(*t);
                       if (thread.IsShadowThread())
                         return true;

                       return m_waves.count(thread.GetWaveID()) == 0;
                     }),
      m_threads.end());

  // Add new threads.
  for (auto wave_id : new_waves) {
    assert(m_waves.count(wave_id) && "New wave not in m_waves");
    m_waves.at(wave_id)->AddThreadsToList(*this, m_threads);
  }
}

template <typename T>
static llvm::Error QueryWaveInfo(amd_dbgapi_wave_id_t wave_id,
                                 amd_dbgapi_wave_info_t info_type, T *dest) {
  amd_dbgapi_status_t status =
      amd_dbgapi_wave_get_info(wave_id, info_type, sizeof(*dest), dest);
  if (status != AMD_DBGAPI_STATUS_SUCCESS) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to get %s for wave %" PRIu64 ": status=%s",
        AmdDbgApiWaveInfoKindToString(info_type), wave_id.handle,
        AmdDbgApiStatusToString(status));
  }
  return llvm::Error::success();
}

llvm::Expected<DbgApiWaveInfo>
ProcessAMDGPU::GetWaveInfo(amd_dbgapi_wave_id_t wave_id) {
  DbgApiWaveInfo wave_info;

  if (llvm::Error err =
          QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_STATE, &wave_info.state))
    return err;

  if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_WORKGROUP,
                                      &wave_info.workgroup_id))
    return err;

  if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_DISPATCH,
                                      &wave_info.dispatch_id))
    return err;

  if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_QUEUE,
                                      &wave_info.queue_id))
    return err;

  if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_AGENT,
                                      &wave_info.agent_id))
    return err;

  if (llvm::Error err =
          QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                        &wave_info.architecture_id))
    return err;

  if (llvm::Error err =
          QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD,
                        &wave_info.workgroup_coord))
    return err;

  if (llvm::Error err =
          QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
                        &wave_info.index_in_workgroup))
    return err;

  if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_LANE_COUNT,
                                      &wave_info.num_lanes_supported))
    return err;

  // Some information can only be queried if the wave is stopped.
  if (wave_info.state == AMD_DBGAPI_WAVE_STATE_STOP) {
    if (llvm::Error err = QueryWaveInfo(
            wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON, &wave_info.stop_reason))
      return err;

    if (llvm::Error err =
            QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_PC, &wave_info.pc))
      return err;

    if (llvm::Error err = QueryWaveInfo(wave_id, AMD_DBGAPI_WAVE_INFO_EXEC_MASK,
                                        &wave_info.exec_mask))
      return err;
  }

  return wave_info;
}

llvm::Expected<DbgApiClientMemoryPtr<amd_dbgapi_wave_id_t>>
ProcessAMDGPU::GetWaveList(size_t *count, amd_dbgapi_changed_t *changed) {
  // Re-enable wave creation on exit from this function.
  auto ResetWaveCreation = llvm::make_scope_exit([this] {
    if (auto error = RunAmdDbgApiCommand([this] {
          return amd_dbgapi_process_set_wave_creation(
              GetDbgApiProcessID(), AMD_DBGAPI_WAVE_CREATION_NORMAL);
        }))
      LLDB_LOG_ERROR(GetLog(GDBRLog::Plugin), std::move(error),
                     "Error: Failed to enable wave creation: {0}");
  });

  // Stop creating new waves get an accurate count.
  if (llvm::Error error = RunAmdDbgApiCommand([this] {
        return amd_dbgapi_process_set_wave_creation(
            GetDbgApiProcessID(), AMD_DBGAPI_WAVE_CREATION_STOP);
      }))
    return error;

  // Get the list of waves
  amd_dbgapi_wave_id_t *wave_list = nullptr;
  if (auto error = RunAmdDbgApiCommand([&] {
        return amd_dbgapi_process_wave_list(GetDbgApiProcessID(), count,
                                            &wave_list, changed);
      }))
    return error;

  return DbgApiClientMemoryPtr<amd_dbgapi_wave_id_t>(wave_list);
}

WaveIdList ProcessAMDGPU::UpdateWavesAndReturnNew() {
  Log *log = GetLog(GDBRLog::Plugin);

  // Get the list of waves
  size_t count = 0;
  amd_dbgapi_changed_t changed = AMD_DBGAPI_CHANGED_NO;
  auto maybe_wave_list = GetWaveList(&count, &changed);
  if (!maybe_wave_list) {
    LLDB_LOG_ERROR(log, maybe_wave_list.takeError(),
                   "Failed to get wave list: {0}");
    m_waves.clear();
    return {};
  }
  amd_dbgapi_wave_id_t *wave_list = maybe_wave_list->get();

  if (changed == AMD_DBGAPI_CHANGED_NO) {
    LLDB_LOGF(log, "No changes in wave list");
    return {};
  }

  // Update the info for our live waves.
  // Any waves that we fail to get info for are considered dead.
  // Keep track of which waves are new so we can return them to the caller.
  WaveIdSet live_waves;
  WaveIdList new_waves;
  for (size_t i = 0; i < count; ++i) {
    amd_dbgapi_wave_id_t wave_id = wave_list[i];

    if (llvm::Expected<DbgApiWaveInfo> wave_info = GetWaveInfo(wave_id)) {
      LLDB_LOGF(log, "Successfully retrieved wave info for wave: %" PRIu64,
                wave_id.handle);
      if (!m_waves.count(wave_id)) {
        LLDB_LOGF(log, "New wave: %" PRIu64, wave_id.handle);
        m_waves.emplace(wave_id, std::make_shared<WaveAMDGPU>(wave_id));
        new_waves.push_back(wave_id);
      }
      live_waves.insert(wave_id);
      m_waves.at(wave_id)->SetDbgApiInfo(*wave_info);
    } else {
      // We failed to get wave info for this wave.
      LLDB_LOG_ERROR(log, wave_info.takeError(),
                     "Failed to get wave info for wave {1}. {0}. Marking wave as dead.",
                     wave_id.handle);
    }
  }

  // Remove any waves from m_waves that are not in the live_waves set.
  // Removing from a std::unordered_map invalidates only the current iterator,
  // so we explicitly control the iterator opdate in this loop.
  for (auto it = m_waves.begin(); it != m_waves.end();) {
    if (live_waves.count(it->first)) {
      ++it;
    } else {
      LLDB_LOGF(log, "Removing dead wave: %" PRIu64, it->first.handle);
      it = m_waves.erase(it);
    }
  }

  return new_waves;
}
