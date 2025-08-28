//===-- ProcessAMDGPU.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessAMDGPU.h"
#include "ThreadAMDGPU.h"

#include "LLDBServerPluginAMDGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UnimplementedError.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <cinttypes>
#include <iostream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessAMDGPU::ProcessAMDGPU(lldb::pid_t pid, NativeDelegate &delegate,
                             LLDBServerPluginAMDGPU *plugin)
    : NativeProcessProtocol(pid, -1, delegate), m_debugger(plugin) {
  m_state = eStateStopped;
  UpdateThreads();
}

Status ProcessAMDGPU::Resume(const ResumeActionList &resume_actions) {
  SetState(StateType::eStateRunning, true);
  ThreadAMDGPU *thread = (ThreadAMDGPU *)GetCurrentThread();
  thread->GetRegisterContext().InvalidateAllRegisters();
  // if (!m_debugger->resume_process()) {
  //   return Status::FromErrorString("resume_process failed");
  // }
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

size_t ProcessAMDGPU::UpdateThreads() {
  if (m_threads.empty()) {
    lldb::tid_t tid = 3456;
    m_threads.push_back(std::make_unique<ThreadAMDGPU>(*this, 3456));
    // ThreadAMDGPU &thread = static_cast<ThreadAMDGPU &>(*m_threads.back());
    SetCurrentThreadID(tid);
  }
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

static std::optional<lldb_private::GPUDynamicLoaderLibraryInfo>
ParseLibraryInfo(const GpuModuleManager::CodeObject &code_object) {
  // This function will parse the shared library string that AMDs GPU driver
  // sends to the debugger. The format is one of:
  //  file://<path>#offset=<file-offset>&size=<file-size>
  //  memory://<name>#offset=<image-addr>&size=<image-size>
  lldb_private::GPUDynamicLoaderLibraryInfo lib_info;
  lib_info.load = code_object.IsLoaded();
  lib_info.load_address = code_object.load_address;

  auto get_offset_and_size = [](llvm::StringRef &values,
                                std::optional<uint64_t> &offset,
                                std::optional<uint64_t> &size) {
    offset = std::nullopt;
    size = std::nullopt;
    llvm::StringRef value;
    uint64_t uint_value;
    std::tie(value, values) = values.split('&');
    while (!value.empty()) {
      if (value.consume_front("offset=")) {
        if (!value.getAsInteger(0, uint_value))
          offset = uint_value;
      } else if (value.consume_front("size=")) {
        if (!value.getAsInteger(0, uint_value))
          size = uint_value;
      }
      std::tie(value, values) = values.split('&');
    }
  };

  llvm::StringRef lib_spec = code_object.uri;
  if (lib_spec.consume_front("file://")) {
    llvm::StringRef path, values;
    std::tie(path, values) = lib_spec.split('#');
    if (path.empty())
      return std::nullopt;
    lib_info.pathname = path.str();
    get_offset_and_size(values, lib_info.file_offset, lib_info.file_size);
  } else if (lib_spec.consume_front("memory://")) {
    llvm::StringRef name, values;
    std::tie(name, values) = lib_spec.split('#');
    if (name.empty())
      return std::nullopt;
    lib_info.pathname = name.str();
    get_offset_and_size(values, lib_info.native_memory_address,
                        lib_info.native_memory_size);
    // We must have a valid address and size for memory objects.
    if (!(lib_info.native_memory_address.has_value() &&
          lib_info.native_memory_size.has_value()))
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  return lib_info;
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
    if (auto lib_info = ParseLibraryInfo(code_object)) {
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
    // auto ip = getPC();
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
    // RemoveGPUBreakpoint(pc);
    // auto thread = std::make_unique<ThreadAMDGPU>(*this, wave_id.handle,
    // wave_id); thread->SetStopReason(lldb::eStopReasonBreakpoint);
    m_wave_ids.emplace_back(wave_id);

    if (m_threads.size() == 1 && m_gpu_state == State::Initializing) {
      m_threads.clear();
      SetCurrentThreadID(wave_id.handle);
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

static const char *event_kind_str(amd_dbgapi_event_kind_t kind) {
  switch (kind) {
  case AMD_DBGAPI_EVENT_KIND_NONE:
    return "NONE";

  case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
    return "WAVE_STOP";

  case AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED:
    return "WAVE_COMMAND_TERMINATED";

  case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
    return "CODE_OBJECT_LIST_UPDATED";

  case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
    return "BREAKPOINT_RESUME";

  case AMD_DBGAPI_EVENT_KIND_RUNTIME:
    return "RUNTIME";

  case AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR:
    return "QUEUE_ERROR";
  }
  assert(false && "unhandled amd_dbgapi_event_kind_t value");
}

bool ProcessAMDGPU::handleDebugEvent(amd_dbgapi_event_id_t eventId,
                                     amd_dbgapi_event_kind_t eventKind) {
  LLDB_LOGF(GetLog(GDBRLog::Plugin), "handleDebugEvent(%" PRIu64 ", %s)",
            eventId.handle, event_kind_str(eventKind));
  bool result = false;
  if (eventKind == AMD_DBGAPI_EVENT_KIND_NONE)
    return result;

  amd_dbgapi_runtime_state_t runtimeState = AMD_DBGAPI_RUNTIME_STATE_UNLOADED;

  // Get runtime state for the event
  amd_dbgapi_status_t status =
      amd_dbgapi_event_get_info(eventId, AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
                                sizeof(runtimeState), &runtimeState);

  if (status == AMD_DBGAPI_STATUS_SUCCESS) {
    // Handle different runtime states
    switch (runtimeState) {
    case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime loaded successfully");
      break;
    case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime load restricted");
      break;
    case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
      LLDB_LOGF(GetLog(GDBRLog::Plugin), "Runtime unloaded");
      break;
    }
  }

  // Handle event kinds
  switch (eventKind) {
  case AMD_DBGAPI_EVENT_KIND_WAVE_STOP: {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Wave stop event received");

    // Handle wave stop
    result = handleWaveStop(eventId);
    m_gpu_state = State::GPUStopped;
    break;
  }

    // case AMD_DBGAPI_EVENT_KIND_BREAKPOINT: {
    //   std::cout << "Breakpoint event received" << std::endl;

    //   // Get breakpoint information for this event
    //   amd_dbgapi_breakpoint_id_t breakpointId;
    //   if (amd_dbgapi_event_get_info(eventId,
    //   AMD_DBGAPI_EVENT_INFO_BREAKPOINT,
    //                                 sizeof(breakpointId), &breakpointId)
    //                                 ==
    //       AMD_DBGAPI_STATUS_SUCCESS) {
    //     std::cout << "Breakpoint ID: " << breakpointId << std::endl;
    //   }
    //   break;
    // }

  case AMD_DBGAPI_EVENT_KIND_RUNTIME: {
    LLDB_LOGF(GetLog(GDBRLog::Plugin),
              "Runtime event received, runtimeState: %d", runtimeState);

    // Additional runtime-specific handling based on state
    if (runtimeState == AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS) {
      // Runtime is now loaded, we can set breakpoints or perform other
      // initialization
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

      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "Code object %zu: %s at address %" PRIu64, i, uri_bytes,
                l_addr);

      m_gpu_module_manager.CodeObjectIsLoaded(uri_bytes, l_addr);
      m_debugger->FreeDbgApiClientMemory(uri_bytes);
    }
    m_gpu_module_manager.EndCodeObjectListUpdate();
    break;
  }

  default:
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Unknown event kind: %d", eventKind);
    break;
  }
  return result;
}

void ProcessAMDGPU::AddThread(amd_dbgapi_wave_id_t wave_id) {
  auto thread = std::make_unique<ThreadAMDGPU>(*this, wave_id.handle, wave_id);
  thread->SetStopReason(lldb::eStopReasonBreakpoint);
  m_threads.emplace_back(std::move(thread));
}
