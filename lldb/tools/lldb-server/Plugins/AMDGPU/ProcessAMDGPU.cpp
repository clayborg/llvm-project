//===-- ProcessAMDGPU.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessAMDGPU.h"
#include "AmdDbgApiHelpers.h"
#include "Plugins/Utils/Utils.h"
#include "ThreadAMDGPU.h"

#include "LLDBServerPluginAMDGPU.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/AmdGpuAddressSpaces.h"
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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessAMDGPU::ProcessAMDGPU(lldb::pid_t pid, NativeDelegate &delegate,
                             LLDBServerPluginAMDGPU *plugin,
                             amd_dbgapi_architecture_id_t architecture_id)
    : NativeProcessProtocol(pid, -1, delegate), m_debugger(plugin),
      m_architecture_id(architecture_id) {
  m_state = eStateStopped;
}

Status ProcessAMDGPU::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "pid {0}", GetID());

  NotifyTracersProcessWillResume();

  struct WaveResumeAction {
    lldb::StateType resume_state = eStateInvalid;
    std::vector<ThreadAMDGPU *> lanes;
  };

  std::unordered_map<WaveAMDGPU *, WaveResumeAction> wave_actions;
  wave_actions.reserve(m_threads.size());
  bool had_resume_action = false;

  // LLDB models every GPU lane as a thread because that is the granularity the
  // user selects, inspects, and steps in the UI. The dbgapi control operation
  // is coarser: a resume request applies to the whole wave. Build one physical
  // resume request per wave from the logical lane requests before calling into
  // dbgapi.
  for (const auto &thread : m_threads) {
    assert(thread && "thread list should not contain NULL threads");
    ThreadAMDGPU *gpu_thread = static_cast<ThreadAMDGPU *>(thread.get());

    const ResumeAction *const action =
        resume_actions.GetActionForThread(thread->GetID(), true);

    // The shadow thread is LLDB's placeholder before real GPU lanes are
    // materialized. It has no backing wave to resume, but a resume action for
    // it still means the GPU process should transition to running below.
    if (gpu_thread->IsShadowThread()) {
      had_resume_action = action && (action->state == eStateRunning ||
                                     action->state == eStateStepping);
      continue;
    }

    WaveResumeAction &wave_action = wave_actions[gpu_thread->GetWave()];
    wave_action.lanes.push_back(gpu_thread);

    if (action == nullptr) {
      LLDB_LOG(log, "no action specified for pid {0} tid {1}", GetID(),
               thread->GetID());
      continue;
    }

    LLDB_LOG(log, "processing resume action state {0} for pid {1} tid {2}",
             action->state, GetID(), thread->GetID());

    switch (action->state) {
    case eStateStepping:
    case eStateRunning: {
      had_resume_action = true;

      // Several lane threads can map to the same wave. If LLDB asks to step one
      // lane while continuing another lane in that wave, the physical operation
      // must be single-step: issuing a normal resume would overrun the user's
      // selected step.
      if (wave_action.resume_state == eStateInvalid) {
        wave_action.resume_state = action->state;
      } else if (wave_action.resume_state != action->state) {
        LLDB_LOG(log,
                 "coalescing conflicting resume actions for wave {0}: "
                 "existing state {1}, new state {2}",
                 gpu_thread->GetDbgApiWaveID().handle,
                 StateAsCString(wave_action.resume_state),
                 StateAsCString(action->state));

        if (action->state == eStateStepping)
          wave_action.resume_state = eStateStepping;
      }
      break;
    }

    case eStateSuspended:
    case eStateStopped:
      // These lane states do not request execution. Other lanes in the same
      // wave may still cause the wave to resume.
      break;

    default:
      return Status::FromErrorStringWithFormat(
          "NativeProcessLinux::%s (): unexpected state %s specified "
          "for pid %" PRIu64 ", tid %" PRIu64,
          __FUNCTION__, StateAsCString(action->state), GetID(),
          thread->GetID());
    }
  }

  // Apply the coalesced actions to each physical wave once. Register values for
  // all lanes in a resumed wave are invalidated because any lane-visible state
  // may have changed after the wave executes.
  for (auto &item : wave_actions) {
    WaveAMDGPU *wave = item.first;
    WaveResumeAction &wave_action = item.second;
    if (wave_action.resume_state != eStateStepping &&
        wave_action.resume_state != eStateRunning)
      continue;

    const bool single_step = wave_action.resume_state == eStateStepping;
    Status resume_result = wave->Resume(single_step);
    if (resume_result.Fail())
      return Status::FromErrorStringWithFormat(
          "NativeProcessLinux::%s: failed to resume wave "
          "for pid %" PRIu64 ", tid %" PRIu64 ", error = %s",
          __FUNCTION__, GetID(), wave->GetWaveID().handle,
          resume_result.AsCString());

    for (ThreadAMDGPU *lane : wave_action.lanes)
      lane->GetRegisterContext().InvalidateAllRegisters();
  }

  if (had_resume_action)
    SetState(StateType::eStateRunning, true);
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
  NativeProcessProtocol *native_process = m_debugger->GetNativeProcess();
  if (!native_process)
    return Status::FromErrorString("native process is unavailable");
  return native_process->ReadMemory(addr, buf, size, bytes_read);
}

Status ProcessAMDGPU::WriteMemory(lldb::addr_t addr, const void *buf,
                                  size_t size, size_t &bytes_written) {
  NativeProcessProtocol *native_process = m_debugger->GetNativeProcess();
  if (!native_process)
    return Status::FromErrorString("native process is unavailable");
  return native_process->WriteMemory(addr, buf, size, bytes_written);
}

std::vector<AddressSpaceInfo> ProcessAMDGPU::GetAddressSpaces() {
  std::vector<AddressSpaceInfo> address_spaces;

  address_spaces.push_back({"generic", (uint64_t)DW_ASPACE_AMDGPU::generic,
                            /*is_thread_specific=*/true});
  address_spaces.push_back({"region", (uint64_t)DW_ASPACE_AMDGPU::region,
                            /*is_thread_specific=*/false});
  address_spaces.push_back({"local", (uint64_t)DW_ASPACE_AMDGPU::local,
                            /*is_thread_specific=*/true});
  address_spaces.push_back({"private_lane",
                            (uint64_t)DW_ASPACE_AMDGPU::private_lane,
                            /*is_thread_specific=*/true});
  address_spaces.push_back({"private_wave",
                            (uint64_t)DW_ASPACE_AMDGPU::private_wave,
                            /*is_thread_specific=*/true});

  return address_spaces;
}

Status ProcessAMDGPU::ReadMemoryWithSpace(lldb::addr_t addr,
                                          uint64_t addr_space,
                                          NativeThreadProtocol *thread,
                                          void *buf, size_t size,
                                          size_t &bytes_readn) {
  // Convert dwarf address space to debug api address space.
  amd_dbgapi_address_space_id_t address_space_id;
  if (llvm::Error err = RunAmdDbgApiCommand([&] {
        return amd_dbgapi_dwarf_address_space_to_address_space(
            m_architecture_id, addr_space, &address_space_id);
      }))
    return Status::FromError(std::move(err));

  // Grab the process, wave, and lane ids if available.
  ThreadAMDGPU *amd_thread = static_cast<ThreadAMDGPU *>(thread);
  amd_dbgapi_process_id_t process_id = GetDbgApiProcessID();
  amd_dbgapi_wave_id_t wave_id =
      amd_thread ? amd_thread->GetDbgApiWaveID() : AMD_DBGAPI_WAVE_NONE;
  amd_dbgapi_lane_id_t lane_id =
      amd_thread ? amd_thread->GetDbgApiLaneID() : AMD_DBGAPI_LANE_NONE;

  // Do the actual read memory.
  if (llvm::Error err = RunAmdDbgApiCommand([&] {
        return amd_dbgapi_read_memory(process_id, wave_id, lane_id,
                                      address_space_id, addr, &size, buf);
      }))
    return Status::FromError(std::move(err));

  bytes_readn = size;
  return Status();
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
      m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
      sizeof(cpu_subtype), &cpu_subtype);
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
  if (hardware)
    return Status::FromErrorString("hardware breakpoints are not supported");

  return SetSoftwareBreakpoint(addr, size);
}

Status ProcessAMDGPU::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  return NativeProcessProtocol::RemoveBreakpoint(addr, hardware);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
ProcessAMDGPU::GetSoftwareBreakpointTrapOpcode(size_t) {
  if (!m_breakpoint_trap_opcode.empty())
    return llvm::ArrayRef<uint8_t>(m_breakpoint_trap_opcode);

  const uint8_t *bp_instruction = nullptr;
  amd_dbgapi_status_t status = amd_dbgapi_architecture_get_info(
      m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
      sizeof(bp_instruction), &bp_instruction);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(
        "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION failed");

  size_t bp_size = 0;
  status = amd_dbgapi_architecture_get_info(
      m_architecture_id,
      AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE, sizeof(bp_size),
      &bp_size);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return llvm::createStringError(
        "AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE failed");

  m_breakpoint_trap_opcode.assign(bp_instruction, bp_instruction + bp_size);
  return llvm::ArrayRef<uint8_t>(m_breakpoint_trap_opcode);
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
  auto proc_up = std::make_unique<ProcessAMDGPU>(pid, native_delegate,
                                                 m_debugger, m_architecture_id);
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
    llvm::Expected<llvm::ArrayRef<uint8_t>> breakpoint_opcode =
        GetSoftwareBreakpointTrapOpcode(/*size_hint=*/0);
    if (!breakpoint_opcode) {
      std::string error = llvm::toString(breakpoint_opcode.takeError());
      LLDB_LOGF(GetLog(GDBRLog::Plugin),
                "GetSoftwareBreakpointTrapOpcode failed: %s", error.c_str());
      exit(-1);
    }
    pc -= breakpoint_opcode->size();
    amd_dbgapi_register_id_t pc_register_id;
    status = amd_dbgapi_architecture_get_info(
        m_architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
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
  } else {
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "Wave stopped due to reason: %d",
              stop_reason);
  }

  GetOrCreateWave(wave_id).UpdateStopReason(stop_reason);
  return true;
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
      case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION: {
        uint32_t major, minor, patch;
        amd_dbgapi_get_version(&major, &minor, &patch);
        std::string err_msg = llvm::formatv(
            "error: AMD GPU runtime version is not supported by the installed "
            "ROCm debug API (librocm-dbgapi v{0}.{1}.{2}). GPU debugging "
            "cannot proceed. Please upgrade ROCm.\n",
            major, minor, patch);
        LLDB_LOGF(GetLog(GDBRLog::Plugin), "%s", err_msg.c_str());
        // Send the error to the LLDB client via the CPU GDB server connection
        // so the user sees it in their console.
        m_debugger->SendErrorToClient(err_msg);
        // Mark the plugin as errored so it won't try to set up a GPU
        // connection or do further GPU debugging operations.
        m_debugger->SetErrorState();
        break;
      }
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
  UpdateWaveList();

  // Remove dead threads.
  m_threads.erase(
      std::remove_if(m_threads.begin(), m_threads.end(),
                     [this](const auto &t) {
                       ThreadAMDGPU &thread = static_cast<ThreadAMDGPU &>(*t);
                       if (thread.IsShadowThread())
                         return true;

                       return m_waves.count(thread.GetDbgApiWaveID()) == 0;
                     }),
      m_threads.end());

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

WaveAMDGPU &ProcessAMDGPU::GetOrCreateWave(amd_dbgapi_wave_id_t wave_id) {
  auto [it, inserted] =
      m_waves.try_emplace(wave_id, std::make_shared<WaveAMDGPU>(wave_id));
  if (inserted)
    LLDB_LOGF(GetLog(GDBRLog::Plugin), "New wave: %" PRIu64, wave_id.handle);
  return *it->second;
}

void ProcessAMDGPU::UpdateWaveList() {
  Log *log = GetLog(GDBRLog::Plugin);

  // Get the list of waves
  size_t count = 0;
  amd_dbgapi_changed_t changed = AMD_DBGAPI_CHANGED_NO;
  auto maybe_wave_list = GetWaveList(&count, &changed);
  if (!maybe_wave_list) {
    LLDB_LOG_ERROR(log, maybe_wave_list.takeError(),
                   "Failed to get wave list: {0}");
    m_waves.clear();
    return;
  }
  amd_dbgapi_wave_id_t *wave_list = maybe_wave_list->get();

  if (changed == AMD_DBGAPI_CHANGED_NO) {
    LLDB_LOGF(log, "No changes in wave list");
    return;
  }

  WaveIdSet waves_with_threads;
  for (const auto &thread : m_threads) {
    ThreadAMDGPU &gpu_thread = static_cast<ThreadAMDGPU &>(*thread);
    if (!gpu_thread.IsShadowThread())
      waves_with_threads.insert(gpu_thread.GetWaveID());
  }

  // Reconcile wave lifetime and cached metadata. Stop reasons are updated only
  // while processing the corresponding WAVE_STOP event.
  WaveIdSet live_waves;
  for (size_t i = 0; i < count; ++i) {
    amd_dbgapi_wave_id_t wave_id = wave_list[i];
    live_waves.insert(wave_id);
    WaveAMDGPU &wave = GetOrCreateWave(wave_id);

    if (llvm::Expected<DbgApiWaveInfo> wave_info = GetWaveInfo(wave_id)) {
      wave.SetDbgApiInfo(*wave_info);
      if (!waves_with_threads.count(wave_id)) {
        wave.AddThreadsToList(*this, m_threads);
        waves_with_threads.insert(wave_id);
      }
    } else {
      LLDB_LOG_ERROR(log, wave_info.takeError(),
                     "Failed to get wave info for wave {1}. {0}",
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
}
