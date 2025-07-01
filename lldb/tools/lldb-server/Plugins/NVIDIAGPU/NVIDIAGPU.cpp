//===-- NVIDIAGPU.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVIDIAGPU.h"
#include "ThreadNVIDIAGPU.h"
#include "Utils.h"

#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "cudadebugger.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataBufferLLVM.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

/// Helper class to provide range-based iteration over
/// std::vector<std::unique_ptr<NativeThreadProtocol>> with automatic casting to
/// ThreadNVIDIAGPU&.
class NVIDIAGPU::GPUThreadRange {
public:
  class iterator {
  public:
    iterator(std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator it)
        : m_it(it) {}

    ThreadNVIDIAGPU &operator*() const {
      return static_cast<ThreadNVIDIAGPU &>(**m_it);
    }

    iterator &operator++() {
      ++m_it;
      return *this;
    }

    bool operator!=(const iterator &other) const { return m_it != other.m_it; }

  private:
    std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator m_it;
  };

  GPUThreadRange(std::vector<std::unique_ptr<NativeThreadProtocol>> &threads)
      : m_threads(threads) {}

  iterator begin() { return iterator(m_threads.begin()); }
  iterator end() { return iterator(m_threads.end()); }

private:
  std::vector<std::unique_ptr<NativeThreadProtocol>> &m_threads;
};

NVIDIAGPU::NVIDIAGPU(lldb::pid_t pid, NativeDelegate &delegate)
    : NativeProcessProtocol(pid, -1, delegate),
      m_arch(ArchSpec("nvptx-nvidia-cuda")), m_api(nullptr) {
  // We need to initialize the state to stopped so that the client can connect
  // to the server. The gdb-remote protocol refuses to connect to running
  // targets.
  m_state = eStateStopped;
  UpdateThreads();
}

Status NVIDIAGPU::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::Resume(). Pre-resume state: {0}",
           StateToString(GetState()));

  // According to Andrew, resume device takes ~25 ms.
  CUDBGResult res = m_api->resumeDevice(/*device_id=*/0);

  Status status;
  if (res != CUDBG_SUCCESS) {
    LLDB_LOG(log, "NVIDIAGPU::Resume(). Failed to resume device: {0}", res);
    return Status::FromErrorString("Failed to resume device");
  }

  for (ThreadNVIDIAGPU &thread : GPUThreads())
    thread.SetRunning();
  SetState(StateType::eStateRunning, true);

  return Status();
}

Status NVIDIAGPU::Halt() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::Halt(). Pre-halt state: {0}",
           StateToString(GetState()));
  // According to Andrew, halting the devices takes ~0.2ms - ~10 ms.
  CUDBGResult res = m_api->suspendDevice(/*device_id=*/0);

  Status status;
  if (res != CUDBG_SUCCESS) {
    std::string error_string =
        std::string("Failed to suspend device. ") + cudbgGetErrorString(res);
    LLDB_LOG(log, "NVIDIAGPU::Halt(). {0}", error_string);
    status.FromErrorString(error_string.c_str());
  }
  return status;
}

Status NVIDIAGPU::HaltDueToDyld() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::HaltDueToDyld(). Pre-halt state: {0}",
           StateToString(GetState()));

  Status status = Halt();
  if (status.Fail())
    return status;

  ThreadNVIDIAGPU &thread = *GPUThreads().begin();
  thread.SetStoppedByDynamicLoader();
  return Status();
}

void NVIDIAGPU::ChangeStateToStopped() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::ChangeStateToStopped(). Pre-stop state: {0}",
           StateToString(GetState()));

  for (ThreadNVIDIAGPU &thread : GPUThreads()) {
    if (StateIsRunningState(thread.GetState()))
      thread.SetStopped();
  }

  SetState(StateType::eStateStopped, true);
}

Status NVIDIAGPU::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

Status NVIDIAGPU::Signal(int signo) {
  return Status::FromErrorString("unimplemented");
}

Status NVIDIAGPU::Interrupt() { return Status(); }

Status NVIDIAGPU::Kill() { return Status(); }

Status NVIDIAGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                             size_t &bytes_read) {
  return Status::FromErrorString("unimplemented");
}

Status NVIDIAGPU::WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                              size_t &bytes_written) {
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t NVIDIAGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t NVIDIAGPU::UpdateThreads() {
  // As part of connecting the client with the server, we need to set the
  // initial state to stopped, which requires sending some thread to the client.
  // Because of that, we create a fake thread with statopped state.
  // We also use this thread to send dyld events before the kernel even runs.
  if (m_threads.empty()) {
    lldb::tid_t tid = 1;
    auto thread = std::make_unique<ThreadNVIDIAGPU>(
        *this, tid, ThreadNVIDIAGPU::PhysicalCoords{});
    thread->SetStoppedByThreadlessState();
    m_threads.push_back(std::move(thread));
    SetCurrentThreadID(tid);
  }
  return m_threads.size();
}

const ArchSpec &NVIDIAGPU::GetArchitecture() const { return m_arch; }

// Breakpoint functions
Status NVIDIAGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                bool hardware) {
  return Status::FromErrorString("unimplemented");
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
NVIDIAGPU::GetAuxvData() const {
  return nullptr;
}

Status NVIDIAGPU::GetLoadedModuleFileSpec(const char *module_path,
                                          FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status NVIDIAGPU::GetFileLoadAddress(const llvm::StringRef &file_name,
                                     lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}

void NVIDIAGPU::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<ProcessInfo &>(launch_info);
}

bool NVIDIAGPU::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NVIDIAGPU::Manager::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = 1;
  auto gpu_up = std::make_unique<NVIDIAGPU>(pid, native_delegate);
  gpu_up->SetLaunchInfo(launch_info);
  return gpu_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NVIDIAGPU::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}

void NVIDIAGPU::OnElfImageLoaded(
    const CUDBGEvent::cases_st::elfImageLoaded_st &event) {
  const auto &[dev_id, context_id, module_id, elf_image_size, handle,
               properties] = event;

  // Note that module_id and handle are the same thing. It is a vestige of the
  // older debug API.
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(
      log,
      "LLDBServerPluginNVIDIAGPU::OnElfImageLoaded() dev_id: {0}, context_id: "
      "{1}, module_id: {2}, elf_image_size: {3}, handle: {4}, properties: {5}",
      dev_id, context_id, module_id, elf_image_size, handle, properties);

  // Obtain the elf image
  auto data_buffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(elf_image_size);
  CUDBGResult res = m_api->getElfImageByHandle(
      dev_id, handle, CUDBGElfImageType::CUDBG_ELF_IMAGE_TYPE_RELOCATED,
      data_buffer->getBufferStart(), elf_image_size);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::OnElfImageLoaded(). Failed to get elf image: {0}",
        cudbgGetErrorString(res));
    return;
  }

  // Encode the ELF image data as Base64 for JSON storage
  llvm::StringRef elf_data_ref(data_buffer->getBufferStart(), elf_image_size);
  std::shared_ptr<std::string> elf_base64 =
      std::make_shared<std::string>(llvm::encodeBase64(elf_data_ref));

  GPUDynamicLoaderLibraryInfo lib1;
  lib1.pathname = "cuda_elf_" + std::to_string(handle) + ".cubin";
  lib1.load = true;
  lib1.elf_image_base64_sp = elf_base64;

  m_unreported_libraries.push_back(lib1);
  m_all_libraries.push_back(lib1);
}

bool NVIDIAGPU::HasUnreportedLibraries() const {
  return !m_unreported_libraries.empty();
}

void NVIDIAGPU::OnAllDevicesSuspended(
    const CUDBGEvent::cases_st::allDevicesSuspended_st &event) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::OnAllDevicesSuspended()");
  // The following code path assumes that the stop event is triggered by an
  // exception. At some point we'll need to generalize this.

  // Find the thread that caused the exception.
  ThreadNVIDIAGPU::PhysicalCoords physical_coords;
  const uint32_t dev_id = 0;
  uint32_t num_sms;
  CUDBGResult res = m_api->getNumSMs(dev_id, &num_sms);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::OnAllDevicesSuspended(). Failed to get number of SMs: {0}",
        cudbgGetErrorString(res));
    return;
  }

  std::vector<uint64_t> sm_exceptions(num_sms / 64 + 1, 0);
  res = m_api->readDeviceExceptionState(dev_id, sm_exceptions.data(),
                                        sm_exceptions.size());
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::OnAllDevicesSuspended(). Failed to read device "
        "exception state: {0}",
        cudbgGetErrorString(res));
    return;
  }

  // Find the first SM with an exception
  uint32_t sm_id = UINT32_MAX;
  for (uint32_t i = 0; i < num_sms; ++i) {
    if (sm_exceptions[i / 64] & (1ULL << (i % 64))) {
      sm_id = i;
      break;
    }
  }
  if (sm_id == UINT32_MAX) {
    logAndReportFatalError(
        "NVIDIAGPU::OnAllDevicesSuspended(). No SMs with exceptions found");
    return;
  }

  // Find the first warp with an exception
  uint32_t num_warps;
  res = m_api->getNumWarps(dev_id, &num_warps);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError("NVIDIAGPU::OnAllDevicesSuspended(). Failed to get "
                           "number of warps: {0}",
                           cudbgGetErrorString(res));
    return;
  }

  uint64_t valid_warps_mask;
  res = m_api->readValidWarps(dev_id, sm_id, &valid_warps_mask);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVIDIAGPU::OnAllDevicesSuspended(). Failed to read valid warps: {0}",
        cudbgGetErrorString(res));
    return;
  }

  for (uint32_t wp = 0; wp < num_warps; ++wp) {
    if (!(valid_warps_mask & (1ULL << wp)))
      continue;

    CUDBGWarpState warp;
    res = m_api->readWarpState(dev_id, sm_id, wp, &warp);
    if (res != CUDBG_SUCCESS) {
      logAndReportFatalError(
          "NVIDIAGPU::OnAllDevicesSuspended(). Failed to read warp state: {0}",
          cudbgGetErrorString(res));
      return;
    }

    if (!warp.validLanes)
      continue;

    for (uint32_t ln = 0; ln < 32; ++ln) {
      if (warp.validLanes & (1 << ln)) {
        CUDBGException_t exception = CUDBGException_t::CUDBG_EXCEPTION_NONE;
        res = m_api->readLaneException(dev_id, sm_id, wp, ln, &exception);
        if (res != CUDBG_SUCCESS) {
          logAndReportFatalError(
              "NVIDIAGPU::OnAllDevicesSuspended(). Failed to read lane "
              "exception: {0}",
              cudbgGetErrorString(res));
          continue;
        }
        if (exception != CUDBGException_t::CUDBG_EXCEPTION_NONE) {
          physical_coords =
              ThreadNVIDIAGPU::PhysicalCoords(dev_id, sm_id, wp, ln);
          LLDB_LOG(log, "Exception: {0}", exception);
          goto found;
        }
      }
    }
  }
  logAndReportFatalError(
      "NVIDIAGPU::OnAllDevicesSuspended(). No lanes with exceptions found");

found:
  LLDB_LOG(log, "Exception found at dev_id: {0}, sm_id: {1}, wp: {2}, ln: {3}",
           physical_coords.dev_id, physical_coords.sm_id,
           physical_coords.warp_id, physical_coords.lane_id);

  // We don't yet handle multiple threads, so we use the only one we have.

  ThreadNVIDIAGPU &thread = *GPUThreads().begin();
  thread.SetPhysicalCoords(physical_coords);
  thread.SetStoppedByException();
  ChangeStateToStopped();
}

NVIDIAGPU::Extension NVIDIAGPU::Manager::GetSupportedExtensions() const {
  return Extension::gpu_dyld;
}

std::optional<GPUDynamicLoaderResponse>
NVIDIAGPU::GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) {
  GPUDynamicLoaderResponse response;
  if (args.full) {
    response.library_infos = m_all_libraries;
  } else {
    response.library_infos = std::move(m_unreported_libraries);
    m_unreported_libraries.clear();
  }

  return response;
}

NVIDIAGPU::GPUThreadRange NVIDIAGPU::GPUThreads() {
  return GPUThreadRange(m_threads);
}