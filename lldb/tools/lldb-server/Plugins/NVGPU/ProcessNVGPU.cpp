//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessNVGPU.h"
#include "../Utils/Utils.h"
#include "AddressSpaces.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ThreadNVGPU.h"
#include "cudadebugger.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/DataBufferLLVM.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessNVGPU::ProcessNVGPU(lldb::pid_t pid, NativeDelegate &delegate)
    : NativeProcessProtocol(pid, -1, delegate),
      m_arch(ArchSpec("nvptx-nvidia-cuda")), m_api(nullptr),
      m_fallback_thread(*this,
                        /*thread_state=*/nullptr, /*tid=*/1) {
  // A tid like -1 would be better, but that would make the first real thread to
  // have a thread id of #2 in the client because the fallback thread would have
  // a different internal id. Therefore, we keep tid=1 for the first HW thread
  // and the fallback thread to avoid confusions when debugging small kernels.

  // We need to initialize the state to stopped so that the client can connect
  // to the server. The gdb-remote protocol refuses to connect to running
  // targets.
  m_state = eStateStopped;

  // As part of connecting the client with the server, we need to set the
  // initial state to stopped, which requires sending some thread to the client.
  // Because of that, we create a fake thread with stopped state.
  m_fallback_thread.SetStoppedByInitialization();
  m_threads.push_back(
      std::unique_ptr<NativeThreadProtocol>(&m_fallback_thread));
  SetCurrentThreadID(m_fallback_thread.GetID());
}

Status ProcessNVGPU::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::Resume(). Pre-resume state: {}",
           StateToString(GetState()));

  if (m_is_faking_a_stop_for_dyld) {
    m_is_faking_a_stop_for_dyld = false;
    // Ack'ing here is fine because the next call to OnDebuggerAPIEvent will
    // be triggered after the Resume packet has been fully processed.
    CUDBGResult res = GetCudaAPI().acknowledgeSyncEvents();
    if (res != CUDBG_SUCCESS) {
      logAndReportFatalError(
          "Failed to acknowledge CUDA Debugger API events. {}",
          cudbgGetErrorString(res));
    }
  } else {
    for (DeviceState &device : m_devices.GetDevices()) {
      CUDBGResult res = GetCudaAPI().resumeDevice(device.GetDeviceId());
      if (res != CUDBG_SUCCESS)
        logAndReportFatalError("Failed to resume device: {}",
                               cudbgGetErrorString(res));
    }
  }

  for (ThreadNVGPU &thread : GPUThreads())
    thread.SetRunning();

  SetState(StateType::eStateRunning, true);

  return Status();
}

void ProcessNVGPU::SetDebuggerAPI(CUDADebuggerAPI &api) {
  Log *log = GetLog(GDBRLog::Plugin);
  m_api = api.GetRawAPI();
  m_devices = DeviceStateRegistry(*this);
  size_t max_num_threads = m_devices.GetMaxNumSupportedThreads();
  LLDB_LOG(log,
           "NVGPU::SetDebuggerAPI(). Reserving space for max num threads: {}",
           max_num_threads);
  m_threads.reserve(max_num_threads);
}

Status ProcessNVGPU::Halt() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::Halt(). Pre-halt state: {}", StateToString(GetState()));
  // According to Andrew, halting the devices takes ~0.2ms - ~10 ms.
  CUDBGResult res = GetCudaAPI().suspendDevice(/*device_id=*/0);

  Status status;
  if (res != CUDBG_SUCCESS) {
    std::string error_string =
        std::string("Failed to suspend device. ") + cudbgGetErrorString(res);
    LLDB_LOG(log, "NVGPU::Halt(). {}", error_string);
    status.FromErrorString(error_string.c_str());
  }
  return status;
}

void ProcessNVGPU::ChangeStateToStopped() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ChangeStateToStopped(). Pre-stop state: {}",
           StateToString(GetState()));

  for (ThreadNVGPU &thread : GPUThreads()) {
    if (StateIsRunningState(thread.GetState()))
      thread.SetStopped();
  }

  SetState(StateType::eStateStopped, true);
}

Status ProcessNVGPU::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

Status ProcessNVGPU::Signal(int signo) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessNVGPU::Interrupt() { return Status(); }

Status ProcessNVGPU::Kill() { return Status(); }

Status ProcessNVGPU::WriteMemory(lldb::addr_t addr, const void *buf,
                                 size_t size, size_t &bytes_written) {
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t ProcessNVGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t ProcessNVGPU::UpdateThreads() {
  // The NVGPU threads are always up to date with
  // respect to thread state and they keep the thread list populated properly.
  // All this method needs to do is return the thread count.
  return m_threads.size();
}

const ArchSpec &ProcessNVGPU::GetArchitecture() const { return m_arch; }

Status ProcessNVGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                   bool hardware) {
  // setBreakpoint doesn't return coherent errors, so we just ignore them.
  for (DeviceState &device : m_devices.GetDevices()) {
    CUDBGResult res = m_api->setBreakpoint(device.GetDeviceId(), addr);
    if (res != CUDBG_SUCCESS) {
      LLDB_LOG(GetLog(GDBRLog::Plugin),
               "NVGPU::SetBreakpoint(). Failed to set breakpoint on device "
               "{}: {}",
               device.GetDeviceId(), cudbgGetErrorString(res));
    }
  }

  return Status();
}

Status ProcessNVGPU::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  // unsetBreakpoint doesn't return coherent errors, so we just ignore them.
  for (DeviceState &device : m_devices.GetDevices()) {
    CUDBGResult res = m_api->unsetBreakpoint(device.GetDeviceId(), addr);
    if (res != CUDBG_SUCCESS) {
      LLDB_LOG(
          GetLog(GDBRLog::Plugin),
          "NVGPU::RemoveBreakpoint(). Failed to unset breakpoint on device "
          "{}: {}",
          device.GetDeviceId(), cudbgGetErrorString(res));
    }
  }

  return Status();
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessNVGPU::GetAuxvData() const {
  return nullptr;
}

Status ProcessNVGPU::GetLoadedModuleFileSpec(const char *module_path,
                                             FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessNVGPU::GetFileLoadAddress(const llvm::StringRef &file_name,
                                        lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}

void ProcessNVGPU::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<ProcessInfo &>(launch_info);
}

bool ProcessNVGPU::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessNVGPU::Manager::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = 1;
  auto gpu_up = std::make_unique<ProcessNVGPU>(pid, native_delegate);
  gpu_up->SetLaunchInfo(launch_info);
  return gpu_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessNVGPU::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}

/// Parse ELF sections from a cubin and extract load address information.
///
/// Analyzes the ELF structure of a CUDA binary (cubin) to extract information
/// about loaded sections, including their virtual addresses and names. Only
/// sections with non-zero addresses are included in the result.
///
/// \param[in] elf_buffer_ref
///     Memory buffer reference containing the ELF data to parse.
///
/// \return
///     Vector of GPUSectionInfo structures containing section details.
static std::vector<GPUSectionInfo>
GetLoadSectionsForCubin(const llvm::MemoryBufferRef &elf_buffer_ref) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::vector<GPUSectionInfo> loaded_sections;

  llvm::Expected<llvm::object::ELF64LEObjectFile> elf_or_err =
      llvm::object::ELF64LEObjectFile::create(elf_buffer_ref);
  if (!elf_or_err) {
    logAndReportFatalError("GetLoadSectionsForCubin(). Failed to parse ELF: {}",
                           llvm::toString(elf_or_err.takeError()));
  }

  const llvm::object::ELF64LEFile &elf_file = elf_or_err->getELFFile();
  llvm::Expected<llvm::object::ELF64LEFile::Elf_Shdr_Range> sections_or_err =
      elf_file.sections();
  if (!sections_or_err) {
    logAndReportFatalError(
        "GetLoadSectionsForCubin(). Failed to get sections: {}",
        llvm::toString(sections_or_err.takeError()));
  }

  LLDB_LOG(log, "GetLoadSectionsForCubin(). Iterating through ELF sections");

  for (const llvm::object::ELF64LEFile::Elf_Shdr &section : *sections_or_err) {
    llvm::Expected<llvm::StringRef> name_or_err =
        elf_file.getSectionName(section);

    if (!name_or_err) {
      LLDB_LOG(log, "GetLoadSectionsForCubin(). Failed to get section name: {}",
               llvm::toString(name_or_err.takeError()));
      continue;
    }

    // Sections with 0 as load address shouldn't be "loaded".
    if (section.sh_addr == 0)
      continue;

    // For NVIDIA cubin images, section virtual addresses are encoded as
    // absolute addresses
    LLDB_LOGV(log, "  Section: {}, Virtual Address: {1:x}, Size: {}",
              *name_or_err, section.sh_addr, section.sh_size);

    // Add the section to the loaded sections list
    GPUSectionInfo section_info;
    section_info.names.push_back(name_or_err->str());
    section_info.load_address = section.sh_addr;
    loaded_sections.push_back(section_info);
  }

  return loaded_sections;
}

void ProcessNVGPU::OnElfImageLoaded(
    const CUDBGEvent::cases_st::elfImageLoaded_st &event) {
  const auto &[dev_id, context_id, module_id, elf_image_size, handle,
               properties] = event;

  // Note that module_id and handle are the same thing. It is a vestige of the
  // older debug API.
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log,
           "LLDBServerPluginNVGPU::OnElfImageLoaded() dev_id: {}, context_id: "
           "{}, module_id: {}, elf_image_size: {}, handle: {}, properties: {}",
           dev_id, context_id, module_id, elf_image_size, handle, properties);

  // Obtain the elf image
  std::unique_ptr<llvm::WritableMemoryBuffer> data_buffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(elf_image_size);
  CUDBGResult res = GetCudaAPI().getElfImageByHandle(
      dev_id, handle, CUDBGElfImageType::CUDBG_ELF_IMAGE_TYPE_RELOCATED,
      data_buffer->getBufferStart(), elf_image_size);
  if (res != CUDBG_SUCCESS) {
    logAndReportFatalError(
        "NVGPU::OnElfImageLoaded(). Failed to get elf image: {}",
        cudbgGetErrorString(res));
    return;
  }

  // Create a MemoryBufferRef from the data buffer
  llvm::MemoryBufferRef elf_buffer_ref(*data_buffer);

  GPUDynamicLoaderLibraryInfo lib1;
  lib1.pathname = "cuda_elf_" + std::to_string(handle) + ".cubin";
  lib1.load = true;
  lib1.elf_image_base64_sp = std::make_shared<std::string>(
      llvm::encodeBase64(elf_buffer_ref.getBuffer()));
  // Parse ELF sections to extract load addresses
  lib1.loaded_sections = GetLoadSectionsForCubin(elf_buffer_ref);

  m_unreported_libraries.push_back(lib1);
  m_all_libraries.push_back(lib1);
}

bool ProcessNVGPU::HasUnreportedLibraries() const {
  return !m_unreported_libraries.empty();
}

void ProcessNVGPU::ReportDyldStop() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ReportDyldStop()");

  ThreadNVGPU &thread = *GPUThreads().begin();
  thread.SetStoppedByDynamicLoader();
  m_is_faking_a_stop_for_dyld = true;
  SetState(StateType::eStateStopped, true);
}

/// Release all pointers to ThreadNVGPU objects and clear the vector.
static void ReleaseAndClearThreads(
    std::vector<std::unique_ptr<NativeThreadProtocol>> &threads) {
  for (std::unique_ptr<NativeThreadProtocol> &thread : threads)
    thread.release();
  threads.clear();
}

void ProcessNVGPU::OnAllDevicesSuspended(
    const CUDBGEvent::cases_st::allDevicesSuspended_st &event,
    std::function<void(llvm::StringRef message)> log_to_client_callback) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::OnAllDevicesSuspended()");

  m_devices.BatchUpdate(log_to_client_callback);
  LLDB_LOG(log, "Device info dump:\n{}", m_devices.Dump());

  ReleaseAndClearThreads(m_threads);

  // We report as the selected thread the one the first one that has an
  // exception, or the first one that is at a breakpoint.
  std::optional<lldb::tid_t> exception_thread_id;
  std::optional<lldb::tid_t> breakpoint_thread_id;

  for (DeviceState &device : m_devices.GetDevices()) {
    for (SMState &sm : device.GetActiveSMs()) {
      for (WarpState &warp : sm.GetValidWarps()) {
        for (ThreadState &thread_state : warp.GetValidThreads()) {
          ThreadNVGPU &thread = thread_state.GetThreadNVGPU();

          StopReason stop_reason = thread.GetStopReason();
          if (stop_reason == lldb::eStopReasonBreakpoint &&
              !breakpoint_thread_id)
            breakpoint_thread_id = thread.GetID();
          else if (stop_reason == lldb::eStopReasonNone && !exception_thread_id)
            exception_thread_id = thread.GetID();

          /// The threads are owned by the DeviceStateRegistry, but we need to
          /// add them to the m_threads vector using a std::unique_ptr that
          /// we'll release later.
          m_threads.push_back(std::unique_ptr<NativeThreadProtocol>(&thread));
        }
      }
    }
  }

  // We need to report a fake thread in case no actual threads are found, as the
  // client doesn't support empty thread lists.
  if (m_threads.empty()) {
    m_fallback_thread.SetStopped(lldb::eStopReasonNone, "No threads found");
    m_threads.push_back(
        std::unique_ptr<NativeThreadProtocol>(&m_fallback_thread));
  }

  SetCurrentThreadID(exception_thread_id.value_or(
      breakpoint_thread_id.value_or(m_threads.front()->GetID())));
  ChangeStateToStopped();
}

ProcessNVGPU::Extension ProcessNVGPU::Manager::GetSupportedExtensions() const {
  return Extension::gpu_dyld | Extension::address_spaces;
}

std::optional<GPUDynamicLoaderResponse>
ProcessNVGPU::GetGPUDynamicLoaderLibraryInfos(
    const GPUDynamicLoaderArgs &args) {
  GPUDynamicLoaderResponse response;
  if (args.full) {
    response.library_infos = m_all_libraries;
  } else {
    response.library_infos = std::move(m_unreported_libraries);
    m_unreported_libraries.clear();
  }

  return response;
}

NVGPUThreadRange ProcessNVGPU::GPUThreads() {
  return NVGPUThreadRange(m_threads);
}

const CUDBGAPI_st &ProcessNVGPU::GetCudaAPI() {
  if (!m_api) {
    logAndReportFatalError("NVGPU::GetCudaAPI(). CUDA Debugger API is "
                           "not initialized");
  }
  return *m_api;
}

std::vector<AddressSpaceInfo> ProcessNVGPU::GetAddressSpaces() {
  std::vector<AddressSpaceInfo> result;
  // is_thread_specific should be true for all address spaces that may return a
  // different value for different threads.
  result.push_back(
      {"const", AddressSpace::ConstStorage, /*is_thread_specific=*/false});
  result.push_back(
      {"global", AddressSpace::GlobalStorage, /*is_thread_specific=*/false});
  result.push_back(
      {"local", AddressSpace::LocalStorage, /*is_thread_specific=*/true});
  result.push_back(
      {"param", AddressSpace::ParamStorage, /*is_thread_specific=*/true});
  result.push_back(
      {"shared", AddressSpace::SharedStorage, /*is_thread_specific=*/true});
  result.push_back(
      {"generic", AddressSpace::GenericStorage, /*is_thread_specific=*/true});
  return result;
}

Status ProcessNVGPU::ReadMemoryWithSpace(lldb::addr_t addr, uint64_t addr_space,
                                         NativeThreadProtocol *thread,
                                         void *buf, size_t size,
                                         size_t &bytes_readn) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ReadMemoryWithSpace(). addr: {}, size: {}", addr, size);

  auto GetPhysicalCoords = [&thread]() -> const ThreadCoords & {
    ThreadNVGPU &nv_thread = *static_cast<ThreadNVGPU *>(thread);
    const ThreadState *thread_state = nv_thread.GetThreadState();
    if (!thread_state)
      logAndReportFatalError(
          "NVGPU::ReadMemoryWithSpace(). ThreadState is null");
    return thread_state->GetCoords();
  };
  CUDBGResult res;

  switch (addr_space) {
  case AddressSpace::ConstStorage:
  case AddressSpace::GlobalStorage: {
    // Const storage can be read as global storage.
    res = GetCudaAPI().readGlobalMemory(addr, buf, size);
    break;
  }
  case AddressSpace::LocalStorage: {
    const ThreadCoords &coords = GetPhysicalCoords();
    res = GetCudaAPI().readLocalMemory(coords.dev_id, coords.sm_id,
                                       coords.warp_id, coords.thread_id, addr,
                                       buf, size);
    break;
  }
  case AddressSpace::ParamStorage: {
    const ThreadCoords &coords = GetPhysicalCoords();
    res = GetCudaAPI().readParamMemory(coords.dev_id, coords.sm_id,
                                       coords.warp_id, addr, buf, size);
    break;
  }
  case AddressSpace::SharedStorage: {
    const ThreadCoords &coords = GetPhysicalCoords();
    res = GetCudaAPI().readSharedMemory(coords.dev_id, coords.sm_id,
                                        coords.warp_id, addr, buf, size);
    break;
  }
  case AddressSpace::GenericStorage: {
    const ThreadCoords &coords = GetPhysicalCoords();
    res = GetCudaAPI().readGenericMemory(coords.dev_id, coords.sm_id,
                                         coords.warp_id, coords.thread_id, addr,
                                         buf, size);
    break;
  }
  default:
    return Status::FromErrorStringWithFormatv("Invalid address space '{}'",
                                              addr_space);
  }

  if (res != CUDBG_SUCCESS) {
    bytes_readn = 0;
    return Status::FromErrorString(cudbgGetErrorString(res));
  }

  bytes_readn = size;
  return Status();
}

Status ProcessNVGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                size_t &bytes_read) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ReadMemory(). addr: {}, size: {}", addr, size);
  return ReadMemoryWithSpace(addr, AddressSpace::GlobalStorage,
                             /*thread=*/nullptr, buf, size, bytes_read);
}

void ProcessNVGPU::OnNativeProcessExit(const WaitStatus &exit_status) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::OnNativeProcessExit(). exit_status: {}", exit_status);
  // Set our exit status to match the native process and notify delegates.
  SetExitStatus(exit_status, /*bNotifyStateChange=*/true);
}

std::vector<std::string> ProcessNVGPU::GetStructuredDataPlugins() {
  return {"nvgpu-monitor"};
}
