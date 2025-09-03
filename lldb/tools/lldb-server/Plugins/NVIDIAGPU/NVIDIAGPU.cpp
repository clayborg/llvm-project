//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVIDIAGPU.h"
#include "../Utils/Utils.h"
#include "AddressSpaces.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ThreadNVIDIAGPU.h"
#include "cudadebugger.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/DataBufferLLVM.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
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
  /// Iterator class that automatically casts to ThreadNVIDIAGPU&.
  class iterator {
  public:
    /// Constructor for the iterator.
    ///
    /// \param[in] it
    ///     Iterator to the underlying thread container.
    iterator(std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator it)
        : m_it(it) {}

    /// Dereference operator with automatic casting to ThreadNVIDIAGPU&.
    ///
    /// \return
    ///     Reference to ThreadNVIDIAGPU object.
    ThreadNVIDIAGPU &operator*() const {
      return static_cast<ThreadNVIDIAGPU &>(**m_it);
    }

    /// Pre-increment operator.
    ///
    /// \return
    ///     Reference to incremented iterator.
    iterator &operator++() {
      ++m_it;
      return *this;
    }

    /// Equality comparison operator.
    ///
    /// \param[in] other
    ///     Iterator to compare against.
    ///
    /// \return
    ///     True if iterators are equal, false otherwise.
    bool operator!=(const iterator &other) const { return m_it != other.m_it; }

  private:
    std::vector<std::unique_ptr<NativeThreadProtocol>>::iterator m_it;
  };

  /// Constructor for the range object.
  ///
  /// \param[in] threads
  ///     Reference to the thread container.
  GPUThreadRange(std::vector<std::unique_ptr<NativeThreadProtocol>> &threads)
      : m_threads(threads) {}

  /// Get iterator to the beginning of the range.
  ///
  /// \return
  ///     Iterator pointing to the first element.
  iterator begin() { return iterator(m_threads.begin()); }

  /// Get iterator to the end of the range.
  ///
  /// \return
  ///     Iterator pointing past the last element.
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

  // As part of connecting the client with the server, we need to set the
  // initial state to stopped, which requires sending some thread to the client.
  // Because of that, we create a fake thread with stopped state.
  lldb::tid_t tid = 1;
  auto thread = std::make_unique<ThreadNVIDIAGPU>(*this, tid, PhysicalCoords{});
  thread->SetStoppedByInitialization();
  m_threads.push_back(std::move(thread));
  SetCurrentThreadID(tid);
}

Status NVIDIAGPU::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::Resume(). Pre-resume state: {}",
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
    // According to Andrew, resume device takes ~25 ms.
    CUDBGResult res = GetCudaAPI().resumeDevice(/*device_id=*/0);

    Status status;
    if (res != CUDBG_SUCCESS) {
      LLDB_LOG(log, "NVIDIAGPU::Resume(). Failed to resume device: {}", res);
      return Status::FromErrorString("Failed to resume device");
    }
  }

  for (ThreadNVIDIAGPU &thread : GPUThreads())
    thread.SetRunning();

  SetState(StateType::eStateRunning, true);

  return Status();
}

Status NVIDIAGPU::Halt() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::Halt(). Pre-halt state: {}",
           StateToString(GetState()));
  // According to Andrew, halting the devices takes ~0.2ms - ~10 ms.
  CUDBGResult res = GetCudaAPI().suspendDevice(/*device_id=*/0);

  Status status;
  if (res != CUDBG_SUCCESS) {
    std::string error_string =
        std::string("Failed to suspend device. ") + cudbgGetErrorString(res);
    LLDB_LOG(log, "NVIDIAGPU::Halt(). {}", error_string);
    status.FromErrorString(error_string.c_str());
  }
  return status;
}

void NVIDIAGPU::ChangeStateToStopped() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::ChangeStateToStopped(). Pre-stop state: {}",
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

Status NVIDIAGPU::WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                              size_t &bytes_written) {
  return Status::FromErrorString("unimplemented");
}

lldb::addr_t NVIDIAGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t NVIDIAGPU::UpdateThreads() { return m_threads.size(); }

const ArchSpec &NVIDIAGPU::GetArchitecture() const { return m_arch; }

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

void NVIDIAGPU::OnElfImageLoaded(
    const CUDBGEvent::cases_st::elfImageLoaded_st &event) {
  const auto &[dev_id, context_id, module_id, elf_image_size, handle,
               properties] = event;

  // Note that module_id and handle are the same thing. It is a vestige of the
  // older debug API.
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(
      log,
      "LLDBServerPluginNVIDIAGPU::OnElfImageLoaded() dev_id: {}, context_id: "
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
        "NVIDIAGPU::OnElfImageLoaded(). Failed to get elf image: {}",
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

bool NVIDIAGPU::HasUnreportedLibraries() const {
  return !m_unreported_libraries.empty();
}

void NVIDIAGPU::ReportDyldStop() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::ReportDyldStop()");

  ThreadNVIDIAGPU &thread = *GPUThreads().begin();
  thread.SetStoppedByDynamicLoader();
  m_is_faking_a_stop_for_dyld = true;
  SetState(StateType::eStateStopped, true);
}

void NVIDIAGPU::OnAllDevicesSuspended(
    const CUDBGEvent::cases_st::allDevicesSuspended_st &event) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::OnAllDevicesSuspended()");
  // The following code path assumes that the stop event is triggered by an
  // exception. At some point we'll need to generalize this.
  std::optional<ExceptionInfo> exception_info =
      FindExceptionInfo(this->GetCudaAPI());
  if (!exception_info)
    logAndReportFatalError(
        "NVIDIAGPU::OnAllDevicesSuspended(). Non-exception stop unsupported");

  // We don't yet handle multiple threads, so we use the only one we have.
  ThreadNVIDIAGPU &thread = *GPUThreads().begin();
  thread.SetPhysicalCoords(exception_info->physical_coords);
  thread.SetStoppedByException(*exception_info);
  ChangeStateToStopped();
}

NVIDIAGPU::Extension NVIDIAGPU::Manager::GetSupportedExtensions() const {
  return Extension::gpu_dyld | Extension::address_spaces;
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

const CUDBGAPI_st &NVIDIAGPU::GetCudaAPI() {
  if (!m_api) {
    logAndReportFatalError("NVIDIAGPU::GetCudaAPI(). CUDA Debugger API is "
                           "not initialized");
  }
  return *m_api;
}

DeviceInformation &NVIDIAGPU::GetDeviceInformation(int device_id) {
  auto [it, inserted] =
      m_device_information.try_emplace(device_id, GetCudaAPI(), device_id);
  return it->second;
}

std::vector<AddressSpaceInfo> NVIDIAGPU::GetAddressSpaces() {
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

Status NVIDIAGPU::ReadMemoryWithSpace(lldb::addr_t addr, uint64_t addr_space,
                                      NativeThreadProtocol *thread, void *buf,
                                      size_t size, size_t &bytes_readn) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::ReadMemoryWithSpace(). addr: {}, size: {}", addr,
           size);

  auto GetPhysicalCoords = [&thread]() -> PhysicalCoords {
    ThreadNVIDIAGPU &nv_thread = *static_cast<ThreadNVIDIAGPU *>(thread);
    return nv_thread.GetPhysicalCoords();
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
    PhysicalCoords coords = GetPhysicalCoords();
    res = GetCudaAPI().readLocalMemory(coords.dev_id, coords.sm_id,
                                       coords.warp_id, coords.lane_id, addr,
                                       buf, size);
    break;
  }
  case AddressSpace::ParamStorage: {
    PhysicalCoords coords = GetPhysicalCoords();
    res = GetCudaAPI().readParamMemory(coords.dev_id, coords.sm_id,
                                       coords.warp_id, addr, buf, size);
    break;
  }
  case AddressSpace::SharedStorage: {
    PhysicalCoords coords = GetPhysicalCoords();
    res = GetCudaAPI().readSharedMemory(coords.dev_id, coords.sm_id,
                                        coords.warp_id, addr, buf, size);
    break;
  }
  case AddressSpace::GenericStorage: {
    PhysicalCoords coords = GetPhysicalCoords();
    res = GetCudaAPI().readGenericMemory(coords.dev_id, coords.sm_id,
                                         coords.warp_id, coords.lane_id, addr,
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

Status NVIDIAGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                             size_t &bytes_read) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVIDIAGPU::ReadMemory(). addr: {}, size: {}", addr, size);
  return ReadMemoryWithSpace(addr, AddressSpace::GlobalStorage,
                             /*thread=*/nullptr, buf, size, bytes_read);
}
