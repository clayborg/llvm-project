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
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessNVGPU::ProcessNVGPU(lldb::pid_t pid, NativeDelegate &delegate,
                           MainLoop &main_loop)
    : NativeProcessProtocol(pid, -1, delegate), m_arch(GetNVPTXArchitecture()),
      m_api(nullptr), m_main_loop(main_loop),
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

  // Make stop-derived device state unavailable before GPU execution resumes.
  for (ThreadNVGPU &thread : GPUThreads())
    thread.SetRunning();

  SetState(StateType::eStateRunning, true);

  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api) {
    // The client takes a failed resume for a lost connection and drops the GPU
    // process. Stop again for the same reason instead, once the server has
    // finished handling the resume request.
    std::string reason = llvm::toString(api.takeError());
    LLDB_LOG(log, "NVGPU::Resume(). Stopping again: {0}", reason);
    m_main_loop.AddPendingCallback(
        [this, reason](MainLoopBase &) { ReportFallbackStop(reason); });
    return Status();
  }

  if (m_is_faking_a_stop_for_dyld) {
    m_is_faking_a_stop_for_dyld = false;
    // Ack'ing here is fine because the next call to OnDebuggerAPIEvent will
    // be triggered after the Resume packet has been fully processed.
    CUDBGResult res = api->acknowledgeSyncEvents();
    if (res != CUDBG_SUCCESS) {
      ReportFaultAfterResume(
          llvm::formatv("Failed to acknowledge CUDA Debugger API events. {0}",
                        cudbgGetErrorString(res))
              .str());
      return Status();
    }
    // The driver only notifies for newly enqueued events, so anything still
    // queued behind the ELF image load would never be drained on its own.
    if (m_sync_event_drain_notifier)
      m_sync_event_drain_notifier();
  } else {
    for (DeviceState &device : m_devices.GetDevices()) {
      CUDBGResult res = api->resumeDevice(device.GetDeviceId());
      if (res != CUDBG_SUCCESS) {
        ReportFaultAfterResume(llvm::formatv("Failed to resume device {0}. {1}",
                                             device.GetDeviceId(),
                                             cudbgGetErrorString(res))
                                   .str());
        return Status();
      }
    }
  }

  return Status();
}

void ProcessNVGPU::ReportFaultAfterResume(std::string description) {
  LLDB_LOG(GetLog(GDBRLog::Plugin), "NVGPU::Resume(). {0}", description);
  // The event drain can run before the callback, so stop using the API now.
  m_api = nullptr;
  m_api_fault = description;
  m_main_loop.AddPendingCallback([this, description](MainLoopBase &) {
    if (m_api_fault_handler)
      m_api_fault_handler(description);
    else
      ReportFallbackStop(description);
  });
}

llvm::Error ProcessNVGPU::SetDebuggerAPI(CUDADebuggerAPI &api) {
  Log *log = GetLog(GDBRLog::Plugin);
  m_api = api.GetRawAPI();
  m_api_version = api.GetAPIVersion();
  llvm::Expected<DeviceStateRegistry> devices =
      DeviceStateRegistry::Create(*this);
  if (!devices)
    return devices.takeError();
  m_devices = std::move(*devices);
  size_t max_num_threads = m_devices.GetMaxNumSupportedThreads();
  LLDB_LOG(log,
           "NVGPU::SetDebuggerAPI(). Reserving space for max num threads: {}",
           max_num_threads);
  m_threads.reserve(max_num_threads);
  return llvm::Error::success();
}

Status ProcessNVGPU::Halt() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::Halt(). Pre-halt state: {}", StateToString(GetState()));
  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api)
    return Status::FromError(api.takeError());
  // According to Andrew, halting the devices takes ~0.2ms - ~10 ms.
  CUDBGResult res = api->suspendDevice(/*device_id=*/0);

  Status status;
  if (res != CUDBG_SUCCESS) {
    std::string error_string =
        std::string("Failed to suspend device. ") + cudbgGetErrorString(res);
    LLDB_LOG(log, "NVGPU::Halt(). {}", error_string);
    status = Status::FromErrorString(error_string.c_str());
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

bool ProcessNVGPU::IsDeviceStateCurrent() const {
  std::lock_guard<std::recursive_mutex> guard(m_state_mutex);
  return m_state == eStateStopped && !m_is_faking_a_stop_for_dyld &&
         m_devices.HasSnapshotForStopID(m_stop_id);
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

ArchSpec ProcessNVGPU::GetNVPTXArchitecture() {
  return ArchSpec(eArchTypeELF, llvm::ELF::EM_CUDA,
                  ArchSpec::eNVPTXSubType_nvptx64);
}

Status ProcessNVGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                   bool hardware) {
  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api)
    return Status::FromError(api.takeError());

  // setBreakpoint doesn't return coherent errors, so we just ignore them.
  for (DeviceState &device : m_devices.GetDevices()) {
    CUDBGResult res = api->setBreakpoint(device.GetDeviceId(), addr);
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
  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api)
    return Status::FromError(api.takeError());

  // unsetBreakpoint doesn't return coherent errors, so we just ignore them.
  for (DeviceState &device : m_devices.GetDevices()) {
    CUDBGResult res = api->unsetBreakpoint(device.GetDeviceId(), addr);
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
  auto gpu_up =
      std::make_unique<ProcessNVGPU>(pid, native_delegate, m_mainloop);
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
/// \param[in] elf
///     The parsed cubin.
///
/// \return
///     Vector of GPUSectionInfo structures containing section details, or an
///     error if the section headers do not parse.
static llvm::Expected<std::vector<GPUSectionInfo>>
GetLoadSectionsForCubin(const llvm::object::ELF64LEObjectFile &elf) {
  Log *log = GetLog(GDBRLog::Plugin);
  std::vector<GPUSectionInfo> loaded_sections;

  const llvm::object::ELF64LEFile &elf_file = elf.getELFFile();
  llvm::Expected<llvm::object::ELF64LEFile::Elf_Shdr_Range> sections_or_err =
      elf_file.sections();
  if (!sections_or_err)
    return sections_or_err.takeError();

  LLDB_LOGV(log, "GetLoadSectionsForCubin(). Iterating through ELF sections");

  for (const llvm::object::ELF64LEFile::Elf_Shdr &section : *sections_or_err) {
    llvm::Expected<llvm::StringRef> name_or_err =
        elf_file.getSectionName(section);

    if (!name_or_err) {
      LLDB_LOG_ERRORV(
          log, name_or_err.takeError(),
          "GetLoadSectionsForCubin(). Failed to get section name: {0}");
      continue;
    }

    // Sections with 0 as load address shouldn't be "loaded".
    if (section.sh_addr == 0)
      continue;

    // For NVIDIA cubin images, section virtual addresses are encoded as
    // absolute addresses
    LLDB_LOGV(log, "  Section: {0}, Virtual Address: {1:x}, Size: {2}",
              *name_or_err, section.sh_addr, section.sh_size);

    // Add the section to the loaded sections list
    GPUSectionInfo section_info;
    section_info.names.push_back(name_or_err->str());
    section_info.load_address = section.sh_addr;
    loaded_sections.push_back(section_info);
  }

  return loaded_sections;
}

/// st_other bit that marks a kernel entry in a cubin's symbol table, from the
/// CUDA ABI. On a data symbol the same bit means STO_CUDA_UNIFIED_OBJECT, so
/// only test it on functions.
static constexpr uint8_t STO_CUDA_ENTRY = 0x10;

/// Get the name of the source file a cubin was compiled from, without its
/// directory or extension, from the name of its first compile unit.
///
/// Only a cubin built with -G has a compile unit. The line table of a cubin
/// built with -lineinfo alone does not say which of its files is the main one,
/// and a header can come first.
///
/// \param[in] elf
///     The parsed cubin.
///
/// \return
///     The name, or std::nullopt if the cubin has no compile unit.
static std::optional<std::string>
GetCubinSourceName(const llvm::object::ELF64LEObjectFile &elf) {
  Log *log = GetLog(GDBRLog::Plugin);
  auto log_error = [log](llvm::Error error) {
    LLDB_LOG_ERROR(log, std::move(error), "GetCubinSourceName(). {0}");
  };
  // The image is already relocated, and LLVM does not know R_CUDA relocations.
  std::unique_ptr<llvm::DWARFContext> dwarf = llvm::DWARFContext::create(
      elf, llvm::DWARFContext::ProcessDebugRelocations::Ignore,
      /*L=*/nullptr, /*DWPName=*/"", log_error, log_error);
  llvm::DWARFContext::compile_unit_range units = dwarf->compile_units();
  if (units.empty())
    return std::nullopt;
  const char *path = llvm::dwarf::toString(
      (*units.begin())->getUnitDIE().find(llvm::dwarf::DW_AT_name), nullptr);
  if (!path)
    return std::nullopt;
  llvm::StringRef name = llvm::sys::path::stem(path);
  if (name.empty())
    return std::nullopt;
  return name.str();
}

/// Get the qualified name of a function, without its parameters, template
/// arguments, or anonymous namespaces, to name a cubin after.
///
/// \param[in] symbol_name
///     The function's symbol name, mangled or not.
///
/// \return
///     The demangled name, \a symbol_name itself if it is not an Itanium
///     mangled name, or std::nullopt if it is one that does not demangle.
static std::optional<std::string>
GetCubinFunctionName(llvm::StringRef symbol_name) {
  if (!symbol_name.starts_with("_Z"))
    return symbol_name.str();

  // The demangler points into the name it parsed until it is destroyed.
  std::string mangled_name = symbol_name.str();
  llvm::ItaniumPartialDemangler demangler;
  if (demangler.partialDemangle(mangled_name.c_str()))
    return std::nullopt;
  std::unique_ptr<char, llvm::FreeDeleter> base_name(
      demangler.getFunctionBaseName(nullptr, nullptr));
  if (!base_name)
    return std::nullopt;
  std::unique_ptr<char, llvm::FreeDeleter> context(
      demangler.getFunctionDeclContextName(nullptr, nullptr));
  std::string name =
      context && *context
          ? llvm::formatv("{0}::{1}", context.get(), base_name.get()).str()
          : std::string(base_name.get());
  // An anonymous namespace would put spaces and parentheses in the name.
  constexpr llvm::StringLiteral anonymous_namespace = "(anonymous namespace)::";
  for (size_t pos = name.find(anonymous_namespace); pos != std::string::npos;
       pos = name.find(anonymous_namespace))
    name.erase(pos, anonymous_namespace.size());
  llvm::replace(name, '/', '_');
  return name;
}

/// Get the name to show a cubin by: the source file it was compiled from if
/// it has a compile unit, or else its first kernel entry, or else its first
/// defined global function.
///
/// The name is only a label for users. Nothing may map it back to a kernel or
/// source file, or find a cubin by a name derived from one.
///
/// \param[in] elf
///     The parsed cubin.
///
/// \return
///     The name, without a file extension, or std::nullopt if the cubin has
///     nothing to name it after or that function's name does not demangle.
static std::optional<std::string>
GetCubinBaseName(const llvm::object::ELF64LEObjectFile &elf) {
  if (std::optional<std::string> source_name = GetCubinSourceName(elf))
    return source_name;

  Log *log = GetLog(GDBRLog::Plugin);
  std::optional<llvm::StringRef> first_global;
  for (const llvm::object::ELFSymbolRef &symbol : elf.symbols()) {
    if (symbol.getELFType() != llvm::ELF::STT_FUNC)
      continue;
    llvm::Expected<uint32_t> flags = symbol.getFlags();
    if (!flags) {
      LLDB_LOG_ERROR(log, flags.takeError(),
                     "GetCubinBaseName(). Failed to get symbol flags: {0}");
      continue;
    }
    // A cubin must not be named after a function it only calls, like vprintf.
    if (*flags & llvm::object::SymbolRef::SF_Undefined)
      continue;
    llvm::Expected<llvm::StringRef> name = symbol.getName();
    if (!name) {
      LLDB_LOG_ERROR(log, name.takeError(),
                     "GetCubinBaseName(). Failed to get symbol name: {0}");
      continue;
    }
    if (symbol.getOther() & STO_CUDA_ENTRY)
      return GetCubinFunctionName(*name);
    if (!first_global && symbol.getBinding() != llvm::ELF::STB_LOCAL)
      first_global = *name;
  }
  if (!first_global)
    return std::nullopt;
  return GetCubinFunctionName(*first_global);
}

llvm::Error ProcessNVGPU::OnElfImageLoaded(
    const CUDBGEvent::cases_st::elfImageLoaded_st &event) {
  const auto &[dev_id, context_id, module_id, elf_image_size, handle,
               properties] = event;

  // Note that module_id and handle are the same thing. It is a vestige of the
  // older debug API.
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGV(log,
            "LLDBServerPluginNVGPU::OnElfImageLoaded() dev_id: {}, context_id: "
            "{}, module_id: {}, elf_image_size: {}, handle: {}, properties: {}",
            dev_id, context_id, module_id, elf_image_size, handle, properties);

  // Only the event drain calls this, and the drain never runs without the API.
  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api)
    return api.takeError();

  // Obtain the elf image
  std::unique_ptr<llvm::WritableMemoryBuffer> data_buffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(elf_image_size);
  CUDBGResult res = api->getElfImageByHandle(
      dev_id, handle, CUDBGElfImageType::CUDBG_ELF_IMAGE_TYPE_RELOCATED,
      data_buffer->getBufferStart(), elf_image_size);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt(
        "Failed to get the ELF image of CUDA module {0:x}. {1}", handle,
        cudbgGetErrorString(res));

  // A cubin that does not parse has no name or section load addresses to
  // report. Skip it rather than stop debugging the GPU.
  llvm::MemoryBufferRef elf_buffer_ref(*data_buffer);
  llvm::Expected<llvm::object::ELF64LEObjectFile> elf_or_err =
      llvm::object::ELF64LEObjectFile::create(elf_buffer_ref);
  if (!elf_or_err) {
    LLDB_LOG_ERROR(log, elf_or_err.takeError(),
                   "NVGPU::OnElfImageLoaded(). Skipping the ELF image of CUDA "
                   "module {1:x}, which does not parse: {0}",
                   handle);
    return llvm::Error::success();
  }
  llvm::Expected<std::vector<GPUSectionInfo>> loaded_sections =
      GetLoadSectionsForCubin(*elf_or_err);
  if (!loaded_sections) {
    LLDB_LOG_ERROR(log, loaded_sections.takeError(),
                   "NVGPU::OnElfImageLoaded(). Skipping the ELF image of CUDA "
                   "module {1:x}, whose sections do not parse: {0}",
                   handle);
    return llvm::Error::success();
  }

  // The client finds a cubin's module by this name, and keeps the modules of
  // cubins that were unloaded, so a name is never given out twice.
  std::string base_name = GetCubinBaseName(*elf_or_err)
                              .value_or("cuda_elf_" + std::to_string(handle));
  uint32_t &times_named = m_cubin_name_counts[base_name];
  ++times_named;
  GPUDynamicLoaderLibraryInfo lib1;
  lib1.pathname =
      times_named == 1
          ? base_name + ".cubin"
          : llvm::formatv("{0}-{1}.cubin", base_name, times_named).str();
  LLDB_LOG(log, "NVGPU::OnElfImageLoaded(). Elf image {0:x} is {1}", handle,
           lib1.pathname);
  lib1.load = true;
  lib1.elf_image_base64_sp = std::make_shared<std::string>(
      llvm::encodeBase64(elf_buffer_ref.getBuffer()));
  lib1.loaded_sections = std::move(*loaded_sections);

  m_unreported_libraries.push_back(lib1);
  m_all_libraries.push_back(lib1);
  return llvm::Error::success();
}

bool ProcessNVGPU::HasUnreportedLibraries() const {
  return !m_unreported_libraries.empty();
}

void ProcessNVGPU::ReportDyldStop() {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ReportDyldStop()");

  m_is_faking_a_stop_for_dyld = true;
  ThreadNVGPU &thread = *GPUThreads().begin();
  thread.SetStoppedByDynamicLoader();
  SetState(StateType::eStateStopped, true);
}

/// Release all pointers to ThreadNVGPU objects and clear the vector.
static void ReleaseAndClearThreads(
    std::vector<std::unique_ptr<NativeThreadProtocol>> &threads) {
  for (std::unique_ptr<NativeThreadProtocol> &thread : threads)
    thread.release();
  threads.clear();
}

ProcessNVGPU::~ProcessNVGPU() {
  // m_threads holds pointers to threads this process does not own: the
  // fallback thread is a member, and the GPU threads belong to m_devices.
  // Release them so that NativeProcessProtocol's destructor does not delete
  // them. The fallback thread is still in m_threads if the GPU never
  // suspended, for example after the debug API failed to initialize.
  ReleaseAndClearThreads(m_threads);
}

llvm::Error ProcessNVGPU::OnAllDevicesSuspended(
    const CUDBGEvent::cases_st::allDevicesSuspended_st &event) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::OnAllDevicesSuspended()");

  if (llvm::Error err = m_devices.BatchUpdate(GetNextStopID()))
    return err;
  LLDB_LOGV(log, "Device info dump:\n{}", m_devices.Dump());

  ReleaseAndClearThreads(m_threads);

  // We report as the selected thread the first one that has an exception, or
  // else the first one that is at a breakpoint.
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
          else if (stop_reason == lldb::eStopReasonException &&
                   !exception_thread_id)
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
  return llvm::Error::success();
}

void ProcessNVGPU::ReportFallbackStop(llvm::StringRef description) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::ReportFallbackStop(). {0}", description);

  ReleaseAndClearThreads(m_threads);
  m_fallback_thread.SetStopped(lldb::eStopReasonException, description);
  m_threads.push_back(
      std::unique_ptr<NativeThreadProtocol>(&m_fallback_thread));
  SetCurrentThreadID(m_fallback_thread.GetID());
  ChangeStateToStopped();
}

void ProcessNVGPU::OnDebuggerAPIFault(std::string description) {
  m_api = nullptr;
  m_api_fault = std::move(description);
  ReportFallbackStop(*m_api_fault);
}

ProcessNVGPU::Extension ProcessNVGPU::Manager::GetSupportedExtensions() const {
  return Extension::lldb_settings | Extension::address_spaces;
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

llvm::Expected<const CUDBGAPI_st &> ProcessNVGPU::GetCudaAPI() {
  if (m_api)
    return *m_api;
  if (m_api_fault)
    return llvm::createStringError(*m_api_fault);
  return llvm::createStringError("The CUDA debugger API is not available");
}

std::vector<AddressSpaceInfo> ProcessNVGPU::GetAddressSpaces() {
  llvm::ArrayRef<AddressSpaceInfo> infos = nvgpu::GetAddressSpaceInfos();
  return {infos.begin(), infos.end()};
}

Status ProcessNVGPU::ReadMemory(const ProcessAddress &process_addr, void *buf,
                                size_t size, size_t &bytes_read) {
  bytes_read = 0;
  llvm::Expected<const CUDBGAPI_st &> api = GetCudaAPI();
  if (!api)
    return Status::FromError(api.takeError());

  const lldb::addr_t addr = process_addr.GetValue();
  const lldb::addr_space_t addr_space =
      process_addr.IsInDefaultAddressSpace()
          ? static_cast<lldb::addr_space_t>(AddressSpace::GlobalStorage)
          : process_addr.GetAddressSpace();

  const AddressSpaceInfo *info = nvgpu::FindAddressSpaceInfo(addr_space);
  if (!info)
    return Status::FromErrorStringWithFormatv("Invalid address space '{}'",
                                              addr_space);

  NativeThreadProtocol *thread = nullptr;
  if (std::optional<lldb::tid_t> tid = process_addr.GetThreadID()) {
    thread = GetThreadByID(*tid);
    if (!thread)
      return Status::FromErrorStringWithFormatv("Invalid thread ID 0x{0:x}",
                                                *tid);
  }

  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGV(log, "NVGPU::ReadMemory(). addr: {}, address space: {}, size: {}",
            addr, addr_space, size);

  if (!info->is_thread_specific) {
    // Const storage can be read as global storage.
    CUDBGResult res = api->readGlobalMemory(addr, buf, size);
    if (res != CUDBG_SUCCESS)
      return Status::FromErrorString(cudbgGetErrorString(res));

    bytes_read = size;
    return Status();
  }

  auto *nv_thread = static_cast<ThreadNVGPU *>(thread);
  if (!nv_thread || !nv_thread->GetThreadState()) {
    return Status::FromErrorStringWithFormatv(
        "Reading from address space '{}' requires a current CUDA thread",
        addr_space);
  }

  const nvgpu::PhysicalCoords &physical =
      nv_thread->GetThreadState()->GetPhysicalCoords();
  if (!physical.device || !physical.sm || !physical.warp) {
    return Status::FromErrorStringWithFormatv(
        "Reading from address space '{}' requires known CUDA device, SM, and "
        "warp coordinates",
        addr_space);
  }

  if ((addr_space == AddressSpace::LocalStorage ||
       addr_space == AddressSpace::GenericStorage) &&
      !physical.lane) {
    return Status::FromErrorStringWithFormatv(
        "Reading from address space '{}' requires a known CUDA lane "
        "coordinate",
        addr_space);
  }

  CUDBGResult res;
  switch (addr_space) {
  case AddressSpace::LocalStorage: {
    res = api->readLocalMemory(*physical.device, *physical.sm, *physical.warp,
                               *physical.lane, addr, buf, size);
    break;
  }
  case AddressSpace::ParamStorage: {
    res = api->readParamMemory(*physical.device, *physical.sm, *physical.warp,
                               addr, buf, size);
    break;
  }
  case AddressSpace::SharedStorage: {
    res = api->readSharedMemory(*physical.device, *physical.sm, *physical.warp,
                                addr, buf, size);
    break;
  }
  case AddressSpace::GenericStorage: {
    res = api->readGenericMemory(*physical.device, *physical.sm, *physical.warp,
                                 *physical.lane, addr, buf, size);
    break;
  }
  default:
    return Status::FromErrorStringWithFormatv("Invalid address space '{}'",
                                              addr_space);
  }

  if (res != CUDBG_SUCCESS)
    return Status::FromErrorString(cudbgGetErrorString(res));

  bytes_read = size;
  return Status();
}

void ProcessNVGPU::OnNativeProcessExit(const WaitStatus &exit_status) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "NVGPU::OnNativeProcessExit(). exit_status: {}", exit_status);
  // Set our exit status to match the native process and notify delegates.
  std::lock_guard<std::recursive_mutex> guard(m_state_mutex);
  SetExitStatus(exit_status, /*bNotifyStateChange=*/true);
}

std::vector<std::string> ProcessNVGPU::GetStructuredDataPlugins() {
  return {"nvgpu-monitor"};
}
