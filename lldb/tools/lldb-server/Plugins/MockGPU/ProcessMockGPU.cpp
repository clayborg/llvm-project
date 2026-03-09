//===-- ProcessMockGPU.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessMockGPU.h"
#include "ThreadMockGPU.h"

#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/UnimplementedError.h"
#include "llvm/Support/Error.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"

#include <thread>


using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

ProcessMockGPU::ProcessMockGPU(lldb::pid_t pid, NativeDelegate &delegate)
    : NativeProcessProtocol(pid, -1, delegate) {
  m_state = eStateStopped;
  UpdateThreads();
}

Status ProcessMockGPU::Resume(const ResumeActionList &resume_actions) {
  SetState(StateType::eStateRunning, true);
  // Always simulate a stop after a short delay so the mock never runs forever.
  // If software breakpoints are set, simulate hitting the first one; otherwise
  // simulate a trace stop.
  if (!m_software_breakpoints.empty()) {
    lldb::addr_t bp_addr = m_software_breakpoints.begin()->first;
    std::thread([this, bp_addr] {
      usleep(50000); // 50ms — let the continue response be sent first
      ThreadMockGPU &thread = static_cast<ThreadMockGPU &>(*m_threads[0]);
      thread.SetStoppedByBreakpoint(bp_addr);
      SetState(StateType::eStateStopped, true);
    }).detach();
  } else {
    std::thread([this] {
      usleep(50000);
      ThreadMockGPU &thread = static_cast<ThreadMockGPU &>(*m_threads[0]);
      thread.SetStoppedByTrace();
      SetState(StateType::eStateStopped, true);
    }).detach();
  }
  return Status();
}

Status ProcessMockGPU::Halt() {
  SetState(StateType::eStateStopped, true);
  return Status();
}

Status ProcessMockGPU::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

/// Sends a process a UNIX signal \a signal.
///
/// \return
///     Returns an error object.
Status ProcessMockGPU::Signal(int signo) {
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
Status ProcessMockGPU::Interrupt() { return Halt(); }

Status ProcessMockGPU::Kill() { return Status(); }

Status ProcessMockGPU::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                  size_t &bytes_read) {
  // Return data for mapped addresses and zeros for unmapped ones. This allows
  // SetSoftwareBreakpoint to succeed for any address (it reads existing bytes
  // before writing the trap opcode), which is needed because the mock GPU has
  // no real memory — breakpoint addresses come from the CPU binary's symbols.
  uint8_t *dst = static_cast<uint8_t *>(buf);
  for (size_t i = 0; i < size; ++i) {
    auto it = m_memory.find(addr + i);
    dst[i] = (it != m_memory.end()) ? it->second : 0;
  }
  bytes_read = size;
  return Status();
}

#define ADDR_SPACE_1 1
#define ADDR_SPACE_2 2

std::vector<AddressSpaceInfo> ProcessMockGPU::GetAddressSpaces() {
  std::vector<AddressSpaceInfo> result;
  result.push_back({"Global", ADDR_SPACE_1, false});
  result.push_back({"Thread", ADDR_SPACE_2, true});
  return result;
}

Status ProcessMockGPU::ReadMemoryWithSpace(lldb::addr_t addr, 
                                           uint64_t addr_space, 
                                           NativeThreadProtocol *thread, 
                                           void *buf, size_t size, 
                                           size_t &bytes_read) {
  bytes_read = 0;
  switch (addr_space) {
  case ADDR_SPACE_1:
    ::memset(buf, '1', size);
    bytes_read = size;
    return Status();

  case ADDR_SPACE_2:
    // Address space 1 requires a thread
    if (thread == nullptr)
      return Status::FromErrorString("reading from address space \"Thread\" "
                                     "requires a thread");
    ::memset(buf, '2', size);
    bytes_read = size;
    return Status();

  default:
    return Status::FromErrorStringWithFormat("invalid address space %" PRIu64, 
                                             addr_space);
  }
}


Status ProcessMockGPU::WriteMemory(lldb::addr_t addr, const void *buf,
                                   size_t size, size_t &bytes_written) {
  const uint8_t *src = static_cast<const uint8_t *>(buf);
  for (size_t i = 0; i < size; ++i)
    m_memory[addr + i] = src[i];
  bytes_written = size;
  return Status();
}

lldb::addr_t ProcessMockGPU::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

size_t ProcessMockGPU::UpdateThreads() {
  if (m_threads.empty()) {
    lldb::tid_t tid = 3456;
    m_threads.push_back(std::make_unique<ThreadMockGPU>(*this, 3456));
    // ThreadMockGPU &thread = static_cast<ThreadMockGPU &>(*m_threads.back());
    SetCurrentThreadID(tid);
  }
  return m_threads.size();
}

const ArchSpec &ProcessMockGPU::GetArchitecture() const {
  m_arch = ArchSpec("mockgpu");
  return m_arch;
}

// Breakpoint functions
Status ProcessMockGPU::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                     bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  else
    return SetSoftwareBreakpoint(addr, size);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
ProcessMockGPU::GetSoftwareBreakpointTrapOpcode(size_t size_hint) {
  // Single-byte trap opcode for the mock GPU architecture.
  static const uint8_t g_mock_gpu_trap_opcode[] = {0xCC};
  return llvm::ArrayRef(g_mock_gpu_trap_opcode);
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessMockGPU::GetAuxvData() const {
  return nullptr; // TODO: try to return
                  // llvm::make_error<UnimplementedError>();
}

Status ProcessMockGPU::GetLoadedModuleFileSpec(const char *module_path,
                                               FileSpec &file_spec) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessMockGPU::GetFileLoadAddress(const llvm::StringRef &file_name,
                                          lldb::addr_t &load_addr) {
  return Status::FromErrorString("unimplemented");
}

void ProcessMockGPU::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<ProcessInfo &>(launch_info);
}

bool ProcessMockGPU::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOGF(log, "ProcessMockGPU::%s() entered", __FUNCTION__);
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

std::optional<GPUDynamicLoaderResponse> 
ProcessMockGPU::GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) {
  GPUDynamicLoaderResponse response;
  // First example of a shared library. This is for cases where there is a file
  // on disk that contains an object file that can be loaded into the process
  // and everything should be slid to the load address. All sections within this
  // file will be loaded at their file address + 0x20000. This is very typical
  // for ELF files.
  GPUDynamicLoaderLibraryInfo lib1;
  lib1.pathname = "/usr/lib/lib1.so";
  lib1.uuid_str = "A5D69E75-92DE-3FAB-BD95-5171EAE860CC";
  lib1.load_address = 0x20000;
  response.library_infos.push_back(lib1);
  // Second example of a shared library. This is for cases where there is an
  // object file contained within another object file at some file offset with
  // a file size. This one is slid to 0x30000, and all sections will get slid
  // by the same amount.
  GPUDynamicLoaderLibraryInfo lib2;
  lib2.pathname = "/tmp/a.out";
  lib1.uuid_str = "9F6F8018-B2D8-3946-8F38-38B0B890CC31";
  lib2.load_address = 0x30000;
  lib2.file_offset = 0x1000;
  lib2.file_size = 0x500;
  response.library_infos.push_back(lib2);
  /// Third example of a shared library. This is for cases where there the 
  /// object file is loaded into the memory of the native process. LLDB will 
  /// need create an in memory object file using the data in this info.
  GPUDynamicLoaderLibraryInfo lib3;
  lib3.pathname = "/usr/lib/lib3.so";
  lib3.native_memory_address = 0x4500000;
  lib3.native_memory_size = 0x2000;
  response.library_infos.push_back(lib3);

  /// Fourth example of a shared library where we load each of the top level
  /// sections of an object file at different addresses. 
  GPUDynamicLoaderLibraryInfo lib4;
  lib4.pathname = "/usr/lib/lib4.so";
  lib4.loaded_sections.push_back({{"PT_LOAD[0]"}, 0x0e0000});
  lib4.loaded_sections.push_back({{"PT_LOAD[1]"}, 0x100000});
  lib4.loaded_sections.push_back({{"PT_LOAD[2]"}, 0x0f0000});
  lib4.loaded_sections.push_back({{"PT_LOAD[3]"}, 0x020000});
  response.library_infos.push_back(lib4);

  /// Fifth example of a shared library. This is for cases where there the 
  /// object file is loaded individual sections are loaded at different 
  /// addresses instead of having a single load address for the entire object 
  /// file. This allows GPU plug-ins to load sections at different addresses 
  /// as they are loaded by the GPU driver. Sections can be created for 
  /// functions in the ObjectFileELF plug-in when parsing the GPU ELF file so
  /// that individual functions can be loaded at different addresses as the 
  /// driver loads them.
  GPUDynamicLoaderLibraryInfo lib5;
  lib5.pathname = "/usr/lib/lib5.so";
  /// Here we are going to assume that the .text section has functions that 
  /// create sections for each function in the object file. Then each function 
  /// can be loaded at a different address as the driver loads them.
  lib5.loaded_sections.push_back({{"PT_LOAD[1]", ".text", "foo"}, 0x80000}); 
  lib5.loaded_sections.push_back({{"PT_LOAD[1]", ".text", "bar"}, 0x80200}); 
  response.library_infos.push_back(lib5);
  return response;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessMockGPU::Manager::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = 1234;
  auto proc_up = std::make_unique<ProcessMockGPU>(pid, native_delegate);
  proc_up->SetLaunchInfo(launch_info);
  return proc_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessMockGPU::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}


ProcessMockGPU::Extension
ProcessMockGPU::Manager::GetSupportedExtensions() const {
  return Extension::lldb_settings | Extension::address_spaces;
}

void ProcessMockGPU::HandleNativeProcessExit(const WaitStatus &exit_status) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "ProcessMockGPU::{0}() native process exited with status=({1})",
           __FUNCTION__, exit_status);

  // Set our exit status to match the native process and notify delegates
  SetExitStatus(exit_status, true);
}
