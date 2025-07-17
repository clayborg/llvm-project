//===-- NVIDIAGPU.h -------------------------------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PROCESSNVIDIAGPU_H
#define LLDB_TOOLS_LLDB_SERVER_PROCESSNVIDIAGPU_H

#include "CUDADebuggerAPI.h"
#include "MainLoopEventNotifier.h"
#include "ThreadNVIDIAGPU.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/ProcessInfo.h"

#include "cudadebugger.h"

namespace lldb_private::lldb_server {

/// This class manages all the threads in the GPU, its memory, and its overall
/// state. It extends NativeProcessProtocol, which was meant for CPU processes,
/// but fortunately it provides most of the abstractions we need to manage
/// the GPU.
class NVIDIAGPU : public NativeProcessProtocol {
public:
  /// Forward declaration of helper class for thread iteration.
  class GPUThreadRange;

  /// Class that manages the creation of NVIDIAGPU objects.
  class Manager : public NativeProcessProtocol::Manager {
  public:
    Manager(MainLoop &mainloop) : NativeProcessProtocol::Manager(mainloop) {}

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info,
           NativeProcessProtocol::NativeDelegate &native_delegate) override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid,
           NativeProcessProtocol::NativeDelegate &native_delegate) override;

    Extension GetSupportedExtensions() const override;
  };

  NVIDIAGPU(lldb::pid_t pid, NativeDelegate &delegate);

  void SetDebuggerAPI(CUDADebuggerAPI &api) { m_api = api.GetRawAPI(); }

  CUDBGAPI GetDebuggerAPI() const { return m_api; }

  /// Resume the GPU, change its state and the state of each thread, then report
  /// the state change to the delegate, which will notify the client.
  Status Resume(const ResumeActionList &resume_actions) override;

  /// Perform \a Halt() and set the stop reason of the first thread to dyld.
  Status HaltDueToDyld();

  /// Halt the GPU but don't change its state. That requires a call to
  /// \a ChangeStateToStopped().
  Status Halt() override;

  Status Detach() override;

  Status Signal(int signo) override;

  Status Interrupt() override;

  Status Kill() override;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  /// Create or update the list of ThreadNVIDIAGPU objects of the GPU.
  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override;

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  /// \return the requested list of dynamic library infos.
  std::optional<GPUDynamicLoaderResponse>
  GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) override;

  /// Change the state of the process to stopped and notify delegates. It
  /// assumes that the GPU is in fact stopped.
  void ChangeStateToStopped();

  /// \return true if there are new cubins since the last time they were
  /// requested.
  bool HasUnreportedLibraries() const;

  /// Handle the AllDevicesSuspended event.
  void OnAllDevicesSuspended(
      const CUDBGEvent::cases_st::allDevicesSuspended_st &event);

  /// Handle the ElfImageLoaded event.
  void OnElfImageLoaded(const CUDBGEvent::cases_st::elfImageLoaded_st &event);

  /// Report a stop reason for the dynamic loader. This is used to notify the
  /// client that it should fetch the new libraries. It uses a fake stop
  /// as described in the comment for m_is_faking_a_stop_for_dyld.
  void ReportDyldStop();

private:
  friend class Manager;

  std::optional<ThreadNVIDIAGPU::ExceptionInfo> FindExceptionInfo();

  /// Accessor for m_api that fails if it's not initialized.
  const CUDBGAPI_st &GetCudaAPI();

  /// Utility for the Manager class to set the launch info for the GPU.
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);

  /// Get a range object that allows iterating over threads with automatic
  /// casting to ThreadNVIDIAGPU&.
  GPUThreadRange GPUThreads();

  ArchSpec m_arch;
  /// This contains some launch/attach/runtime information for the GPU. We are
  /// using the class used for CPU processes for this for simplicity.
  ProcessInstanceInfo m_process_info;
  CUDBGAPI m_api;

  /// The list of all the cubins loaded by the GPU.
  std::vector<GPUDynamicLoaderLibraryInfo> m_all_libraries;
  /// The list of all the cubins loaded by the GPU that haven't been reported to
  /// the client yet.
  std::vector<GPUDynamicLoaderLibraryInfo> m_unreported_libraries;

  /// A flag that indicates that we are faking a stop to the client to report
  /// dyld events. A fake stop means that we don't actually stop the GPU, but
  /// we stop processing more APU events until we have resumed after the dyld
  /// event.
  bool m_is_faking_a_stop_for_dyld = false;
};

} // namespace lldb_private::lldb_server

#endif
