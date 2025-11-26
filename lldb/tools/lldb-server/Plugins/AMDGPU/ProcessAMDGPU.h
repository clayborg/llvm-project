//===-- ProcessAMDGPU.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PROCESSAMDGPU_H
#define LLDB_TOOLS_LLDB_SERVER_PROCESSAMDGPU_H

#include "AmdDbgApiHelpers.h"
#include "GpuModuleManager.h"
#include "ThreadAMDGPU.h"
#include "WaveAMDGPU.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/ProcessInfo.h"
#include <amd-dbgapi/amd-dbgapi.h>

namespace lldb_private {
namespace lldb_server {

class LLDBServerPluginAMDGPU;
/// \class ProcessAMDGPU
/// Abstract class that extends \a NativeProcessProtocol for a mock GPU. This
/// class is used to unit testing the GPU plugins in lldb-server.
class ProcessAMDGPU : public NativeProcessProtocol {
  // TODO: change NativeProcessProtocol::GetArchitecture() to return by value
  mutable ArchSpec m_arch;
  ProcessInstanceInfo m_process_info;

public:
  ProcessAMDGPU(lldb::pid_t pid, NativeDelegate &delegate, 
                LLDBServerPluginAMDGPU *plugin);

  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  /// Sends a process a UNIX signal \a signal.
  ///
  /// \return
  ///     Returns an error object.
  Status Signal(int signo) override;

  /// Tells a process to interrupt all operations as if by a Ctrl-C.
  ///
  /// The default implementation will send a local host's equivalent of
  /// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
  /// operation.
  ///
  /// \return
  ///     Returns an error object.
  Status Interrupt() override;

  Status Kill() override;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  std::vector<AddressSpaceInfo> GetAddressSpaces() override;

  Status ReadMemoryWithSpace(lldb::addr_t addr, uint64_t addr_space,
                             NativeThreadProtocol *thread, void *buf,
                             size_t size, size_t &bytes_readn) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override;

  // Breakpoint functions
  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  // Custom accessors
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);

  std::optional<GPUDynamicLoaderResponse> 
  GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) override;

  /// Called when the native process exits to exit the GPU process
  void HandleNativeProcessExit(const WaitStatus &exit_status);

  bool handleWaveStop(amd_dbgapi_event_id_t eventId);

  bool handleDebugEvent(amd_dbgapi_event_id_t eventId,
                        amd_dbgapi_event_kind_t eventKind);

  bool HasDyldChangesToReport() const {
    return m_gpu_module_manager.HasChangedCodeObjects();
  }

  amd_dbgapi_process_id_t GetDbgApiProcessID() const {
    return amd_dbgapi_process_id_t{m_pid};
  }

  LLDBServerPluginAMDGPU* m_debugger = nullptr;
  GpuModuleManager m_gpu_module_manager;

  enum class State {
    Initializing,
    ModuleLoadStopped,
    Running,
    GPUStopped,
  };
  State m_gpu_state = State::Initializing;
  std::vector<amd_dbgapi_wave_id_t> m_wave_ids;

  ThreadAMDGPU *GetCurrentThreadAMDGPU() {
    return static_cast<ThreadAMDGPU *>(GetCurrentThread());
  }

  void ForEachThread(
      std::function<lldb_private::IterationAction(ThreadAMDGPU &)> const
          &callback);

private:
  WaveIdMap<std::shared_ptr<WaveAMDGPU>> m_waves;
  WaveIdList UpdateWavesAndReturnNew();
  llvm::Expected<DbgApiClientMemoryPtr<amd_dbgapi_wave_id_t>>
  GetWaveList(size_t *count, amd_dbgapi_changed_t *changed);
  llvm::Expected<DbgApiWaveInfo> GetWaveInfo(amd_dbgapi_wave_id_t wave_id);
  void UpdateThreadListFromWaves();
  ThreadAMDGPU *FindThread(std::function<bool(ThreadAMDGPU &)> pred);
  lldb::tid_t ChooseCurrentThread();
  void UpdateCurrentThread();
};

class ProcessManagerAMDGPU : public NativeProcessProtocol::Manager {
public:
  ProcessManagerAMDGPU(MainLoop &mainloop)
      : NativeProcessProtocol::Manager(mainloop) {}

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Launch(ProcessLaunchInfo &launch_info,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;

  NativeProcessProtocol::Extension GetSupportedExtensions() const override {
    return NativeProcessProtocol::Extension::lldb_settings |
           NativeProcessProtocol::Extension::address_spaces;
  }

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Attach(lldb::pid_t pid,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;

    LLDBServerPluginAMDGPU* m_debugger = nullptr;
};

} // namespace lldb_server
} // namespace lldb_private

#endif
