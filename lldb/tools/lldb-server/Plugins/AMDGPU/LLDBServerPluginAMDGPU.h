//===-- LLDBServerPluginAMDGPU.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H

#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/Status.h"

#include "AmdDbgApiHelpers.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "ProcessAMDGPU.h"
#include "llvm/ADT/StringRef.h"
#include <amd-dbgapi/amd-dbgapi.h>

namespace lldb_private {

class TCPSocket;

namespace lldb_server {

class GPUIOObject : public IOObject {
public:
  GPUIOObject(int notifier_fd)
      : lldb_private::IOObject(eFDTypeSocket), m_notifier_fd(notifier_fd) {}

  Status Read(void *buf, size_t &num_bytes) override {
    Status error;
    return error;
  }
  Status Write(const void *buf, size_t &num_bytes) override {
    Status error;
    return error;
  }
  virtual bool IsValid() const override { return true; }
  virtual Status Close() override {
    Status error;
    return error;
  }

  virtual WaitableHandle GetWaitableHandle() override { return m_notifier_fd; }

private:
  int m_notifier_fd = -1;
};

class LLDBServerPluginAMDGPU : public LLDBServerPlugin {
public:
  LLDBServerPluginAMDGPU(LLDBServerPlugin::GDBServer &native_process,
                         MainLoop &main_loop);
  ~LLDBServerPluginAMDGPU() override;
  llvm::StringRef GetPluginName() override;
  GPUActions GetInitializeActions() override;
  std::optional<struct GPUActions> NativeProcessIsStopping() override;
  void NativeProcessDidExit(const WaitStatus &exit_status) override;
  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  NativeProcessProtocol *GetNativeProcess() {
    return m_native_process.GetCurrentProcess();
  }
  ProcessAMDGPU *GetGPUProcess() {
    return (ProcessAMDGPU *)m_gdb_server->GetCurrentProcess();
  }

  // Free the memory using the matching callback provided to the debug library.
  static void FreeDbgApiClientMemory(void *mem);

  bool CreateGPUBreakpoint(uint64_t addr);

  void GpuRuntimeDidLoad();

  // TODO: make this private
  struct GPUInternalBreakpoinInfo {
    uint64_t addr;
    amd_dbgapi_breakpoint_id_t breakpoind_id;
  };
  std::optional<GPUInternalBreakpoinInfo> m_gpu_internal_bp;
  bool m_wait_for_gpu_internal_bp_stop = false;
  amd_dbgapi_architecture_id_t m_architecture_id = AMD_DBGAPI_ARCHITECTURE_NONE;

private:
  Status InitializeAmdDbgApi();
  Status AttachAmdDbgApi();
  Status DetachAmdDbgApi();
  Status InstallAmdDbgApiNotifierOnMainLoop();
  Status CreateGpuProcess();
  std::optional<GPUPluginConnectionInfo> CreateConnection();
  GPUActions SetGpuLoaderBreakpointByAddress();
  GPUActions SetConnectionInfo();
  Status HandleAmdDbgApiAttachError(llvm::StringRef error_msg,
                                    amd_dbgapi_status_t status);

  bool HandleGPUInternalBreakpointHit(const GPUInternalBreakpoinInfo &bp);
  // An enum to control how to process the target event in `process_event_queue`
  // If the boundary is inclusive, the event will be handled and marked as
  // processed. If the boundary is exclusive, the event will not be handled and
  // left as unprocessed. An event is marked as processed by calling
  // `amd_dbgapi_event_processed`. An unprocessed event can be retrieved from
  // the returned event set by calling `GetLastEventID()`.
  enum EventBoundaryType { ProcessEventInclusive, ProcessEventExclusive };
  AmdDbgApiEventSet
  process_event_queue(amd_dbgapi_event_kind_t until_event_kind,
                      EventBoundaryType boundary_type = ProcessEventInclusive);
  void HandleNotifierDataReady();
  bool SetGPUBreakpoint(uint64_t addr, const uint8_t *bp_instruction,
                        size_t size);
  bool ReadyToAttachDebugLibrary();
  bool ReadyToSendConnectionRequest();
  bool ReadyToSetGpuLoaderBreakpointByAddress();
  ProcessAMDGPU *GetGPUProcess() const;

  Status m_main_loop_status;
  MainLoopBase::ReadHandleUP m_gpu_event_read_up;
  std::vector<MainLoopBase::ReadHandleUP> m_read_handles;
  std::unique_ptr<TCPSocket> m_listen_socket; // Keep socket alive for main_loop
  std::shared_ptr<GPUIOObject> m_gpu_event_io_obj_sp;

  amd_dbgapi_process_id_t m_gpu_pid = AMD_DBGAPI_PROCESS_NONE;
  static constexpr int INVALID_NOTIFIER_ID = -1;
  int m_notifier_fd = INVALID_NOTIFIER_ID;
  enum class AmdDbgApiState {
    Uninitialized,
    Initialized,
    Attached,
    RuntimeLoaded,
    Detached,
    Error,
  };
  AmdDbgApiState m_amd_dbg_api_state = AmdDbgApiState::Uninitialized;
};

// Helper to free memory with custom deleter for unique_ptr.
inline void DbgApiClientMemoryDeleter::operator()(void *ptr) const {
  if (ptr) {
    LLDBServerPluginAMDGPU::FreeDbgApiClientMemory(ptr);
  }
}

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINAMDGPU_H
