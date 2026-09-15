//===-- ProcessIntelGT.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_PROCESSINTELGT_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_PROCESSINTELGT_H

#include "EUThreadIntelGT.h"
#include "GpuModuleManager.h"
#include "LevelZeroHelpers.h"
#include "ThreadIntelGT.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/AddressSpace.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace lldb_private {
namespace lldb_server {

class LLDBServerPluginIntelGT;

/// Per-device ZE debug session record.
struct DeviceSession {
  ze_device_handle_t device = nullptr;
  zet_debug_session_handle_t session = nullptr;
  ze_device_properties_t properties{};
  uint32_t device_index = 0;
  uint64_t tid_base = 0;
  uint64_t max_threads = 0;

  /// Device-level register metadata (discovered once per device).
  std::vector<zet_debug_regset_properties_t> regset_props;
  std::vector<lldb_private::RegisterInfo> reg_infos;
  std::vector<lldb_private::RegisterSet> reg_sets;
  std::vector<std::string> reg_names;

  struct RegLocation {
    uint32_t regset_index;
    uint32_t byte_offset;
    uint32_t byte_size;
    uint32_t full_byte_size;
    bool is_simd_wide;
  };
  std::vector<RegLocation> reg_locations;

  std::vector<std::vector<uint32_t>> reg_set_registers;
  std::vector<const uint32_t *> reg_set_regs_ptrs;

  uint32_t pc_reg_num = UINT32_MAX;
  uint32_t ip_reg_num = UINT32_MAX;
  uint32_t isabase_reg_num = UINT32_MAX;

  /// True if device-level register discovery succeeded.
  bool registers_discovered = false;

  /// Total EU threads on this device (slices*subslices*EUs*threads).
  size_t nthreads = 0;
  /// Threads currently running; drain barrier waits for it to reach 0
  /// (underflow guarded).
  size_t nresumed = 0;
  /// Outstanding wildcard interrupt requests; avoids redundant sends.
  size_t ninterrupts = 0;
};

/// NativeProcessProtocol wrapping all Intel GPU EU threads across one or more
/// Level Zero debug sessions. Stopped EU threads expand into per-SIMD-lane
/// ThreadIntelGT objects; resume advances one EU thread at a time.
class ProcessIntelGT : public NativeProcessProtocol {
public:
  ProcessIntelGT(lldb::pid_t pid, NativeDelegate &delegate,
                 LLDBServerPluginIntelGT *plugin,
                 std::vector<DeviceSession> device_sessions);

  // ----- NativeProcessProtocol interface ------------------------------------

  Status Resume(const ResumeActionList &resume_actions) override;
  Status Halt() override;
  Status Detach() override;
  Status Signal(int signo) override;
  Status Interrupt() override;
  Status Kill() override;

  /// Interrupt all running EU threads on every device so siblings stop
  /// together; caller must then drain events until every thread reports.
  void PauseAll();

  /// Total EU threads still running across all devices; 0 signals the
  /// drain-until-quiescent barrier is complete.
  size_t GetResumedThreadCount() const;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;
  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  std::vector<AddressSpaceInfo> GetAddressSpaces() override;

  Status ReadMemoryWithSpace(lldb::addr_t addr, uint64_t addr_space,
                             NativeThreadProtocol *thread, void *buf,
                             size_t size, size_t &bytes_read) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override;

  /// Overrides that patch Intel GPU instruction bits via Level Zero.
  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;
  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  /// IntelGT PC points AT the breakpoint instruction.
  size_t GetSoftwareBreakpointPCOffset() override { return 0; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  std::optional<GPUDynamicLoaderResponse>
  GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) override;

  // ----- IntelGT-specific ---------------------------------------------------

  /// Called when the native CPU process exits.
  void HandleNativeProcessExit(const WaitStatus &exit_status);

  /// Handle a ZE MODULE_LOAD event.
  void HandleModuleLoad(const zet_debug_event_t &event,
                        const DeviceSession &dev_session);

  /// Handle a ZE MODULE_UNLOAD event.
  void HandleModuleUnload(const zet_debug_event_t &event);

  bool HasDyldChangesToReport() const {
    return m_gpu_module_manager.HasChangedCodeObjects();
  }

  /// Return the DeviceSession for a given device index, or nullptr.
  const DeviceSession *GetDeviceSession(uint32_t device_index) const;

  /// Mutable variant, for updating per-device counters (nresumed/ninterrupts).
  DeviceSession *GetDeviceSessionMutable(uint32_t device_index);

  /// Return the DeviceSession for the current stopped thread, or first session.
  const DeviceSession *GetCurrentDeviceSession() const;

  /// Lazy-init device-level register tables on first access.
  Status EnsureDeviceRegistersDiscovered(uint32_t device_index);

  /// Get or assign a kernel instance ID (thread-safe).
  unsigned int GetOrAssignKernelInstanceID(uint32_t implicit_args);

  /// Module manager for tracking GPU code objects.
  GpuModuleManager m_gpu_module_manager;

  /// MODULE_LOAD events with need_ack=true, ACKed in Resume() after LLDB
  /// has set breakpoints.
  struct PendingModuleAck {
    zet_debug_session_handle_t session;
    zet_debug_event_t event;
  };
  std::vector<PendingModuleAck> m_pending_module_acks;

  /// Maps a zebin ELF's pre-linked VA range (DWARF) to its execution VA.
  struct ModuleAddrRange {
    lldb::addr_t pre_linked_base; ///< ELF text section VMA
    lldb::addr_t execution_base;  ///< GPU execution VA
    lldb::addr_t size;            ///< module size in bytes
    uint32_t simd_width = 16;     ///< SIMD width from .ze_info (8/16/32)
  };
  std::vector<ModuleAddrRange> m_module_addr_ranges;

  /// Translate a pre-linked GPU VA (from DWARF) to the actual execution VA.
  lldb::addr_t TranslateToExecutionAddr(lldb::addr_t addr) const;

  /// Scan a zebin ELF in GPU memory: return the minimum SHF_ALLOC VMA (or
  /// LLDB_INVALID_ADDRESS on error) and set *out_simd_width from .ze_info.
  lldb::addr_t ScanElfModule(lldb::addr_t module_begin,
                             lldb::addr_t module_size, const DeviceSession *ds,
                             uint32_t *out_simd_width);

  /// SIMD width of the module containing \a isa_base (default 16).
  uint32_t LookupSimdWidthForIsaBase(lldb::addr_t isa_base) const;

  // ----- Thread management called from LLDBServerPluginIntelGT -------------

  /// Set basic process info from a launch info.
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);

  /// Mark GPU process as running (no delegate notification).
  void MarkRunning() { SetState(lldb::eStateRunning, /*notify=*/false); }

  /// Return true if only the shadow thread exists (no EU threads stopped).
  bool HasOnlyShadowThread() const { return m_stopped_eu_threads.empty(); }

  /// Return the number of currently stopped EU threads.
  size_t GetStoppedEUThreadCount() const { return m_stopped_eu_threads.size(); }

  /// Drop stale EU threads from m_threads before processing new stops.
  void ClearOldEUThreads();

  /// Check if a thread is armed for single-step.
  bool IsSteppingThread(ze_device_thread_t ze_thread) const {
    return m_stepping_eu_threads.find(ze_thread) != m_stepping_eu_threads.end();
  }

  /// True if any stopped thread has eStopReasonTrace (step-complete).
  bool IsAnyThreadSteppingCompleted() const;

  /// Remove trace-stop EU threads from the stopped list and m_threads.
  void RemoveSteppedThreads();

  /// Handle a THREAD_STOPPED event; create the EUThreadIntelGT and lane
  /// threads. Returns the TID of the first lane of the first stopped EU.
  lldb::tid_t HandleZeThreadStopped(const DeviceSession &ds,
                                    ze_device_thread_t ze_thread,
                                    bool is_first_stop = false);

  /// Stable TID base for a hardware EU thread; reserves simd_width slots
  /// so each physical lane gets a fixed slot across stops.
  lldb::tid_t GetOrAllocateTidBase(ze_device_thread_t ze_thread,
                                   uint32_t simd_width);

  /// Remove all lane threads for a ze_device_thread_t that became unavailable.
  void HandleZeThreadUnavailable(const DeviceSession &ds,
                                 ze_device_thread_t ze_thread);

  /// Handle a page fault event.
  void HandleZePageFault(uint64_t fault_address);

  /// Return the EUThreadIntelGT for the current lane thread, or nullptr.
  EUThreadIntelGT *GetCurrentEUThread();

  /// Resume action for an EU thread; searches its lane threads, then the
  /// default action.
  const ResumeAction *
  FindResumeActionForEUThread(const ResumeActionList &actions,
                              const EUThreadIntelGT *eu);

private:
  /// Discover and build register tables for a device.
  Status DiscoverDeviceRegisterSets(DeviceSession &session);

  ThreadIntelGT *FindThread(lldb::tid_t tid);
  ThreadIntelGT *GetCurrentThreadIntelGT();

  LLDBServerPluginIntelGT *m_plugin = nullptr;
  std::vector<DeviceSession> m_device_sessions;

  /// Canonical set of currently loaded module URIs.
  std::set<std::string> m_loaded_modules_canonical;

  /// Saved original instruction bytes for each software breakpoint.
  std::unordered_map<lldb::addr_t, std::vector<uint8_t>> m_bp_saved_opcodes;

  /// Currently stopped EU threads; each owns lane ThreadIntelGT objects.
  ZeThreadMap<std::shared_ptr<EUThreadIntelGT>> m_stopped_eu_threads;

  /// EU threads armed for single-step (CR0). Value is the lane TID the client
  /// requested to step; falls back to the first active lane if inactive.
  ZeThreadMap<lldb::tid_t> m_stepping_eu_threads;

  /// Persistent EU-thread to TID-base map; survives stop/resume cycles.
  ZeThreadMap<lldb::tid_t> m_ze_thread_tid_map;

  /// Implicit-args pointer -> kernel instance ID; guarded by mutex.
  std::unordered_map<uint32_t, unsigned int> m_kernel_instances;
  mutable std::mutex m_kernel_instances_mutex;

  /// Fallback SIMD width when a thread has no matching module.
  static constexpr uint32_t kFallbackSimdWidth = 16;

  mutable ArchSpec m_arch;
  ProcessInstanceInfo m_process_info;
};

/// NativeProcessProtocol::Manager for IntelGT.
class ProcessManagerIntelGT : public NativeProcessProtocol::Manager {
public:
  explicit ProcessManagerIntelGT(MainLoop &mainloop)
      : NativeProcessProtocol::Manager(mainloop) {}

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Launch(ProcessLaunchInfo &launch_info,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Attach(lldb::pid_t pid,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;

  NativeProcessProtocol::Extension GetSupportedExtensions() const override {
    return NativeProcessProtocol::Extension::lldb_settings |
           NativeProcessProtocol::Extension::address_spaces;
  }

  LLDBServerPluginIntelGT *m_plugin = nullptr;
  std::vector<DeviceSession> m_device_sessions;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_PROCESSINTELGT_H
