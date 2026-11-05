//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PROCESSNVGPU_H
#define LLDB_TOOLS_LLDB_SERVER_PROCESSNVGPU_H

#include "CUDADebuggerAPI.h"
#include "DeviceState.h"
#include "ThreadNVGPU.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/ProcessInfo.h"
#include "llvm/ADT/StringMap.h"

#include "cudadebugger.h"

#include <functional>

namespace lldb_private::lldb_server {

/// Manages GPU process debugging and thread execution state.
///
/// This class manages all the threads in the GPU, its memory, and its overall
/// state. It extends NativeProcessProtocol, which was meant for CPU
/// processes, but fortunately it provides most of the abstractions we need to
/// manage the GPU.
///
/// How the GPU process stops:
///
/// The state of the GPU process is what lldb-server has told the client. It
/// runs from Resume until a stop is reported, which sends the client a stop
/// reply. The devices have a state of their own. The driver suspends every
/// device when a thread hits a breakpoint or raises an exception, and when
/// Interrupt asks it to, and reports the suspension with
/// CUDBG_EVENT_ALL_DEVICES_SUSPENDED. A module load is reported as a dyld stop
/// that leaves the devices running.
///
/// LLDBServerPluginNVGPU::DrainSyncEvents handles the driver's events only
/// while the GPU process runs, that is, while the client waits for a stop.
/// Reporting a stop ends the drain, and Resume drains the events that waited.
/// Handling an event can report a stop, and over a stop that the client
/// already has, it would rewrite that stop without telling the client. And
/// acknowledgeSyncEvents acknowledges every event read so far, so the only way
/// to keep an event unacknowledged, as a dyld stop keeps its module load to
/// hold the CPU thread that loads the module, is to stop reading at it.
///
/// Only a thread whose own event stopped the GPU reports a stop reason: a
/// breakpoint, an exception, or dyld for the first thread at a module load.
/// The other threads stopped because their device did, and report none, as
/// the other threads of a stopped CPU process do. Tools act on each thread
/// that has a reason: lldb-dap sends a stopped event for each, for example.
///
/// The client takes the next stop after its interrupt for the answer,
/// whatever its reason. When the suspension that Interrupt requested is the
/// answer, the selected thread reports SIGSTOP, as a CPU process interrupted
/// the same way does. A breakpoint or an exception that comes first answers
/// it instead, and the driver sends no event for the suspension. A dyld stop
/// that comes first leaves the devices running, so the suspension's event
/// still comes, after the client has resumed, and only resumes the devices:
/// see InterruptState. lldb also interrupts the GPU to send a packet
/// while it runs, such as one that sets a breakpoint, and resumes it at once.
/// lldb-server cannot tell that from the user's interrupt, so an interrupt
/// leaves the CPU running, while a breakpoint or an exception stops it too,
/// unless NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP is set.
///
/// When lldb attaches to a running application, the GPU stops once the driver
/// has attached: StopAfterAttach suspends the devices as Interrupt does, and
/// the selected thread stops with SIGSTOP, as a CPU process does when lldb
/// attaches to it. This stop stops the CPU too. A dyld stop does not answer
/// it.
///
/// Halt reports a stop, as in the other GPU plug-ins, and never suspends the
/// devices.
class ProcessNVGPU : public NativeProcessProtocol {
public:
  /// Class that manages the creation of NVGPU objects.
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

  /// Constructor for the NVIDIA GPU process.
  ///
  /// Initializes the GPU process with a fake stopped state to allow debugger
  /// connection. Creates an initial thread in stopped state as required by
  /// the GDB remote protocol.
  ///
  /// \param[in] pid
  ///     Process ID for the GPU process.
  ///
  /// \param[in] delegate
  ///     Delegate for handling process events and notifications.
  ///
  /// \param[in] main_loop
  ///     The main loop of the GPU's gdb-remote server, for work that has to
  ///     wait until the request being handled is answered.
  ProcessNVGPU(lldb::pid_t pid, NativeDelegate &delegate, MainLoop &main_loop);

  ~ProcessNVGPU() override;

  /// Start using \a api and set up the state of the devices it reports.
  ///
  /// \param[in] api
  ///     The initialized debugger API.
  ///
  /// \return
  ///     An error if the debugger API cannot describe the devices. The caller
  ///     then has to release the API with ClearDebuggerAPI.
  llvm::Error SetDebuggerAPI(CUDADebuggerAPI &api);

  CUDBGAPI GetDebuggerAPI() const { return m_api; }

  /// Drop the debugger API pointer just before the plugin finalizes and
  /// destroys it, so this process cannot be left holding a dangling table.
  void ClearDebuggerAPI() { m_api = nullptr; }

  /// Register a notifier that Resume calls to drain the sync events that
  /// waited while the GPU process was stopped. The driver notifies only for
  /// newly enqueued events, so nothing else would drain them.
  void SetSyncEventDrainNotifier(std::function<void()> notifier) {
    m_sync_event_drain_notifier = std::move(notifier);
  }

  /// Register how to report a call into the debugger API that fails while a
  /// resume or interrupt request is handled. The plug-in reports it as it
  /// reports an internal error.
  void SetDebuggerAPIFaultHandler(
      std::function<void(std::string description)> handler) {
    m_api_fault_handler = std::move(handler);
  }

  /// The API version this session runs at: the lesser of the compiled and the
  /// live driver's versions. Gate calls to entry points newer than the
  /// major's baseline on this -- `if (GetAPIVersion().AtLeast(13, 4, 0))
  /// api->newCall(...)` -- since a driver older than the compiled header lacks
  /// those (appended) slots and reading the pointer would be out of bounds.
  nvgpu::CudbgApiVersion GetAPIVersion() const { return m_api_version; }

  /// Mark the GPU process and its threads running, notify the delegate, then
  /// resume GPU execution. Without a debugger API, report a stop on the
  /// fallback thread again instead of resuming. If a call into the API fails,
  /// report the failure through the handler from SetDebuggerAPIFaultHandler
  /// once the server has answered the request.
  ///
  /// \param[in] resume_actions
  ///     List of resume actions to apply to threads.
  ///
  /// \return
  ///     Status indicating success or failure of the resume operation.
  Status Resume(const ResumeActionList &resume_actions) override;

  /// Report that the GPU stopped: mark the threads that still run as stopped,
  /// and change the state of the process to stopped, which tells the client.
  /// As in the other GPU plug-ins, this does not suspend the devices, which the
  /// driver or Interrupt must already have done.
  ///
  /// \return
  ///     Always success.
  Status Halt() override;

  /// Detach from the GPU process.
  ///
  /// \return
  ///     Status indicating success or failure of the detach operation.
  Status Detach() override;

  /// Send a signal to the GPU process.
  ///
  /// \param[in] signo
  ///     Signal number to send.
  ///
  /// \return
  ///     Status indicating success or failure of the signal operation.
  Status Signal(int signo) override;

  /// Stop the GPU at the client's request. The devices are suspended in the
  /// background, and the debugger API reports the suspension with
  /// CUDBG_EVENT_ALL_DEVICES_SUSPENDED, whose handler reports the stop: see
  /// OnAllDevicesSuspended. Suspending the devices this way needs a CUDA 13.2
  /// or newer driver. With an older one, no stop answers the interrupt, and
  /// the client ends the GPU process with "lost connection" when its
  /// interrupt times out.
  ///
  /// \return
  ///     Success. A failure to suspend the devices is reported as a fault of
  ///     the debugger API instead, since the client takes a failed interrupt
  ///     for a lost connection.
  Status Interrupt() override;

  /// Stop the GPU once a late attach has finished, so that lldb shows the
  /// threads of the kernels that were already running. As Interrupt does,
  /// this asks the debugger API to suspend every device in the background,
  /// and OnAllDevicesSuspended reports the stop. Suspending the devices this
  /// way needs a CUDA 13.2 or newer driver. With an older one, the GPU goes on
  /// running.
  ///
  /// \return
  ///     An error if the debugger API fails to suspend the devices.
  llvm::Error StopAfterAttach();

  /// Terminate the GPU process.
  ///
  /// \return
  ///     Status indicating success or failure of the kill operation.
  Status Kill() override;

  /// Read memory from the GPU address space.
  ///
  /// \param[in] addr
  ///     GPU memory address to read from.
  ///
  /// \param[out] buf
  ///     Buffer to store the read data.
  ///
  /// \param[in] size
  ///     Number of bytes to read.
  ///
  /// \param[out] bytes_read
  ///     Number of bytes actually read.
  ///
  /// \return
  ///     Status indicating success or failure of the read operation.
  Status ReadMemory(const ProcessAddress &process_addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  /// Write memory to the GPU address space.
  ///
  /// \param[in] addr
  ///     GPU memory address to write to.
  ///
  /// \param[in] buf
  ///     Buffer containing data to write.
  ///
  /// \param[in] size
  ///     Number of bytes to write.
  ///
  /// \param[out] bytes_written
  ///     Number of bytes actually written.
  ///
  /// \return
  ///     Status indicating success or failure of the write operation.
  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  /// Get the address of shared library information structure.
  ///
  /// \return
  ///     Address of the shared library info structure, or
  ///     LLDB_INVALID_ADDRESS if not available.
  lldb::addr_t GetSharedLibraryInfoAddress() override;

  /// Create or update the list of ThreadNVGPU objects of the GPU.
  ///
  /// This method scans the GPU's execution state and creates thread objects
  /// for all active threads, updating existing ones as needed.
  ///
  /// \return
  ///     Number of threads currently active on the GPU.
  size_t UpdateThreads() override;

  /// Get the architecture specification for this GPU process.
  ///
  /// \return
  ///     Reference to the ArchSpec describing the GPU architecture.
  const ArchSpec &GetArchitecture() const override;

  /// Get the architecture of the GPU code this plugin debugs, for use before
  /// any ProcessNVGPU exists.
  ///
  /// \return
  ///     64-bit NVPTX, built from the ELF mapping the client uses to decode
  ///     cubins so both sides report the same triple. CUDA 13 produces only
  ///     64-bit cubins.
  static ArchSpec GetNVPTXArchitecture();

  /// Set a breakpoint at the specified address.
  ///
  /// \param[in] addr
  ///     Address where the breakpoint should be set.
  ///
  /// \param[in] size
  ///     Size of the breakpoint in bytes.
  ///
  /// \param[in] hardware
  ///     True if this should be a hardware breakpoint, false for software.
  ///
  /// \return
  ///     Status indicating success or failure of the breakpoint operation.
  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  /// Remove a breakpoint from the GPU at the specified address.
  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware) override;

  /// Get auxiliary vector data for the GPU process.
  ///
  /// \return
  ///     ErrorOr containing memory buffer with aux data, or error if
  ///     unavailable.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  /// Get the file specification for a loaded module.
  ///
  /// \param[in] module_path
  ///     Path to the module to look up.
  ///
  /// \param[out] file_spec
  ///     File specification to populate with module information.
  ///
  /// \return
  ///     Status indicating success or failure of the lookup operation.
  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  /// Get the load address for a specific file.
  ///
  /// \param[in] file_name
  ///     Name of the file to get load address for.
  ///
  /// \param[out] load_addr
  ///     Variable to store the load address.
  ///
  /// \return
  ///     Status indicating success or failure of the lookup operation.
  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  /// Get process information for this GPU process.
  ///
  /// \param[out] info
  ///     ProcessInstanceInfo to populate with process details.
  ///
  /// \return
  ///     True if process info was successfully retrieved, false otherwise.
  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  /// \return the requested list of dynamic library infos.
  ///
  /// \param[in] args
  ///     Arguments containing criteria for library selection.
  ///
  /// \return
  ///     Optional response containing dynamic library information, or nullopt
  ///     if no libraries match the criteria.
  std::optional<GPUDynamicLoaderResponse>
  GetGPUDynamicLoaderLibraryInfos(const GPUDynamicLoaderArgs &args) override;

  /// \return true if there are new cubins since the last time they were
  /// requested.
  ///
  /// Check whether new CUDA binary files (cubins) have been loaded since
  /// the last query, indicating that dynamic libraries may need to be
  /// reported.
  ///
  /// \return
  ///     True if unreported libraries exist, false otherwise.
  bool HasUnreportedLibraries() const;

  /// The stop that OnAllDevicesSuspended reports.
  enum class SuspendStop {
    /// No stop: a dyld stop answered the interrupt that requested the
    /// suspension.
    None,
    /// A stop that answers an interrupt, which no thread stopped for on its
    /// own. lldb interrupts the GPU for the user, and also to send a packet
    /// while it runs, such as to set a breakpoint, after which it resumes the
    /// GPU at once. lldb-server cannot tell the two apart.
    Interrupted,
    /// A stop that the GPU made on its own, such as for a breakpoint or an
    /// exception, or that ends a late attach.
    Stopped,
  };

  /// Handle the AllDevicesSuspended event, which the driver sends once it has
  /// suspended every device. The event drain handles it only while the GPU
  /// process is running, that is, while the client waits for a stop.
  ///
  /// Reads the state of the devices and reports a stop. While the client
  /// waits for the stop that answers its interrupt, or StopAfterAttach waits
  /// for the stop that ends a late attach, the selected thread stops with
  /// SIGSTOP, unless a thread hit a breakpoint or raised an exception.
  /// The event for the suspension that Interrupt requested reports no stop
  /// when a dyld stop answered the interrupt first, and resumes the devices
  /// instead, since the client has resumed the GPU from that stop.
  ///
  /// \param[in] event
  ///     Event data containing information about the suspended devices.
  ///
  /// \return
  ///     The stop reported, or an error if the state of the devices cannot be
  ///     read or decoded, or the devices cannot be resumed. The threads are
  ///     then partially updated, so the caller has to stop with
  ///     OnDebuggerAPIFault, which reports the fallback thread alone.
  llvm::Expected<SuspendStop> OnAllDevicesSuspended(
      const CUDBGEvent::cases_st::allDevicesSuspended_st &event);

  /// Stop using the debugger API after it faults, by reporting an internal
  /// error, failing a call, or reporting a device state that cannot be
  /// decoded, and report an exception-class stop with \a description on the
  /// fallback thread alone. From then on, requests that need the API fail with
  /// \a description, and each resume reports this stop again.
  ///
  /// \param[in] description
  ///     What went wrong, which the stop shows.
  void OnDebuggerAPIFault(std::string description);

  /// \return
  ///     True once the debugger API has faulted. Calling into it after that can
  ///     wedge or crash the session.
  bool HasDebuggerAPIFaulted() const { return m_api_fault.has_value(); }

  /// Handle the ElfImageLoaded event.
  ///
  /// Processes the event indicating that a new ELF image (cubin) has been
  /// loaded, adding it to the list of available libraries. A cubin that does
  /// not parse is logged and skipped, so HasUnreportedLibraries can stay false.
  ///
  /// \param[in] event
  ///     Event data containing information about the loaded ELF image.
  ///
  /// \return
  ///     An error if the debugger API cannot provide the ELF image.
  llvm::Error
  OnElfImageLoaded(const CUDBGEvent::cases_st::elfImageLoaded_st &event);

  /// Report a stop reason for the dynamic loader. This is used to notify the
  /// client that it should fetch the new libraries. It uses a fake stop
  /// as described in the comment for m_is_faking_a_stop_for_dyld.
  ///
  /// This creates a temporary stop state to allow the client to process
  /// dynamic library changes without suspending the devices.
  void ReportDyldStop();

  /// \return the registry of all devices.
  DeviceStateRegistry &GetAllDevices() { return m_devices; }

  /// \return true if the device registry contains a complete snapshot for the
  /// current real GPU stop.
  bool IsDeviceStateCurrent() const;

  std::vector<AddressSpaceInfo> GetAddressSpaces() override;

  /// Handle the native process exit event.
  ///
  /// \param[in] exit_status
  ///     The exit status of the native process.
  void OnNativeProcessExit(const WaitStatus &exit_status);

  /// \return the list of structured data plugins for this lldb-server Process.
  std::vector<std::string> GetStructuredDataPlugins() override;

private:
  friend class Manager;

  /// Accessor for m_api.
  ///
  /// \return
  ///     The CUDA debugger API, or an error that says why there is none: it is
  ///     not set up yet, was released, or faulted.
  llvm::Expected<const CUDBGAPI_st &> GetCudaAPI();

  /// Report an exception-class stop on the fallback thread alone, without
  /// calling into the debugger API, for when there is none or it can no longer
  /// be trusted.
  void ReportFallbackStop(llvm::StringRef description);

  /// Stop using the debugger API after a call into it fails while a resume or
  /// interrupt request is handled, and report \a description through the
  /// handler from SetDebuggerAPIFaultHandler once the server has answered the
  /// request: the client takes a failed request for a lost connection.
  void QueueDebuggerAPIFault(std::string description);

  /// Resume every device, with a single call on drivers that have one, CUDA
  /// 13.2 and newer. A device that is already running is not an error.
  ///
  /// \param[in] api
  ///     The debugger API to resume the devices through.
  ///
  /// \return
  ///     An error if the debugger API fails to resume a device.
  llvm::Error ResumeDevices(const CUDBGAPI_st &api);

  /// Utility for the Manager class to set the launch info for the GPU.
  ///
  /// \param[in] launch_info
  ///     Launch information to apply to this GPU process.
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);

  /// Get a range object that allows iterating over threads with automatic
  /// casting to ThreadNVGPU&.
  ///
  /// \return
  ///     Range object for iterating over GPU threads.
  NVGPUThreadRange GPUThreads();

  ArchSpec m_arch;
  /// This contains some launch/attach/runtime information for the GPU. We are
  /// using the class used for CPU processes for this for simplicity.
  ProcessInstanceInfo m_process_info;
  CUDBGAPI m_api;
  /// What went wrong when the debugger API faulted, after which this process
  /// stopped using it. See OnDebuggerAPIFault.
  std::optional<std::string> m_api_fault;
  /// API version in use with the live driver, for runtime gating of
  /// version-specific API calls.
  nvgpu::CudbgApiVersion m_api_version;

  /// The list of all the cubins loaded by the GPU.
  std::vector<GPUDynamicLoaderLibraryInfo> m_all_libraries;
  /// The list of all the cubins loaded by the GPU that haven't been reported
  /// to the client yet.
  std::vector<GPUDynamicLoaderLibraryInfo> m_unreported_libraries;
  /// How many cubins have been given each base name, so that the next one gets
  /// a "-<n>" suffix.
  llvm::StringMap<uint32_t> m_cubin_name_counts;

  /// A flag that indicates that we are faking a stop to the client to report
  /// dyld events. A fake stop means that we don't actually stop the GPU, but
  /// we stop processing more APU events until we have resumed after the dyld
  /// event.
  bool m_is_faking_a_stop_for_dyld = false;

  /// Where an interrupt stands between Interrupt and the event for the
  /// suspension it requested. The client takes the next stop after an
  /// interrupt as its answer, whatever its reason.
  enum class InterruptState {
    /// No suspension that Interrupt requested is on its way.
    None,
    /// The client waits for the stop that answers the interrupt.
    Waiting,
    /// A dyld stop answered the interrupt, and the event for the suspension
    /// is still to come. It answers nothing, and only resumes the devices.
    StaleSuspension,
  };
  InterruptState m_interrupt_state = InterruptState::None;

  /// Set by StopAfterAttach until the next suspension of the devices is
  /// reported, which ends the attach.
  bool m_attach_stop_pending = false;

  /// See SetSyncEventDrainNotifier.
  std::function<void()> m_sync_event_drain_notifier;

  /// See SetDebuggerAPIFaultHandler.
  std::function<void(std::string description)> m_api_fault_handler;

  MainLoop &m_main_loop;

  /// Cached device information, updated at every real GPU stop.
  DeviceStateRegistry m_devices;

  /// A fallback thread that is used to report the state of the GPU when no
  /// actual threads are active.
  ThreadNVGPU m_fallback_thread;
};

} // namespace lldb_private::lldb_server

#endif
