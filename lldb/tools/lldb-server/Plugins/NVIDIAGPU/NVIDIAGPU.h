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
#include "DeviceInformation.h"
#include "ThreadNVIDIAGPU.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/ProcessInfo.h"

#include "cudadebugger.h"
#include <unordered_map>

namespace lldb_private::lldb_server {

/// Manages GPU process debugging and thread execution state.
///
/// This class manages all the threads in the GPU, its memory, and its overall
/// state. It extends NativeProcessProtocol, which was meant for CPU
/// processes, but fortunately it provides most of the abstractions we need to
/// manage the GPU.
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
  NVIDIAGPU(lldb::pid_t pid, NativeDelegate &delegate);

  void SetDebuggerAPI(CUDADebuggerAPI &api) { m_api = api.GetRawAPI(); }

  CUDBGAPI GetDebuggerAPI() const { return m_api; }

  /// Resume the GPU, change its state and the state of each thread, then
  /// report the state change to the delegate, which will notify the client.
  ///
  /// \param[in] resume_actions
  ///     List of resume actions to apply to threads.
  ///
  /// \return
  ///     Status indicating success or failure of the resume operation.
  Status Resume(const ResumeActionList &resume_actions) override;

  /// Perform \a Halt() and set the stop reason of the first thread to dyld.
  ///
  /// This is used to signal that dynamic libraries have been loaded/unloaded
  /// and the client should update its symbol tables.
  ///
  /// \return
  ///     Status indicating success or failure of the halt operation.
  Status HaltDueToDyld();

  /// Halt the GPU but don't change its state. That requires a call to
  /// \a ChangeStateToStopped().
  ///
  /// \return
  ///     Status indicating success or failure of the halt operation.
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

  /// Interrupt execution of the GPU process.
  ///
  /// \return
  ///     Status indicating success or failure of the interrupt operation.
  Status Interrupt() override;

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
  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
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

  /// Create or update the list of ThreadNVIDIAGPU objects of the GPU.
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

  /// Change the state of the process to stopped and notify delegates. It
  /// assumes that the GPU is in fact stopped.
  ///
  /// This method updates internal state and notifies any registered delegates
  /// that the process has stopped, allowing them to take appropriate action.
  void ChangeStateToStopped();

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

  /// Handle the AllDevicesSuspended event.
  ///
  /// Processes the event indicating that all GPU devices have been suspended,
  /// updating internal state and notifying clients as appropriate.
  ///
  /// \param[in] event
  ///     Event data containing information about the suspended devices.
  void OnAllDevicesSuspended(
      const CUDBGEvent::cases_st::allDevicesSuspended_st &event);

  /// Handle the ElfImageLoaded event.
  ///
  /// Processes the event indicating that a new ELF image (cubin) has been
  /// loaded, adding it to the list of available libraries.
  ///
  /// \param[in] event
  ///     Event data containing information about the loaded ELF image.
  void OnElfImageLoaded(const CUDBGEvent::cases_st::elfImageLoaded_st &event);

  /// Report a stop reason for the dynamic loader. This is used to notify the
  /// client that it should fetch the new libraries. It uses a fake stop
  /// as described in the comment for m_is_faking_a_stop_for_dyld.
  ///
  /// This creates a temporary stop state to allow the client to process
  /// dynamic library changes without actually halting GPU execution.
  void ReportDyldStop();

  /// \return the DeviceInformation object for the given device id.
  DeviceInformation &GetDeviceInformation(int device_id);

  std::vector<AddressSpaceInfo> GetAddressSpaces() override;

  Status ReadMemoryWithSpace(lldb::addr_t addr, uint64_t addr_space,
                             NativeThreadProtocol *thread, void *buf,
                             size_t size, size_t &bytes_readn) override;

private:
  friend class Manager;

  /// Accessor for m_api that fails if it's not initialized.
  ///
  /// \return
  ///     Reference to the CUDA debugger API structure.
  ///
  /// \throws
  ///     Assertion failure if the API is not properly initialized.
  const CUDBGAPI_st &GetCudaAPI();

  /// Utility for the Manager class to set the launch info for the GPU.
  ///
  /// \param[in] launch_info
  ///     Launch information to apply to this GPU process.
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);

  /// Get a range object that allows iterating over threads with automatic
  /// casting to ThreadNVIDIAGPU&.
  ///
  /// \return
  ///     Range object for iterating over GPU threads.
  GPUThreadRange GPUThreads();

  ArchSpec m_arch;
  /// This contains some launch/attach/runtime information for the GPU. We are
  /// using the class used for CPU processes for this for simplicity.
  ProcessInstanceInfo m_process_info;
  CUDBGAPI m_api;

  /// The list of all the cubins loaded by the GPU.
  std::vector<GPUDynamicLoaderLibraryInfo> m_all_libraries;
  /// The list of all the cubins loaded by the GPU that haven't been reported
  /// to the client yet.
  std::vector<GPUDynamicLoaderLibraryInfo> m_unreported_libraries;

  /// A flag that indicates that we are faking a stop to the client to report
  /// dyld events. A fake stop means that we don't actually stop the GPU, but
  /// we stop processing more APU events until we have resumed after the dyld
  /// event.
  bool m_is_faking_a_stop_for_dyld = false;

  /// A map of device id to DeviceInformation object.
  std::unordered_map<int, DeviceInformation> m_device_information;
};

} // namespace lldb_private::lldb_server

#endif
