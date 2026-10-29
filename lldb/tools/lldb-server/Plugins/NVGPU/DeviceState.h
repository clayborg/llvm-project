//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_DEVICESTATE_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_DEVICESTATE_H

#include "../Utils/Bitset.h"
#include "ThreadNVGPU.h"
#include "cudadebugger.h"
#include "lldb/Utility/NVGPU/ThreadCoords.h"
#include "lldb/Utility/Stream.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lldb_private::lldb_server {

using ThreadIdx = nvgpu::Dim3u;
using BlockIdx = nvgpu::Dim3u;

/// Represents the state of a single CUDA thread.
class ThreadState {
public:
  /// Construct a thread state with physical coordinates.
  ///
  /// \param[in] gpu
  ///     The GPU that this thread state belongs to.
  /// \param[in] physical
  ///     These physical coordinates won't change for the lifetime of this
  ///     object.
  ThreadState(ProcessNVGPU &gpu, nvgpu::PhysicalCoords physical,
              WarpState &warp_state);

  /// Non-copyable.
  ThreadState(const ThreadState &) = delete;
  ThreadState &operator=(const ThreadState &) = delete;

  /// We want to make ThreadState non-movable and non-copyable, but as its
  /// stored in a vector, we need to define a move constructor. This move
  /// constructor must abort if invoked.
  ThreadState(ThreadState &&);

  /// Decode thread information from a buffer received from CUDA debugger API.
  ///
  /// \param[in] buffer
  ///     Buffer containing thread information data.
  /// \param[in] device_info
  ///     Device information structure.
  /// \param[in] device_info_sizes
  ///     Size information for device data structures.
  /// \param[in] thread_attributes_present
  ///     Whether thread attributes are present in the buffer.
  /// \param[in] warp_exception
  ///     Optional exception information from the containing warp.
  /// \param[in] thread_idx
  ///     The 3D thread index within the thread block.
  /// \param[in] at_breakpoint
  ///     Whether the parent warp is at a breakpoint.
  /// \param[in] is_active
  ///     Whether the thread is in the active thread set.
  ///
  /// \return
  ///     The number of bytes consumed from the buffer.
  size_t DecodeThreadInfoBuffer(uint8_t *buffer,
                                const CUDBGDeviceInfo &device_info,
                                const CUDBGDeviceInfoSizes &device_info_sizes,
                                bool thread_attributes_present,
                                const ExceptionInfo *warp_exception,
                                const ThreadIdx &thread_idx, bool at_breakpoint,
                                bool is_active);

  /// \return
  ///     True if this thread is valid in the GPU, false otherwise.
  bool IsValid() const { return m_is_valid; }

  /// Set the validity status of this object.
  void SetIsValid(bool is_valid) { m_is_valid = is_valid; }

  /// Output thread state information to a stream for debugging.
  void Dump(Stream &s);

  /// \return
  ///     A reference to the immutable physical coordinates.
  const nvgpu::PhysicalCoords &GetPhysicalCoords() const {
    return m_physical_coords;
  }

  /// \return
  ///     The 3D thread index, or nullopt if it is unknown.
  const std::optional<ThreadIdx> &GetThreadIdx() const { return m_thread_idx; }

  /// \return
  ///     A snapshot of the currently available CUDA thread coordinates.
  nvgpu::ThreadCoords GetThreadCoords() const;

  /// \return
  ///     Whether this thread is in the active thread set, or nullopt if it is
  ///     unknown.
  std::optional<bool> IsActive() const { return m_is_active; }

  /// Set whether this thread is in the active thread set.
  void SetIsActive(bool is_active) { m_is_active = is_active; }

  /// \return
  ///     Exception information of this thread, or nullptr if no exception
  ///     occurred.
  const ExceptionInfo *GetException() const { return m_exception; }

  /// \return
  ///     True if the thread has an exception, false otherwise.
  bool HasException() const { return m_exception != nullptr; }

  /// \return
  ///     The program counter value for this thread.
  uint64_t GetPC() const { return m_pc; }

  /// \return
  ///     The ThreadNVGPU object associated with this thread.
  ThreadNVGPU &GetThreadNVGPU() { return m_thread_nvgpu; }

  /// \return
  ///     The warp state associated with this thread.
  WarpState &GetWarpState() const { return *m_warp_state; }

private:
  /// Whether this thread is valid in the GPU.
  bool m_is_valid = false;

  /// Whether this thread is in the active thread set, if known.
  std::optional<bool> m_is_active;

  /// The program counter value for this thread.
  uint64_t m_pc = 0;

  /// Exception information if the parent warp encountered an exception.
  const ExceptionInfo *m_exception = nullptr;

  /// The 3D thread index within the thread block, if known.
  std::optional<ThreadIdx> m_thread_idx;

  /// The immutable physical coordinates of this thread on the device.
  const nvgpu::PhysicalCoords m_physical_coords;

  /// The ThreadNVGPU object associated with this thread. This object is the
  /// interface LLGS wants to interact with.
  ThreadNVGPU m_thread_nvgpu;

  WarpState *m_warp_state;
};

/// Represents the state of a CUDA warp.
class WarpState {
public:
  /// Construct a warp state with the specified parameters.
  ///
  /// \param[in] gpu
  ///     The GPU that this warp state belongs to.
  /// \param[in] num_threads
  ///     The number of threads in this warp.
  /// \param[in] device_id
  ///     The device ID where this warp resides.
  /// \param[in] sm_id
  ///     The streaming multiprocessor ID where this warp resides.
  /// \param[in] warp_id
  ///     The warp ID within the streaming multiprocessor.
  /// \param[in] sm_state
  ///     The streaming multiprocessor state that this warp belongs to.
  WarpState(ProcessNVGPU &gpu, uint32_t num_threads, uint32_t device_id,
            uint32_t sm_id, uint32_t warp_id, SMState &sm_state);

  /// Decode warp information from a buffer received from CUDA debugger API.
  ///
  /// \param[in] buffer
  ///     Buffer containing warp information data.
  /// \param[in] device_info
  ///     Device information structure.
  /// \param[in] device_info_sizes
  ///     Size information for device data structures.
  /// \param[in] get_grid_info
  ///     Function to retrieve grid information by grid ID.
  /// \param[in] at_breakpoint
  ///     Whether the warp is at a breakpoint.
  ///
  /// \return
  ///     The number of bytes consumed from the buffer, or an error if the
  ///     warp information is inconsistent or its grid information is not
  ///     available.
  llvm::Expected<size_t> DecodeWarpInfoBuffer(
      uint8_t *buffer, const CUDBGDeviceInfo &device_info,
      const CUDBGDeviceInfoSizes &device_info_sizes,
      std::function<llvm::Expected<const CUDBGGridInfo &>(uint64_t)>
          get_grid_info,
      bool at_breakpoint);

  /// \return
  ///     True if the warp is valid in the GPU, false otherwise.
  bool IsValid() const { return m_is_valid; }

  /// Set the validity status of this warp state.
  void SetIsValid(bool is_valid) { m_is_valid = is_valid; }

  /// Output warp state information to a stream for debugging.
  void Dump(Stream &s);

  /// \return
  ///     True if the warp has an exception, false otherwise.
  bool HasException() const { return m_exception.has_value(); }

  /// \return
  ///     A reference to the collection of threads.
  llvm::MutableArrayRef<ThreadState> GetThreads() const { return m_threads; }

  /// \return
  ///     An iterator to the collection of valid threads.
  auto GetValidThreads() {
    return llvm::make_filter_range(
        m_threads, [](ThreadState &thread) { return thread.IsValid(); });
  }

  /// Get the number of regular registers currently used by this warp. It is
  /// cached until the warp state changes.
  ///
  /// \return
  ///     The number of regular registers currently used by this warp, or an
  ///     error if the debugger API cannot read it.
  llvm::Expected<uint32_t> GetCurrentNumRegularRegisters();

  /// Get the registers for this warp. The result is cached until the warp state
  /// changes. A read that fails is logged and leaves its registers invalid.
  ///
  /// \return
  ///     A reference to the registers for this warp.
  const WarpSharedRegisterCache &GetRegisters();

  const std::optional<BlockIdx> &GetBlockIdx() const {
    return m_logical_coords.block_idx;
  }

  /// \return
  ///     The logical coordinates shared by threads in this warp. The thread
  ///     index is always disengaged.
  const nvgpu::LogicalCoords &GetLogicalCoords() const {
    return m_logical_coords;
  }

  /// \return
  ///     A reference to the streaming multiprocessor state that this warp
  ///     belongs to.
  SMState &GetSMState() const { return *m_sm_state; }

  /// \return
  ///     The immutable physical coordinates of this warp. The lane is always
  ///     disengaged.
  const nvgpu::PhysicalCoords &GetPhysicalCoords() const {
    return m_physical_coords;
  }

  /// \return
  ///     The grid ID for this warp, or nullopt if it is unknown.
  const std::optional<uint64_t> &GetGridId() const {
    return m_logical_coords.grid_id;
  }

private:
  /// Whether this warp is valid in the GPU.
  bool m_is_valid = false;

  /// Exception information if this warp encountered an exception.
  std::optional<ExceptionInfo> m_exception;

  /// All threads in this warp.
  std::vector<ThreadState> m_threads;

  /// The number of regular registers currently available for each thread of
  /// this warp.
  std::optional<uint32_t> m_current_num_regular_registers;

  /// The registers shared by all threads in this warp.
  WarpSharedRegisterCache m_shared_registers{};

  /// Whether the registers for this warp have been calculated.
  bool m_regs_calculated = false;

  /// Logical coordinates shared by threads in this warp. The thread index is
  /// always disengaged.
  nvgpu::LogicalCoords m_logical_coords;

  /// The immutable physical coordinates of this warp on the device.
  const nvgpu::PhysicalCoords m_physical_coords;

  /// The streaming multiprocessor state that this warp belongs to.
  SMState *m_sm_state;
};

/// Represents the state of a CUDA Streaming Multiprocessor (SM).
class SMState {
public:
  /// Construct an SM state with the specified parameters.
  ///
  /// \param[in] gpu
  ///     The GPU that this SM state belongs to.
  /// \param[in] num_warps
  ///     The number of warps in this SM.
  /// \param[in] num_threads_per_warp
  ///     The number of threads per warp.
  /// \param[in] device_id
  ///     The device ID where this SM resides.
  /// \param[in] sm_id
  ///     The SM ID within the device.
  /// \param[in] device_state
  ///     The device state that this SM belongs to.
  SMState(ProcessNVGPU &gpu, uint32_t num_warps, uint32_t num_threads_per_warp,
          uint32_t device_id, uint32_t sm_id, DeviceState &device_state);

  /// Decode SM information from a buffer received from CUDA debugger API.
  ///
  /// \param[in] buffer
  ///     Buffer containing SM information data.
  /// \param[in] device_info
  ///     Device information structure.
  /// \param[in] device_info_sizes
  ///     Size information for device data structures.
  /// \param[in] get_grid_info
  ///     Function to retrieve grid information by grid ID.
  ///
  /// \return
  ///     The number of bytes consumed from the buffer, or an error if a warp
  ///     cannot be decoded.
  llvm::Expected<size_t> DecodeSMInfoBuffer(
      uint8_t *buffer, const CUDBGDeviceInfo &device_info,
      const CUDBGDeviceInfoSizes &device_info_sizes,
      std::function<llvm::Expected<const CUDBGGridInfo &>(uint64_t)>
          get_grid_info);

  /// Set the activity status of this SM.
  void SetIsActive(bool is_active);

  /// \return
  ///     True if the SM is active in the GPU, false otherwise.
  bool IsActive() const { return m_is_active; }

  /// Output SM state information to a stream for debugging.
  void Dump(Stream &s);

  /// \return
  ///     A reference to the collection of warps.
  llvm::MutableArrayRef<WarpState> GetWarps() { return m_warps; }

  /// \return
  ///     An iterator to the collection of valid warps.
  auto GetValidWarps() {
    return llvm::make_filter_range(
        m_warps, [](WarpState &warp) { return warp.IsValid(); });
  }

  /// \return
  ///     A reference to the device state that this SM belongs to.
  DeviceState &GetDeviceState() const { return *m_device_state; }

private:
  /// Whether this SM is currently active in the GPU.
  bool m_is_active;

  /// All warps in this SM.
  std::vector<WarpState> m_warps;

  /// The device state that this SM belongs to.
  DeviceState *m_device_state;
};

/// Manages and caches information about a particular CUDA device.
class DeviceState {
public:
  /// Construct a device state manager for the specified device. It has no SMs
  /// until \a Initialize succeeds.
  ///
  /// \param[in] gpu
  ///     The GPU that this device state belongs to.
  /// \param[in] device_id
  ///     The ID of the CUDA device to manage.
  DeviceState(ProcessNVGPU &gpu, uint32_t device_id);

  /// Query the information that is constant for the lifetime of the device,
  /// and create the state of its SMs. The SMs point back at this object, so it
  /// cannot be set up elsewhere and moved into place: DeviceStateRegistry
  /// constructs it where it stays, then calls this once.
  ///
  /// \param[in] gpu
  ///     The GPU that this device state belongs to.
  ///
  /// \return
  ///     An error if the debugger API cannot describe the device.
  llvm::Error Initialize(ProcessNVGPU &gpu);

  /// \return
  ///     The ID of the CUDA device.
  uint32_t GetDeviceId() const { return m_device_id; }

  /// Get the number of R registers available on this device. This is cached for
  /// the lifetime of the device.
  ///
  /// \return
  ///     The number of R registers for the device, or an error if the debugger
  ///     API cannot report it.
  llvm::Expected<size_t> GetMaxNumSupportedRegularRegister();

  /// Get the number of predicate registers available on this device. This is
  /// cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of predicate registers for the device, or an error if the
  ///     debugger API cannot report it.
  llvm::Expected<size_t> GetNumPredicateRegisters();

  /// Get the number of uniform registers available on this device. This is
  /// cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of uniform registers for the device, or an error if the
  ///     debugger API cannot report it.
  llvm::Expected<size_t> GetNumUniformRegisters();

  /// Get the number of uniform predicate registers available on this device.
  /// This is cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of uniform predicate registers for the device, or an error
  ///     if the debugger API cannot report it.
  llvm::Expected<size_t> GetNumUniformPredicateRegisters();

  /// Update all device state information from the CUDA debugger API.
  ///
  /// This method performs a batch update of all dynamic state information
  /// including SM states, warp states, and thread states.
  ///
  /// \return
  ///     An error if the debugger API cannot report the state, or the state
  ///     cannot be decoded. The state is then partially updated.
  llvm::Error BatchUpdate();

  /// Output device state information to a stream for debugging.
  void Dump(Stream &s);

  /// Get grid information for the specified grid ID.
  ///
  /// \param[in] grid_id
  ///     The ID of the grid to retrieve information for.
  ///
  /// \return
  ///     A reference to the grid information structure, or an error if the
  ///     debugger API cannot report it.
  llvm::Expected<const CUDBGGridInfo &> GetGridInfo(uint64_t grid_id);

  /// \return
  ///     A reference to the collection of SM states.
  llvm::MutableArrayRef<SMState> GetSMs() { return m_sms; }

  /// \return
  ///     An iterator to the collection of active SMs.
  auto GetActiveSMs() {
    return llvm::make_filter_range(m_sms,
                                   [](SMState &sm) { return sm.IsActive(); });
  }

  /// \return
  ///     The maximum number of threads on this device supported by the HW.
  size_t GetMaxNumSupportedThreads() const;

  /// \return
  ///     A reference to the CUDA debugger API.
  CUDBGAPI &GetAPI() { return m_api; }

  /// \return
  ///     The number of threads per warp on this device.
  uint32_t GetNumThreadsPerWarp() const { return m_num_threads_per_warp; }

private:
  /// Decode device information from a buffer received from CUDA debugger API.
  ///
  /// \param[in] buffer
  ///     Buffer containing device information data.
  /// \param[in] size
  ///     Size of the buffer in bytes.
  ///
  /// \return
  ///     An error if the buffer cannot be decoded.
  llvm::Error DecodeDeviceInfoBuffer(uint8_t *buffer, size_t size);

  /// Reference to the CUDA debugger API structure.
  CUDBGAPI m_api;

  /// Information that is constant for the lifetime of the device.
  /// @{
  /// The unique identifier for this CUDA device.
  uint32_t m_device_id;

  /// The number of R registers available on this device (cached).
  std::optional<size_t> m_num_r_registers;

  std::optional<size_t> m_num_predicate_registers;

  std::optional<size_t> m_num_uniform_registers;

  std::optional<size_t> m_num_uniform_predicate_registers;

  /// The total number of streaming multiprocessors on this device.
  uint32_t m_num_sms = 0;

  /// The number of warps per streaming multiprocessor.
  uint32_t m_num_warps_per_sm = 0;

  /// The number of threads per warp.
  uint32_t m_num_threads_per_warp = 0;
  /// @}

  /// Information that changes upon every stop.
  /// @{
  /// Bitmask indicating which SMs are currently active.
  DynamicBitset m_sm_active_mask;

  /// Bitmask indicating which SMs have exceptions.
  DynamicBitset m_sm_exception_mask;

  /// Collection of SM states for all SMs on this device.
  std::vector<SMState> m_sms;

  /// Cached grid information indexed by grid ID.
  std::unordered_map<uint64_t, CUDBGGridInfo> m_grid_info;
  /// @}

  /// Fields that are used to decode the device information from the batch API.
  /// @{
  /// Size information for various device data structures.
  CUDBGDeviceInfoSizes m_device_info_sizes{};

  /// Temporary buffer for device information (disposed after each stop).
  /// Objects should not hold references to data in this buffer beyond a stop.
  std::vector<uint8_t> m_device_info_buffer;
  /// @}
};

/// Registry for managing all devices.
class DeviceStateRegistry {
public:
  /// Default constructor creates an empty registry.
  DeviceStateRegistry() = default;

  /// Create a registry with all available CUDA devices.
  ///
  /// \param[in] gpu
  ///     The GPU that this device state registry belongs to.
  ///
  /// \return
  ///     The registry, or an error if the debugger API cannot describe the
  ///     devices.
  static llvm::Expected<DeviceStateRegistry> Create(ProcessNVGPU &gpu);

  /// Update state information for all registered devices and associate the
  /// complete snapshot with \p stop_id.
  ///
  /// The snapshot is invalid while any device is being updated, and stays
  /// invalid if an update fails.
  ///
  /// \param[in] stop_id
  ///     The process stop ID that owns the completed snapshot.
  ///
  /// \return
  ///     An error if the state of a device cannot be read or decoded.
  llvm::Error BatchUpdate(uint32_t stop_id);

  /// Return whether the cached device snapshot belongs to \p stop_id.
  bool HasSnapshotForStopID(uint32_t stop_id) const {
    return m_snapshot_stop_id == stop_id;
  }

  /// Output registry state information to a stream for debugging.
  void Dump(Stream &s);

  /// Get a string representation of the registry state for debugging.
  ///
  /// \return
  ///     A string containing the registry state information.
  std::string Dump();

  /// Access a device state by index.
  ///
  /// \param[in] index
  ///     The index of the device to access.
  ///
  /// \return
  ///     A reference to the DeviceState at the specified index.
  DeviceState &operator[](size_t index) { return m_devices[index]; }

  /// \return
  ///     A reference to the collection of device states.
  llvm::MutableArrayRef<DeviceState> GetDevices() { return m_devices; }

  /// \return
  ///     The maximum number of threads on all devices supported by the HW.
  size_t GetMaxNumSupportedThreads() const;

private:
  /// Collection of device states for all CUDA devices in the system.
  std::vector<DeviceState> m_devices;

  /// The process stop that owns the complete cached device snapshot.
  std::optional<uint32_t> m_snapshot_stop_id;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVGPU_DEVICESTATE_H
