//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICESTATE_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICESTATE_H

#include "../Utils/Bitset.h"
#include "ThreadNVIDIAGPU.h"
#include "cudadebugger.h"
#include "lldb/Utility/Stream.h"
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lldb_private::lldb_server {

/// This struct represents the coordinates of a warp in a GPU.
struct WarpCoords {
  int64_t dev_id;
  int64_t sm_id;
  int64_t warp_id;

  WarpCoords(int64_t dev_id, int64_t sm_id, int64_t warp_id)
      : dev_id(dev_id), sm_id(sm_id), warp_id(warp_id) {}
};

/// This struct represents the physical coordinates of a HW thread in a GPU.
struct PhysicalCoords {
  int64_t dev_id;
  int64_t sm_id;
  int64_t warp_id;
  int64_t thread_id;

  PhysicalCoords(int64_t dev_id, int64_t sm_id, int64_t warp_id,
                 int64_t thread_id)
      : dev_id(dev_id), sm_id(sm_id), warp_id(warp_id), thread_id(thread_id) {}

  /// \return
  ///     A corresponding warp coordinates structure.
  WarpCoords GetWarpCoords() const {
    return WarpCoords(dev_id, sm_id, warp_id);
  }

  /// \return
  //    A string representation of the coordinates used for logging.
  std::string Dump() const;
};

/// Represents information about an exception that occurred during CUDA kernel execution.
struct ExceptionInfo {
  /// The type of CUDA exception that occurred. It's guaranteed not to be CUDBG_EXCEPTION_NONE.
  CUDBGException_t exception;

  /// The program counter address where the exception occurred, if available.
  std::optional<uint64_t> errorPC;

  /// Construct exception information.
  ///
  /// \param[in] exception
  ///     The CUDA exception type that occurred.
  /// \param[in] errorPC
  ///     Optional program counter where the exception occurred.
  ExceptionInfo(CUDBGException_t exception, std::optional<uint64_t> errorPC);

  /// Convert the exception information to a human-readable string.
  ///
  /// \return
  ///     A string representation of the exception information.
  std::string ToString() const;
};

/// Represents the state of a single CUDA thread.
class ThreadState {
public:
  /// Construct a thread state with physical coordinates.
  ///
  /// \param[in] gpu
  ///     The GPU that this thread state belongs to.
  /// \param[in] physical_coords
  ///     This physical coordinates won't change for the lifetime of this
  ///     object.
  ThreadState(NVIDIAGPU &gpu, const PhysicalCoords &physical_coords);

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
                                CuDim3 thread_idx, bool at_breakpoint,
                                bool is_active);

  /// \return
  ///     True if this thread is valid in the GPU, false otherwise.
  bool IsValid() const { return m_is_valid; }

  /// Set the validity status of this object.
  void SetIsValid(bool is_valid) { m_is_valid = is_valid; }

  /// Output thread state information to a stream for debugging.
  void Dump(Stream &s);

  /// \return
  ///     A reference to the physical coordinates structure.
  const PhysicalCoords &GetPhysicalCoords() const { return m_physical_coords; }

  /// \return
  ///     A reference to the 3D thread index.
  const CuDim3 &GetThreadIdx() const { return m_thread_idx; }

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
  ///     The ThreadNVIDIAGPU object associated with this thread.
  ThreadNVIDIAGPU &GetThreadNVIDIAGPU() { return m_thread_nvidiagpu; }

private:
  /// Whether this thread is valid in the GPU.
  bool m_is_valid = false;

  /// The program counter value for this thread.
  uint64_t m_pc = 0;

  /// Exception information if the parent warp encountered an exception.
  const ExceptionInfo *m_exception = nullptr;

  /// The 3D thread index within the thread block.
  CuDim3 m_thread_idx;

  /// The physical coordinates of this thread on the device.
  PhysicalCoords m_physical_coords;

  /// The ThreadNVIDIAGPU object associated with this thread. This object is the
  /// interface LLGS wants to interact with.
  ThreadNVIDIAGPU m_thread_nvidiagpu;
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
  WarpState(NVIDIAGPU &gpu, uint32_t num_threads, uint32_t device_id,
            uint32_t sm_id, uint32_t warp_id);

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
  ///     The number of bytes consumed from the buffer.
  size_t DecodeWarpInfoBuffer(
      uint8_t *buffer, const CUDBGDeviceInfo &device_info,
      const CUDBGDeviceInfoSizes &device_info_sizes,
      std::function<const CUDBGGridInfo &(uint64_t)> get_grid_info,
      std::function<void(llvm::StringRef message)> log_to_client_callback,
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

  /// Find a thread within this warp that has an exception.
  ///
  /// \return
  ///     Pointer to a thread state with an exception, or nullptr if none found.
  const ThreadState *FindSomeThreadWithException() const;

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
  /// \param[in] gpu
  ///     The GPU that this warp state belongs to.
  ///
  /// \return
  ///     The number of regular registers currently used by this warp.
  size_t GetCurrentNumRegularRegisters();

  /// Get the registers for this warp. The result is cached until the warp state
  /// changes.
  ///
  /// \return
  ///     A reference to the registers for this warp.
  const WarpRegistersWithValidity &GetRegisters();

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

  /// The registers for this warp.
  WarpRegistersWithValidity m_regs;

  /// Whether the registers for this warp have been calculated.
  bool m_regs_calculated = false;
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
  SMState(NVIDIAGPU &gpu, uint32_t num_warps, uint32_t num_threads_per_warp,
          uint32_t device_id, uint32_t sm_id);

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
  /// \param[in] log_to_client_callback
  ///     Function to log messages to the client.
  ///
  /// \return
  ///     The number of bytes consumed from the buffer.
  size_t DecodeSMInfoBuffer(
      uint8_t *buffer, const CUDBGDeviceInfo &device_info,
      const CUDBGDeviceInfoSizes &device_info_sizes,
      std::function<const CUDBGGridInfo &(uint64_t)> get_grid_info,
      std::function<void(llvm::StringRef message)> log_to_client_callback);

  /// Set the activity status of this SM.
  void SetIsActive(bool is_active);

  /// \return
  ///     True if the SM is active in the GPU, false otherwise.
  bool IsActive() const { return m_is_active; }

  /// Output SM state information to a stream for debugging.
  void Dump(Stream &s);

  /// Find a thread within this SM that has an exception.
  ///
  /// \return
  ///     Pointer to a thread state with an exception, or nullptr if none found.
  const ThreadState *FindSomeThreadWithException() const;

  /// \return
  ///     A reference to the collection of warps.
  llvm::MutableArrayRef<WarpState> GetWarps() { return m_warps; }

  /// \return
  ///     An iterator to the collection of valid warps.
  auto GetValidWarps() {
    return llvm::make_filter_range(
        m_warps, [](WarpState &warp) { return warp.IsValid(); });
  }

private:
  /// Whether this SM is currently active in the GPU.
  bool m_is_active;

  /// All warps in this SM.
  std::vector<WarpState> m_warps;
};

/// Manages and caches information about a particular CUDA device.
class DeviceState {
public:
  /// Construct a device state manager for the specified device.
  ///
  /// \param[in] gpu
  ///     The GPU that this device state belongs to.
  /// \param[in] device_id
  ///     The ID of the CUDA device to manage.
  DeviceState(NVIDIAGPU &gpu, uint32_t device_id);

  /// \return
  ///     The ID of the CUDA device.
  uint32_t GetDeviceId() const { return m_device_id; }

  /// Get the number of R registers available on this device. This is cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of R registers for the device.
  size_t GetMaxNumSupportedRegularRegister();

  /// Get the number of predicate registers available on this device. This is
  /// cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of predicate registers for the device.
  size_t GetNumPredicateRegisters();

  /// Get the number of uniform registers available on this device. This is
  /// cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of uniform registers for the device.
  size_t GetNumUniformRegisters();

  /// Get the number of uniform predicate registers available on this device.
  /// This is cached for the lifetime of the device.
  ///
  /// \return
  ///     The number of uniform predicate registers for the device.
  size_t GetNumUniformPredicateRegisters();

  /// Update all device state information from the CUDA debugger API.
  ///
  /// This method performs a batch update of all dynamic state information
  /// including SM states, warp states, and thread states.
  void BatchUpdate(
      std::function<void(llvm::StringRef message)> log_to_client_callback);

  /// Output device state information to a stream for debugging.
  void Dump(Stream &s);

  /// Get grid information for the specified grid ID.
  ///
  /// \param[in] grid_id
  ///     The ID of the grid to retrieve information for.
  ///
  /// \return
  ///     A reference to the grid information structure.
  const CUDBGGridInfo &GetGridInfo(uint64_t grid_id);

  /// Find a thread within this device that has an exception.
  ///
  /// \return
  ///     Pointer to a thread state with an exception, or nullptr if none found.
  const ThreadState *FindSomeThreadWithException() const;

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

private:
  /// Decode device information from a buffer received from CUDA debugger API.
  ///
  /// \param[in] buffer
  ///     Buffer containing device information data.
  /// \param[in] size
  ///     Size of the buffer in bytes.
  /// \param[in] log_to_client_callback
  ///     Function to log messages to the client.
  void DecodeDeviceInfoBuffer(
      uint8_t *buffer, size_t size,
      std::function<void(llvm::StringRef message)> log_to_client_callback);

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
  uint32_t m_num_sms;

  /// The number of warps per streaming multiprocessor.
  uint32_t m_num_warps_per_sm;

  /// The number of threads per warp.
  uint32_t m_num_threads_per_warp;
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
  CUDBGDeviceInfoSizes m_device_info_sizes;

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

  /// Construct a registry and initialize it with all available CUDA devices.
  ///
  /// \param[in] gpu
  ///     The GPU that this device state registry belongs to.
  DeviceStateRegistry(NVIDIAGPU &gpu);

  /// Update state information for all registered devices.
  void BatchUpdate(
      std::function<void(llvm::StringRef message)> log_to_client_callback);

  /// Output registry state information to a stream for debugging.
  void Dump(Stream &s);

  /// Get a string representation of the registry state for debugging.
  ///
  /// \return
  ///     A string containing the registry state information.
  std::string Dump();

  /// Find a thread across all devices that has an exception.
  ///
  /// \return
  ///     Pointer to a thread state with an exception, or nullptr if none found.
  const ThreadState *FindSomeThreadWithException() const;

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
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_NVIDIAGPU_DEVICESTATE_H
