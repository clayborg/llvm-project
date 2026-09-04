//===-- EUThreadIntelGT.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_EUTHREADINTELGT_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_EUTHREADINTELGT_H

#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lldb_private {
namespace lldb_server {

class ProcessIntelGT;

/// A GPU thread's coordinate: kernel instance, workgroup indices, thread id.
struct ze_thread_coord {
  constexpr bool operator==(const ze_thread_coord &other) const {
    return (kernel_instance == other.kernel_instance && x == other.x &&
            y == other.y && z == other.z && id_in_wg == other.id_in_wg);
  }
  constexpr bool operator!=(const ze_thread_coord &other) const {
    return !(*this == other);
  }

  uint32_t kernel_instance;
  uint32_t x, y, z;           ///< Workgroup indices.
  uint16_t id_in_wg;          ///< Thread id in workgroup.
  uint16_t num_threads_in_wg; ///< Extra info for printing.
};

/// Wraps one hardware EU thread; owns the shared register cache and stop
/// reason. Multiple ThreadIntelGT lane objects reference the same instance.
class EUThreadIntelGT : public std::enable_shared_from_this<EUThreadIntelGT> {
public:
  /// Construct from a hardware thread identifier, device index, ZE debug
  /// session, SIMD width, and Level Zero device ID.
  EUThreadIntelGT(ze_device_thread_t ze_thread, uint32_t device_index,
                  zet_debug_session_handle_t session, uint32_t simd_width,
                  uint32_t device_id);

  /// Return the underlying Level Zero thread identifier.
  ze_device_thread_t GetZeThread() const { return m_ze_thread; }

  /// Return the index of the device this thread belongs to.
  uint32_t GetDeviceIndex() const { return m_device_index; }

  /// Return the ZE debug session handle.
  zet_debug_session_handle_t GetSession() const { return m_session; }

  /// Return the SIMD width (8, 16, or 32).
  uint32_t GetSimdWidth() const { return m_simd_width; }

  /// Update SIMD width and re-clamp the already-read active lanes mask.
  void SetSimdWidth(uint32_t w) {
    m_simd_width = w;
    if (w < 32)
      m_active_lanes &= (1u << w) - 1;
    if (m_active_lanes == 0)
      m_active_lanes = 0x1;
  }

  /// Return the bitmask of active SIMD lanes (CE & dispatch_mask).
  uint32_t GetActiveLanes() const { return m_active_lanes; }

  /// Set the active lanes bitmask directly.
  void SetActiveLanes(uint32_t mask) { m_active_lanes = mask; }

  // ----- Stop reason (shared by all lanes) ---------------------------------

  /// Fill \a stop_info and \a description with the stop reason details.
  bool GetStopReason(ThreadStopInfo &stop_info,
                     std::string &description) const {
    stop_info = m_stop_info;
    description = m_stop_description;
    return true;
  }

  /// Read CR0.1 from hardware, set the stop reason, and cache PC.
  void ReadStopReason();

  /// Return the cached PC (set by ReadStopReason).
  lldb::addr_t GetPC() const { return m_pc; }

  /// Return the cached ISA base address (identifies the module).
  lldb::addr_t GetIsaBase() const { return m_isa_base; }

  /// Set the stop reason from CR0.1 bits read after the hardware stop.
  void SetStopReasonFromCR0(uint32_t cr0_dword1);

  /// Clear the exception bit at \a bit_position in CR0.1.
  void ClearExceptionBit(uint32_t bit_position);

  /// Set an explicit stop reason.
  void SetStopReason(lldb::StopReason reason, uint32_t signo = 0);

  /// Set an explicit stop description string.
  void SetStopDescription(std::string desc) {
    m_stop_description = std::move(desc);
  }

  // ----- Single-stepping via CR0 --------------------------------------------

  /// Whether the next zetDebugResume should run or single-step.
  enum class ResumeState { Run, Step };

  ResumeState GetResumeState() const { return m_resume_state; }
  void SetResumeState(ResumeState state) { m_resume_state = state; }

  /// Set CR0.0 bit 15 so this EU can execute past the current breakpoint
  /// without re-triggering. Call before zetDebugResume() at every BP.
  void SuppressCurrentBreakpoint();

  /// Arm a single-step by setting the suppress bit and CR0.1 bit 31 so the
  /// next instruction triggers a breakpoint exception.
  void PrepareStep();

  /// Clear CR0.1 bit 31 (breakpoint_status) after a single-step completes.
  void ClearStepBits();

  // ----- SIMD lane management -----------------------------------------------

  /// Append one ThreadIntelGT per active SIMD lane; the lane at \a focus_tid
  /// (or the first active lane if invalid) receives the EU thread's stop
  /// reason. Lane TIDs are tid_base + absolute_lane_number.
  lldb::tid_t
  AddLaneThreads(ProcessIntelGT &process,
                 std::vector<std::unique_ptr<NativeThreadProtocol>> &threads,
                 lldb::tid_t tid_base, lldb::tid_t focus_tid);

  /// Read CE & SR0 dispatch mask to compute active SIMD lanes; falls back
  /// to all-lanes-active on register read failure.
  void ReadExecutionMask();

  // ----- Shared register data cache -----------------------------------------

  /// Allocate the shared register data cache from regset properties.
  void
  AllocateRegsetCache(const std::vector<zet_debug_regset_properties_t> &props);

  /// Return whether the cache has been allocated.
  bool IsRegsetCacheAllocated() const { return m_regsets_cache_allocated; }

  /// Read register set \a regset_index into the shared cache (no-op if
  /// already cached).
  Status ReadRegisterSet(uint32_t regset_index,
                         const zet_debug_regset_properties_t &props);

  /// Return whether register set \a regset_index has valid cached data.
  bool IsRegsetValid(uint32_t regset_index) const;

  /// Return the cached raw data for register set \a regset_index.
  const std::vector<uint8_t> &GetRegsetData(uint32_t regset_index) const;

  /// Invalidate all cached register data (called on resume).
  void InvalidateRegsetCache();

  /// Read thread coordinates; needs ProcessIntelGT for kernel-instance ID.
  bool GetCoordinates(ProcessIntelGT &process, ze_thread_coord &coords);

  // ----- Shared register set properties (discovered once per EU thread) ------

  /// Return true if register set properties have been discovered.
  bool IsRegsetPropsDiscovered() const { return m_regset_props_discovered; }

  /// Store discovered register set properties (called by the first lane).
  void SetRegsetProps(std::vector<zet_debug_regset_properties_t> props) {
    m_regset_props = std::move(props);
    m_regset_props_discovered = true;
  }

  /// Return the shared register set properties.
  const std::vector<zet_debug_regset_properties_t> &GetRegsetProps() const {
    return m_regset_props;
  }

private:
  ze_device_thread_t m_ze_thread;
  uint32_t m_device_index;
  zet_debug_session_handle_t m_session;
  uint32_t m_simd_width;
  uint32_t m_device_id;
  uint32_t m_active_lanes; ///< Bitmask of active SIMD lanes.

  ResumeState m_resume_state = ResumeState::Run;
  lldb::addr_t m_pc = LLDB_INVALID_ADDRESS;
  lldb::addr_t m_isa_base = LLDB_INVALID_ADDRESS;

  ThreadStopInfo m_stop_info;
  std::string m_stop_description;

  /// TID base for this EU thread's lanes (set by AddLaneThreads).
  lldb::tid_t m_tid_base = 0;

  /// Regset properties: discovered once per EU thread, shared across lanes.
  bool m_regset_props_discovered = false;
  std::vector<zet_debug_regset_properties_t> m_regset_props;

  /// Register data cache: read once per EU stop, shared across lanes.
  bool m_regsets_cache_allocated = false;
  std::vector<std::vector<uint8_t>> m_regset_data;
  std::vector<bool> m_regset_valid;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_EUTHREADINTELGT_H
