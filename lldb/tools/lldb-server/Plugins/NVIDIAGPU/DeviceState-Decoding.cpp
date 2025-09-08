//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the decoding algorithms of
// DeviceState.h. The purpose is to improve readability when inspecting
// this functionality by keeping these functions together.
//
//===----------------------------------------------------------------------===//

#include "../Utils/Utils.h"
#include "DeviceState.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"

using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

/// Calculate a linearized thread index for a thread given its thread index
/// and the block dimension.
static uint64_t CalculateFlatThreadIdx(const CuDim3 &base_thread_idx,
                                       const CuDim3 &block_dim) {
  return base_thread_idx.x + base_thread_idx.y * block_dim.x +
         base_thread_idx.z * block_dim.x * block_dim.y;
}

/// Calculate the thread index for a thread given its linearized thread index
/// and the block dimension.
static CuDim3 CalculateThreadIdx(uint64_t flat_thread_idx,
                                 const CuDim3 &block_dim) {
  CuDim3 thread_idx{0, 0, 0};
  if (block_dim.x == 0 || block_dim.y == 0 || block_dim.z == 0)
    return thread_idx;

  uint64_t current_id = flat_thread_idx;
  thread_idx.x = current_id % block_dim.x;
  current_id /= block_dim.x;

  thread_idx.y = current_id % block_dim.y;
  current_id /= block_dim.y;

  thread_idx.z = current_id % block_dim.z;

  return thread_idx;
}

size_t ThreadState::DecodeThreadInfoBuffer(
    uint8_t *buffer, const CUDBGDeviceInfo &device_info,
    const CUDBGDeviceInfoSizes &device_info_sizes,
    bool thread_attributes_present, const ExceptionInfo *warp_exception,
    CuDim3 thread_idx) {
  Log *log = GetLog(GDBRLog::Plugin);

  size_t buffer_offset = 0;
  const CUDBGLaneInfo &thread_info =
      *reinterpret_cast<const CUDBGLaneInfo *>(buffer);
  buffer_offset += device_info_sizes.laneInfoSize;

  if (thread_attributes_present) {
    uint32_t thread_attribute_flags = *(uint32_t *)(buffer + buffer_offset);
    buffer_offset += sizeof(uint32_t);

    for (uint32_t flags = thread_attribute_flags; flags;
         flags = flags & (flags - 1)) {
      int flag = __builtin_ctz(flags);
      switch (flag) {
      default:
        LLDB_LOGV(log, "Unknown thread/lane attribute flag: {}", flag);
      }
      buffer_offset += device_info_sizes.laneInfoAttributeSizes[flag];
    }
  }

  m_pc = thread_info.virtualPC;
  m_exception = warp_exception;
  m_thread_idx = thread_idx;

  return buffer_offset;
}

size_t WarpState::DecodeWarpInfoBuffer(
    uint8_t *buffer, const CUDBGDeviceInfo &device_info,
    const CUDBGDeviceInfoSizes &device_info_sizes,
    std::function<const CUDBGGridInfo &(uint64_t)> get_grid_info) {
  Log *log = GetLog(GDBRLog::Plugin);
  llvm::StringRef log_indent = "    ";

  size_t buffer_offset = 0;
  const CUDBGWarpInfo &warp_info =
      *reinterpret_cast<const CUDBGWarpInfo *>(buffer);
  buffer_offset += device_info_sizes.warpInfoSize;

  StaticBitset<uint32_t> thread_update_mask(
      std::numeric_limits<uint32_t>::max());
  bool thread_attributes_present = false;

  m_exception = std::nullopt;
  std::optional<uint64_t> errorPC;

  CUDBGException_t exception = CUDBG_EXCEPTION_NONE;

  for (uint32_t flags = warp_info.warpAttributeFlags; flags;
       flags = flags & (flags - 1)) {
    int flag = __builtin_ctz(flags);
    switch (flag) {
    case CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES: {
      thread_attributes_present = true;
      break;
    }
    case CUDBG_WARP_ATTRIBUTE_EXCEPTION: {
      exception = static_cast<CUDBGException_t>(
          *reinterpret_cast<const uint32_t *>(buffer + buffer_offset));
      break;
    }
    case CUDBG_WARP_ATTRIBUTE_ERRORPC: {
      errorPC = *reinterpret_cast<const uint64_t *>(buffer + buffer_offset);
      break;
    }
    case CUDBG_WARP_ATTRIBUTE_CLUSTERIDX: {
      // TODO: store the warp cluster index.
      break;
    }
    case CUDBG_WARP_ATTRIBUTE_LANE_UPDATE_MASK: {
      thread_update_mask.Update(
          *reinterpret_cast<const uint32_t *>(buffer + buffer_offset));
      break;
    }
    case CUDBG_WARP_ATTRIBUTE_CLUSTER_EXCEPTION_TARGET_BLOCK_IDX: {
      // TODO: store the warp cluster exception target block index.
      break;
    }
    default: {
      LLDB_LOGV(log, "{}Unknown warp attribute flag: {}", log_indent, flag);
    }
    }

    buffer_offset += device_info_sizes.warpInfoAttributeSizes[flag];
  }

  if (errorPC.has_value() && exception == CUDBG_EXCEPTION_NONE)
    logAndReportFatalError("WarpInformation::DecodeWarpInfoBuffer(). "
                           "errorPC set to {} without an exception",
                           errorPC);

  if (exception != CUDBG_EXCEPTION_NONE)
    m_exception = ExceptionInfo(exception, errorPC);

  const CUDBGGridInfo &grid_info = get_grid_info(warp_info.gridId);

  uint64_t flat_thread_idx =
      CalculateFlatThreadIdx(warp_info.baseThreadIdx, grid_info.blockDim);

  LLDB_LOG(log, "{}Updated threads mask: {:x}", log_indent,
           thread_update_mask.GetStorage());
  if (thread_update_mask.AreAllBitsSet())
    LLDB_LOG(log, "{}All threads will be updated", log_indent);

  StaticBitset<uint32_t> valid_lanes(warp_info.validLanes);

  for (uint32_t thread_id = 0; thread_id < m_threads.size(); ++thread_id) {
    bool is_valid = valid_lanes[thread_id];
    m_threads[thread_id].SetIsValid(is_valid);
    if (!is_valid)
      continue;

    bool is_updated = thread_update_mask[thread_id];

    if (log && !thread_update_mask.AreAllBitsSet())
      LLDB_LOG(log, "{}Thread {} is updated: {}", log_indent, thread_id,
               is_updated);

    if (!is_updated)
      continue;

    buffer_offset += m_threads[thread_id].DecodeThreadInfoBuffer(
        buffer + buffer_offset, device_info, device_info_sizes,
        thread_attributes_present, m_exception ? &m_exception.value() : nullptr,
        CalculateThreadIdx(flat_thread_idx + thread_id, grid_info.blockDim));
  }

  return buffer_offset;
}

size_t SMState::DecodeSMInfoBuffer(
    uint8_t *buffer, const CUDBGDeviceInfo &device_info,
    const CUDBGDeviceInfoSizes &device_info_sizes,
    std::function<const CUDBGGridInfo &(uint64_t)> get_grid_info) {
  Log *log = GetLog(GDBRLog::Plugin);
  llvm::StringRef log_indent = "  ";

  size_t buffer_offset = 0;
  const CUDBGSMInfo &sm_info = *reinterpret_cast<const CUDBGSMInfo *>(buffer);
  buffer_offset += device_info_sizes.smInfoSize;

  StaticBitset<uint64_t> updated_warps_mask(
      std::numeric_limits<uint64_t>::max());

  for (uint32_t flags = sm_info.smAttributeFlags; flags;
       flags = flags & (flags - 1)) {
    int flag = __builtin_ctz(flags);
    switch (flag) {
    case CUDBG_SM_ATTRIBUTE_WARP_UPDATE_MASK: {
      updated_warps_mask.Update(
          *reinterpret_cast<const uint64_t *>(buffer + buffer_offset));
      break;
    }
    default:
      LLDB_LOGV(log, "{}Unknown SM attribute flag: {}", log_indent, flag);
    }

    buffer_offset += device_info_sizes.smInfoAttributeSizes[flag];
  }

  LLDB_LOG(log, "{}Updated warps mask: {:x}", log_indent,
           updated_warps_mask.GetStorage());

  StaticBitset<uint64_t> valid_warps(sm_info.warpValidMask);

  for (uint32_t warp_id = 0; warp_id < m_warps.size(); ++warp_id) {
    bool is_valid = valid_warps[warp_id];
    m_warps[warp_id].SetIsValid(is_valid);
    if (!is_valid)
      continue;

    bool is_updated = updated_warps_mask[warp_id];

    LLDB_LOG(log, "{}Warp {} is updated: {}", log_indent, warp_id, is_updated);
    if (is_updated)
      buffer_offset += m_warps[warp_id].DecodeWarpInfoBuffer(
          buffer + buffer_offset, device_info, device_info_sizes,
          get_grid_info);
  }

  return buffer_offset;
}

void DeviceState::DecodeDeviceInfoBuffer(uint8_t *buffer, size_t size) {
  Log *log = GetLog(GDBRLog::Plugin);

  size_t buffer_offset = 0;
  const CUDBGDeviceInfo &device_info =
      *reinterpret_cast<const CUDBGDeviceInfo *>(buffer);
  buffer_offset += m_device_info_sizes.deviceInfoSize;

  std::optional<DynamicBitset> sm_update_mask;

  for (uint32_t flags = device_info.deviceAttributeFlags; flags;
       flags = flags & (flags - 1)) {
    int flag = __builtin_ctz(flags);
    size_t attribute_bytes_size =
        m_device_info_sizes.deviceInfoAttributeSizes[flag];
    llvm::StringRef bytes(
        reinterpret_cast<const char *>(buffer + buffer_offset),
        attribute_bytes_size);

    switch (flag) {
    case CUDBG_DEVICE_ATTRIBUTE_SM_ACTIVE_MASK: {
      m_sm_active_mask = DynamicBitset(bytes, m_num_sms);
      LLDB_LOG(log, "SM active mask: 0x{}", m_sm_active_mask.AsHex());
      break;
    }
    case CUDBG_DEVICE_ATTRIBUTE_SM_EXCEPTION_MASK: {
      m_sm_exception_mask = DynamicBitset(bytes, m_num_sms);
      LLDB_LOG(log, "SM exception mask: 0x{}", m_sm_exception_mask.AsHex());
      break;
    }
    case CUDBG_DEVICE_ATTRIBUTE_SM_UPDATE_MASK: {
      LLDB_LOG(log, "SM update mask: {}", bytes);
      sm_update_mask = DynamicBitset(bytes, m_num_sms);
      break;
    }
    default:
      LLDB_LOGV(log, "Unknown device attribute flag: {}", flag);
    }

    buffer_offset += attribute_bytes_size;
  }

  for (uint32_t sm_index = 0; sm_index < m_num_sms; ++sm_index) {
    bool is_active = m_sm_active_mask[sm_index];
    m_sms[sm_index].SetIsActive(is_active);
    if (!is_active)
      continue;

    bool is_updated = sm_update_mask.has_value()
                          ?  sm_update_mask->Get(sm_index)
                          : true;
    LLDB_LOG(log, "SM {} is updated: {}", sm_index, is_updated);
    if (is_updated)
      buffer_offset += m_sms[sm_index].DecodeSMInfoBuffer(
          buffer + buffer_offset, device_info, m_device_info_sizes,
          [this](uint64_t grid_id) -> const CUDBGGridInfo & {
            return this->GetGridInfo(grid_id);
          });
  }

  if (buffer_offset != size)
    logAndReportFatalError("DeviceInformation::DecodeDeviceInfoBuffer(). "
                           "Didn't parse the entire device info buffer. "
                           "Expected {} bytes but read {} bytes",
                           size, buffer_offset);
}

void DeviceState::BatchUpdate() {
  uint32_t data_length = 0;
  // TODO: handle CUDBG_RESPONSE_TYPE_UPDATE
  CUDBGResult res = m_api->getDeviceInfo(
      m_device_id, CUDBG_RESPONSE_TYPE_FULL, m_device_info_buffer.data(),
      m_device_info_sizes.requiredBufferSize, &data_length);
  if (res != CUDBG_SUCCESS)
    logAndReportFatalError("DeviceInformation::BatchUpdate(). "
                           "getDeviceInfo failed: {}",
                           cudbgGetErrorString(res));

  DecodeDeviceInfoBuffer(m_device_info_buffer.data(), data_length);
}
