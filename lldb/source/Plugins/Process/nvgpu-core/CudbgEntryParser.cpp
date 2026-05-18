//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CudbgEntryParser.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private::nvgpu_core {

llvm::Expected<DeviceEntry> DeviceEntry::Decode(const DataExtractor &data,
                                                uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated device table entry");
  DeviceEntry out{};
  offset_t off = 0;
  out.devName = data.GetAddress(&off);
  out.devType = data.GetAddress(&off);
  out.smType = data.GetAddress(&off);
  out.devId = data.GetU32(&off);
  out.pciBusId = data.GetU32(&off);
  out.pciDevId = data.GetU32(&off);
  out.numSMs = data.GetU32(&off);
  out.numWarpsPerSM = data.GetU32(&off);
  out.numLanesPerWarp = data.GetU32(&off);
  out.numRegsPerLane = data.GetU32(&off);
  out.numPredicatesPrLane = data.GetU32(&off);
  out.smMajor = data.GetU32(&off);
  out.smMinor = data.GetU32(&off);
  out.instructionSize = data.GetU32(&off);
  out.status = data.GetU32(&off);
  // Since CUDA driver r400.
  out.numUniformRegsPrWarp = data.GetU32(&off);
  out.numUniformPredicatesPrWarp = data.GetU32(&off);
  // Since CUDA driver r575.
  out.numConvergenceBarriersPrWarp = data.GetU32(&off);
  return out;
}

llvm::Expected<SMEntry> SMEntry::Decode(const DataExtractor &data,
                                        uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated SM table entry");
  SMEntry out{};
  offset_t off = 0;
  out.smId = data.GetU32(&off);
  out.padding0 = data.GetU32(&off);
  // Since CUDA driver r555.
  out.exception = data.GetU32(&off);
  out.errorPCValid = data.GetU32(&off);
  out.errorPC = data.GetAddress(&off);
  out.clusterExceptionTargetBlockIdxValid = data.GetU32(&off);
  out.clusterExceptionTargetBlockIdxX = data.GetU32(&off);
  out.clusterExceptionTargetBlockIdxY = data.GetU32(&off);
  out.clusterExceptionTargetBlockIdxZ = data.GetU32(&off);
  // Since CUDA driver r580.
  out.exceptionString = data.GetAddress(&off);
  return out;
}

llvm::Expected<CTAEntry> CTAEntry::Decode(const DataExtractor &data,
                                          uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated CTA table entry");
  CTAEntry out{};
  offset_t off = 0;
  out.gridId64 = data.GetAddress(&off);
  out.blockIdxX = data.GetU32(&off);
  out.blockIdxY = data.GetU32(&off);
  out.blockIdxZ = data.GetU32(&off);
  out.padding0 = data.GetU32(&off);
  // Since CUDA driver r525.
  out.clusterIdxX = data.GetU32(&off);
  out.clusterIdxY = data.GetU32(&off);
  out.clusterIdxZ = data.GetU32(&off);
  out.padding1 = data.GetU32(&off);
  // Since CUDA driver r565.
  out.clusterDimX = data.GetU32(&off);
  out.clusterDimY = data.GetU32(&off);
  out.clusterDimZ = data.GetU32(&off);
  return out;
}

llvm::Expected<WarpEntry> WarpEntry::Decode(const DataExtractor &data,
                                            uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated warp table entry");
  WarpEntry out{};
  offset_t off = 0;
  out.errorPC = data.GetAddress(&off);
  out.warpId = data.GetU32(&off);
  out.validLanesMask = data.GetU32(&off);
  out.activeLanesMask = data.GetU32(&off);
  out.isWarpBroken = data.GetU32(&off);
  out.errorPCValid = data.GetU32(&off);
  out.padding0 = data.GetU32(&off);
  // Since CUDA driver r525.
  out.numRegs = data.GetU32(&off);
  out.padding1 = data.GetU32(&off);
  // Since CUDA driver r570.
  out.sharedMemSize = data.GetU32(&off);
  out.padding2 = data.GetU32(&off);
  // Since CUDA driver r575.
  out.inSyscallLanesMask = data.GetU32(&off);
  out.cbuActiveLanesMask = data.GetU32(&off);
  out.cbuExitedLanesMask = data.GetU32(&off);
  out.cbuCollectiveLanesMask = data.GetU32(&off);
  // Since CUDA driver r590.
  out.barrierScope = data.GetU32(&off);
  out.padding3 = data.GetU32(&off);
  out.additionalBarrierInfo = data.GetAddress(&off);
  return out;
}

llvm::Expected<LaneEntry> LaneEntry::Decode(const DataExtractor &data,
                                            uint64_t entry_size) {
  if (data.GetByteSize() < entry_size)
    return llvm::createStringError("truncated lane table entry");
  LaneEntry out{};
  offset_t off = 0;
  out.virtualPC = data.GetAddress(&off);
  out.physPC = data.GetAddress(&off);
  out.ln = data.GetU32(&off);
  out.threadIdxX = data.GetU32(&off);
  out.threadIdxY = data.GetU32(&off);
  out.threadIdxZ = data.GetU32(&off);
  out.exception = data.GetU32(&off);
  out.callDepth = data.GetU32(&off);
  out.syscallCallDepth = data.GetU32(&off);
  out.ccRegister = data.GetU32(&off);
  // Since CUDA driver r575.
  out.cbuThreadState = data.GetU32(&off);
  out.padding0 = data.GetU32(&off);
  // Since CUDA driver r615.
  out.rpcLo = data.GetU32(&off);
  out.rpcHi = data.GetU32(&off);
  return out;
}

std::string FormatThreadName(const CTAEntry &cta, const LaneEntry &lane) {
  return nvgpu::FormatThreadName(cta.blockIdxX, cta.blockIdxY, cta.blockIdxZ,
                                 lane.threadIdxX, lane.threadIdxY,
                                 lane.threadIdxZ);
}

uint32_t ComputeAttributedException(const LaneEntry &lane, uint32_t lane_idx,
                                    lldb::SectionSP warp_section_sp,
                                    lldb::SectionSP sm_section_sp,
                                    ObjectFileELF *core) {
  if (lane.exception != 0)
    return lane.exception;

  Log *log = GetLog(LLDBLog::Process);
  llvm::Expected<WarpEntry> warp_or =
      ReadAndDecode<WarpEntry>(warp_section_sp, core);
  if (!warp_or) {
    LLDB_LOG_ERROR(log, warp_or.takeError(),
                   "Failed to decode GPU warp data while attributing "
                   "exception to lane {1}: {0}",
                   lane_idx);
    return 0;
  }
  if (!warp_or->errorPCValid || !warp_or->IsLaneActive(lane_idx))
    return 0;

  llvm::Expected<SMEntry> sm_or = ReadAndDecode<SMEntry>(sm_section_sp, core);
  if (!sm_or) {
    LLDB_LOG_ERROR(log, sm_or.takeError(),
                   "Failed to decode GPU SM data while attributing "
                   "exception to lane {1}: {0}",
                   lane_idx);
    return 0;
  }
  return sm_or->exception;
}

} // namespace lldb_private::nvgpu_core
