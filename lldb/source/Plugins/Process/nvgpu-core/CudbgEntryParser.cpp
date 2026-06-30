//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CudbgEntryParser.h"

#include "llvm/Support/Error.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private::nvgpu_core {

llvm::Expected<DeviceEntry> DeviceEntry::Decode(const DataExtractor &data,
                                                offset_t *offset_ptr,
                                                uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated device table entry");
  DeviceEntry out{};
  out.devName = data.GetU64(offset_ptr);
  out.devType = data.GetU64(offset_ptr);
  out.smType = data.GetU64(offset_ptr);
  out.devId = data.GetU32(offset_ptr);
  out.pciBusId = data.GetU32(offset_ptr);
  out.pciDevId = data.GetU32(offset_ptr);
  out.numSMs = data.GetU32(offset_ptr);
  out.numWarpsPerSM = data.GetU32(offset_ptr);
  out.numLanesPerWarp = data.GetU32(offset_ptr);
  out.numRegsPerLane = data.GetU32(offset_ptr);
  out.numPredicatesPrLane = data.GetU32(offset_ptr);
  out.smMajor = data.GetU32(offset_ptr);
  out.smMinor = data.GetU32(offset_ptr);
  out.instructionSize = data.GetU32(offset_ptr);
  out.status = data.GetU32(offset_ptr);
  // Since CUDA driver r400.
  out.numUniformRegsPrWarp = data.GetU32(offset_ptr);
  out.numUniformPredicatesPrWarp = data.GetU32(offset_ptr);
  // Since CUDA driver r575 (CUDBG API revision 156).
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(156)
  out.numConvergenceBarriersPrWarp = data.GetU32(offset_ptr);
#endif
  return out;
}

llvm::Expected<SMEntry> SMEntry::Decode(const DataExtractor &data,
                                        offset_t *offset_ptr,
                                        uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated SM table entry");
  SMEntry out{};
  out.smId = data.GetU32(offset_ptr);
  out.padding0 = data.GetU32(offset_ptr);
  // Since CUDA driver r555.
  out.exception = data.GetU32(offset_ptr);
  out.errorPCValid = data.GetU32(offset_ptr);
  out.errorPC = data.GetU64(offset_ptr);
  out.clusterExceptionTargetBlockIdxValid = data.GetU32(offset_ptr);
  out.clusterExceptionTargetBlockIdxX = data.GetU32(offset_ptr);
  out.clusterExceptionTargetBlockIdxY = data.GetU32(offset_ptr);
  out.clusterExceptionTargetBlockIdxZ = data.GetU32(offset_ptr);
  // Since CUDA driver r580 (CUDBG API revision 163).
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(163)
  out.exceptionString = data.GetU64(offset_ptr);
#endif
  return out;
}

llvm::Expected<GridEntry> GridEntry::Decode(const DataExtractor &data,
                                            offset_t *offset_ptr,
                                            uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated grid table entry");
  GridEntry out{};
  out.gridId64 = data.GetU64(offset_ptr);
  out.contextId = data.GetU64(offset_ptr);
  out.function = data.GetU64(offset_ptr);
  out.functionEntry = data.GetU64(offset_ptr);
  out.moduleHandle = data.GetU64(offset_ptr);
  // Deprecated since CUDA 13.4 (formerly parentGridId64); kept for ABI.
  out.reserved0 = data.GetU64(offset_ptr);
  out.paramsOffset = data.GetU64(offset_ptr);
  out.kernelType = data.GetU32(offset_ptr);
  out.origin = data.GetU32(offset_ptr);
  out.gridStatus = data.GetU32(offset_ptr);
  out.numRegs = data.GetU32(offset_ptr);
  out.gridDimX = data.GetU32(offset_ptr);
  out.gridDimY = data.GetU32(offset_ptr);
  out.gridDimZ = data.GetU32(offset_ptr);
  out.blockDimX = data.GetU32(offset_ptr);
  out.blockDimY = data.GetU32(offset_ptr);
  out.blockDimZ = data.GetU32(offset_ptr);
  out.attrLaunchBlocking = data.GetU32(offset_ptr);
  out.attrHostTid = data.GetU32(offset_ptr);
  // Since CUDA driver r525.
  out.clusterDimX = data.GetU32(offset_ptr);
  out.clusterDimY = data.GetU32(offset_ptr);
  out.clusterDimZ = data.GetU32(offset_ptr);
  out.padding0 = data.GetU32(offset_ptr);
  // Since CUDA driver r565.
  out.preferredClusterDimX = data.GetU32(offset_ptr);
  out.preferredClusterDimY = data.GetU32(offset_ptr);
  out.preferredClusterDimZ = data.GetU32(offset_ptr);
  out.padding1 = data.GetU32(offset_ptr);
  return out;
}

llvm::Expected<CTAEntry> CTAEntry::Decode(const DataExtractor &data,
                                          offset_t *offset_ptr,
                                          uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated CTA table entry");
  CTAEntry out{};
  out.gridId64 = data.GetU64(offset_ptr);
  out.blockIdxX = data.GetU32(offset_ptr);
  out.blockIdxY = data.GetU32(offset_ptr);
  out.blockIdxZ = data.GetU32(offset_ptr);
  out.padding0 = data.GetU32(offset_ptr);
  // Since CUDA driver r525.
  out.clusterIdxX = data.GetU32(offset_ptr);
  out.clusterIdxY = data.GetU32(offset_ptr);
  out.clusterIdxZ = data.GetU32(offset_ptr);
  out.padding1 = data.GetU32(offset_ptr);
  // Since CUDA driver r565.
  out.clusterDimX = data.GetU32(offset_ptr);
  out.clusterDimY = data.GetU32(offset_ptr);
  out.clusterDimZ = data.GetU32(offset_ptr);
  return out;
}

llvm::Expected<WarpEntry> WarpEntry::Decode(const DataExtractor &data,
                                            offset_t *offset_ptr,
                                            uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated warp table entry");
  WarpEntry out{};
  out.errorPC = data.GetU64(offset_ptr);
  out.warpId = data.GetU32(offset_ptr);
  out.validLanesMask = data.GetU32(offset_ptr);
  out.activeLanesMask = data.GetU32(offset_ptr);
  out.isWarpBroken = data.GetU32(offset_ptr);
  out.errorPCValid = data.GetU32(offset_ptr);
  out.padding0 = data.GetU32(offset_ptr);
  // Since CUDA driver r525.
  out.numRegs = data.GetU32(offset_ptr);
  out.padding1 = data.GetU32(offset_ptr);
  // Since CUDA driver r570.
  out.sharedMemSize = data.GetU32(offset_ptr);
  out.padding2 = data.GetU32(offset_ptr);
  // Since CUDA driver r575.
  out.inSyscallLanesMask = data.GetU32(offset_ptr);
  out.cbuActiveLanesMask = data.GetU32(offset_ptr);
  out.cbuExitedLanesMask = data.GetU32(offset_ptr);
  out.cbuCollectiveLanesMask = data.GetU32(offset_ptr);
  // Since CUDA driver r590 (CUDBG API revision 167).
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(167)
  out.barrierScope = data.GetU32(offset_ptr);
  out.padding3 = data.GetU32(offset_ptr);
  out.additionalBarrierInfo = data.GetU64(offset_ptr);
#endif
  return out;
}

llvm::Expected<LaneEntry> LaneEntry::Decode(const DataExtractor &data,
                                            offset_t *offset_ptr,
                                            uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated lane table entry");
  LaneEntry out{};
  out.virtualPC = data.GetU64(offset_ptr);
  out.physPC = data.GetU64(offset_ptr);
  out.ln = data.GetU32(offset_ptr);
  out.threadIdxX = data.GetU32(offset_ptr);
  out.threadIdxY = data.GetU32(offset_ptr);
  out.threadIdxZ = data.GetU32(offset_ptr);
  out.exception = data.GetU32(offset_ptr);
  out.callDepth = data.GetU32(offset_ptr);
  out.syscallCallDepth = data.GetU32(offset_ptr);
  out.ccRegister = data.GetU32(offset_ptr);
  // Since CUDA driver r575 (CUDBG API revision 156).
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(156)
  out.cbuThreadState = data.GetU32(offset_ptr);
  out.padding0 = data.GetU32(offset_ptr);
#endif
  // Since CUDA driver r615 (CUDBG API revision 192).
#if LLDB_NVGPU_CUDBG_API_REV_AT_LEAST(192)
  out.rpcLo = data.GetU32(offset_ptr);
  out.rpcHi = data.GetU32(offset_ptr);
#endif
  return out;
}

llvm::Expected<ConstBankEntry> ConstBankEntry::Decode(const DataExtractor &data,
                                                      offset_t *offset_ptr,
                                                      uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated constbank table entry");
  ConstBankEntry out{};
  out.addr = data.GetU64(offset_ptr);
  out.size = data.GetU32(offset_ptr);
  out.bankId = data.GetU32(offset_ptr);
  return out;
}

std::optional<ProducerInfo> DecodeProducerInfo(const DataExtractor &data) {
  if (data.GetByteSize() == 0)
    return std::nullopt; // No metadata section.

  // CudbgMetaDataEntry layout (since CUDA driver r565): uint64 generatorName,
  // uint32 driverVersion{Major,Minor}, uint32 cudaDriverVersion{Major,Minor},
  // then flags/timestamp we don't need. driverVersionMajor is the GPU driver
  // branch (e.g. 575/580/615); it is not set on Tegra (stays 0 = "unknown").
  ProducerInfo info;
  offset_t offset = 0;
  data.GetU64(&offset); // generatorName (string-table index; unused)
  info.driver_branch = data.GetU32(&offset);
  data.GetU32(&offset); // driverVersionMinor (unused)
  info.cuda_major = data.GetU32(&offset);
  info.cuda_minor = data.GetU32(&offset);
  return info;
}

llvm::Expected<BacktraceEntry>
BacktraceEntry::Decode(const DataExtractor &data, offset_t *offset_ptr,
                       uint64_t entry_size) {
  if (!data.ValidOffsetForDataOfSize(*offset_ptr, entry_size))
    return llvm::createStringError("truncated backtrace table entry");
  BacktraceEntry out{};
  out.returnAddress = data.GetU64(offset_ptr);
  out.virtualReturnAddress = data.GetU64(offset_ptr);
  out.level = data.GetU32(offset_ptr);
  out.pad = data.GetU32(offset_ptr);
  return out;
}

std::string FormatThreadName(const CTAEntry &cta, const LaneEntry &lane) {
  return nvgpu::FormatThreadName(cta.blockIdxX, cta.blockIdxY, cta.blockIdxZ,
                                 lane.threadIdxX, lane.threadIdxY,
                                 lane.threadIdxZ);
}

std::optional<StopAttribution>
ComputeStopAttribution(const LaneEntry &lane, uint32_t lane_idx,
                       lldb::SectionSP warp_section_sp,
                       lldb::SectionSP sm_section_sp, ObjectFile *core) {
  if (lane.exception != 0)
    return StopAttribution{lane.exception, false, ""};

  Log *log = GetLog(LLDBLog::Process);
  llvm::Expected<WarpEntry> warp_or =
      ReadAndDecode<WarpEntry>(warp_section_sp, core);
  if (!warp_or) {
    std::string err =
        llvm::formatv("failed to decode GPU warp data for lane {0}: {1}",
                      lane_idx, llvm::toString(warp_or.takeError()))
            .str();
    LLDB_LOG(log, "{0}", err);
    return StopAttribution{0, false, std::move(err)};
  }

  // The SDK has no `CUDBG_EXCEPTION_TRAP`; an inline trap surfaces as
  // `isWarpBroken`. Gate by active lane so unrelated lanes on the same
  // warp don't claim the trap.
  const bool at_trap = warp_or->isWarpBroken && warp_or->IsLaneActive(lane_idx);

  uint32_t attributed_exception = 0;
  std::string decode_error;
  if (warp_or->errorPCValid && warp_or->IsLaneActive(lane_idx)) {
    llvm::Expected<SMEntry> sm_or = ReadAndDecode<SMEntry>(sm_section_sp, core);
    if (!sm_or) {
      decode_error =
          llvm::formatv("failed to decode GPU SM data for lane {0}: {1}",
                        lane_idx, llvm::toString(sm_or.takeError()))
              .str();
      LLDB_LOG(log, "{0}", decode_error);
    } else {
      attributed_exception = sm_or->exception;
    }
  }

  if (attributed_exception == 0 && !at_trap && decode_error.empty())
    return std::nullopt;
  return StopAttribution{attributed_exception, at_trap,
                         std::move(decode_error)};
}

} // namespace lldb_private::nvgpu_core
