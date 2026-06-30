//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindNVGPUCore.h"
#include "CudbgEntryParser.h"
#include "ProcessNVGPUCore.h"
#include "RegisterContextNVGPUCore.h"
#include "SectionUtils.h"
#include "ThreadNVGPUCore.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Target/UnwindLLDB.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace {

/// Decode every per-lane backtrace frame. Each frame is its own self-
/// describing Section (file_size == the table's sh_entsize), fanned out by
/// `ObjectFileELF::BuildNVGPUSectionList`, so the per-row stride is simply
/// `Section::GetFileSize()` and no entry-size lookup is required.
llvm::Expected<llvm::SmallVector<nvgpu_core::BacktraceEntry, 8>>
DecodeBacktraceTable(const Section &lane_section, ObjectFile *core) {
  llvm::SmallVector<lldb::SectionSP, 8> frames = nvgpu_core::FindChildrenByType(
      lane_section, eSectionTypeNVGPUBacktraceEntry);

  llvm::SmallVector<nvgpu_core::BacktraceEntry, 8> entries;
  entries.reserve(frames.size());
  for (const lldb::SectionSP &frame_sp : frames) {
    const uint64_t row_size = frame_sp->GetFileSize();
    DataExtractor data;
    core->ReadSectionData(frame_sp.get(), data);
    lldb::offset_t off = 0;
    llvm::Expected<nvgpu_core::BacktraceEntry> entry_or =
        nvgpu_core::BacktraceEntry::Decode(data, &off, row_size);
    if (!entry_or)
      return entry_or.takeError();
    entries.push_back(*entry_or);
  }
  return entries;
}

/// Build frame PCs for a lane from its backtrace table, sorted by `level`.
/// Frame 0 always uses `lane.virtualPC`. Each backtrace row's `level` names
/// the caller frame at that depth (level 0 is frame 1, level 1 is frame 2,
/// and so on); its `virtualReturnAddress` is that frame's PC.
///
/// Returns std::nullopt when the backtrace table cannot be used for this
/// lane, so the caller falls back to DWARF-CFI. This covers both the
/// wrong-unwinder cases (the lane has real local memory, or it has no
/// backtrace rows) and decode failures, which are logged here before being
/// consumed.
std::optional<llvm::SmallVector<lldb::addr_t, 8>>
BuildBacktracePCs(lldb::SectionSP lane_section_sp, ObjectFile *core) {
  if (!lane_section_sp || !core)
    return std::nullopt;

  // Lanes with real local memory unwind via DWARF CFI; the backtrace table is
  // only the substitute for lanes the producer omitted local memory for.
  if (nvgpu_core::FindChildByType(*lane_section_sp,
                                  eSectionTypeNVGPULocalMemory))
    return std::nullopt;

  llvm::Expected<llvm::SmallVector<nvgpu_core::BacktraceEntry, 8>> entries_or =
      DecodeBacktraceTable(*lane_section_sp, core);
  if (!entries_or) {
    LLDB_LOG(GetLog(LLDBLog::Unwind),
             "NVGPU backtrace table unusable, falling back to DWARF-CFI: {0}",
             llvm::toString(entries_or.takeError()));
    return std::nullopt;
  }
  if (entries_or->empty())
    return std::nullopt;

  llvm::Expected<nvgpu_core::LaneEntry> lane_or =
      nvgpu_core::ReadAndDecode<nvgpu_core::LaneEntry>(lane_section_sp, core);
  if (!lane_or) {
    LLDB_LOG(GetLog(LLDBLog::Unwind),
             "NVGPU lane decode failed, falling back to DWARF-CFI: {0}",
             llvm::toString(lane_or.takeError()));
    return std::nullopt;
  }

  llvm::SmallVector<nvgpu_core::BacktraceEntry, 8> sorted = *entries_or;
  llvm::sort(sorted,
             [](const auto &a, const auto &b) { return a.level < b.level; });

  llvm::SmallVector<lldb::addr_t, 8> pcs;
  pcs.reserve(sorted.size() + 1);

  pcs.push_back(lane_or->virtualPC);

  std::optional<uint32_t> prev_level;
  for (const nvgpu_core::BacktraceEntry &entry : sorted) {
    if (prev_level && entry.level == *prev_level)
      continue;
    pcs.push_back(entry.virtualReturnAddress);
    prev_level = entry.level;
  }

  return pcs;
}

} // namespace

UnwindNVGPUCore::UnwindNVGPUCore(Thread &thread) : Unwind(thread) {}

void UnwindNVGPUCore::DoClear() {
  m_table_pcs.clear();
  m_use_backtrace_table = false;
  m_initialized = false;
  if (m_dwarf_unwinder_up)
    m_dwarf_unwinder_up->Clear();
}

void UnwindNVGPUCore::EnsureInitialized() {
  if (m_initialized)
    return;

  auto &gpu_thread = static_cast<ThreadNVGPUCore &>(m_thread);
  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return;

  auto *nvgpu_process = static_cast<ProcessNVGPUCore *>(process_sp.get());
  ObjectFile *core = nvgpu_process->GetCoreObjectFile();
  SectionSP lane_sp = gpu_thread.GetLaneSection();
  if (!core || !lane_sp)
    return;

  std::optional<llvm::SmallVector<addr_t, 8>> pcs =
      BuildBacktracePCs(lane_sp, core);
  if (pcs) {
    m_table_pcs = std::move(*pcs);
    m_use_backtrace_table = true;
  } else {
    // No usable backtrace table for this lane (absent, wrong unwinder, or a
    // decode failure already logged in BuildBacktracePCs): use DWARF-CFI.
    m_dwarf_unwinder_up = std::make_unique<UnwindLLDB>(m_thread);
  }

  m_initialized = true;
}

uint32_t UnwindNVGPUCore::DoGetFrameCount() {
  EnsureInitialized();
  if (m_use_backtrace_table)
    return m_table_pcs.size();
  if (!m_dwarf_unwinder_up)
    return 0;
  return m_dwarf_unwinder_up->GetFrameCount();
}

bool UnwindNVGPUCore::DoGetFrameInfoAtIndex(uint32_t frame_idx, addr_t &cfa,
                                            addr_t &pc,
                                            bool &behaves_like_zeroth_frame) {
  EnsureInitialized();
  if (m_use_backtrace_table) {
    if (frame_idx >= m_table_pcs.size())
      return false;
    // PC-only synthetic unwind: there is no real stack, so no genuine CFA is
    // available. Use the frame index as a unique synthetic CFA (as
    // HistoryUnwind does) so StackID can distinguish recursive or
    // repeated-symbol frames.
    cfa = frame_idx;
    pc = m_table_pcs[frame_idx];
    behaves_like_zeroth_frame = (frame_idx == 0);
    return true;
  }
  if (!m_dwarf_unwinder_up)
    return false;
  return m_dwarf_unwinder_up->GetFrameInfoAtIndex(frame_idx, cfa, pc,
                                                  behaves_like_zeroth_frame);
}

lldb::RegisterContextSP
UnwindNVGPUCore::DoCreateRegisterContextForFrame(StackFrame *frame) {
  EnsureInitialized();

  const uint32_t idx = frame ? frame->GetConcreteFrameIndex() : 0;
  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return {};

  auto *nvgpu_process = static_cast<ProcessNVGPUCore *>(process_sp.get());
  ObjectFile *core = nvgpu_process->GetCoreObjectFile();

  if (idx == 0)
    return std::make_shared<RegisterContextNVGPUCore>(m_thread, core);

  if (m_use_backtrace_table) {
    if (idx >= m_table_pcs.size())
      return {};
    return std::make_shared<RegisterContextNVGPUCore>(m_thread, core, idx,
                                                      m_table_pcs[idx]);
  }

  if (!m_dwarf_unwinder_up)
    return {};
  return m_dwarf_unwinder_up->CreateRegisterContextForFrame(frame);
}
