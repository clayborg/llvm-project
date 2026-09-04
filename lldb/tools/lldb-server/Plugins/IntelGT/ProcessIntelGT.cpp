//===-- ProcessIntelGT.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessIntelGT.h"
#include "AddressSpaces.h"
#include "IntelGTArch.h"
#include "LLDBServerPluginIntelGT.h"
#include "LevelZeroHelpers.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "RegisterContextIntelGT.h"
#include "ThreadIntelGT.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Utility/AddressSpace.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <cinttypes>
#include <cstring>
#include <limits>
#include <signal.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

// ---------------------------------------------------------------------------
// ProcessIntelGT
// ---------------------------------------------------------------------------

ProcessIntelGT::ProcessIntelGT(lldb::pid_t pid, NativeDelegate &delegate,
                               LLDBServerPluginIntelGT *plugin,
                               std::vector<DeviceSession> device_sessions)
    : NativeProcessProtocol(pid, -1, delegate), m_plugin(plugin),
      m_device_sessions(std::move(device_sessions)) {
  // Start in stopped state; CreateGpuProcess will transition to running.
  m_state = eStateStopped;
}

// ---------------------------------------------------------------------------
// Resume
//
// x86-like all-stop resume: arm CR0 for the stepping thread, resume all
// stopped EU threads, ACK pending MODULE_LOAD events, transition to running.
// ---------------------------------------------------------------------------

Status ProcessIntelGT::Resume(const ResumeActionList &resume_actions) {

  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);

  // Check ALL stopped EU threads for stepping actions since LLDB may send
  // vCont;s:TID where TID != current thread.
  for (auto &[ze_thread, eu_thread] : m_stopped_eu_threads) {
    const ResumeAction *action =
        FindResumeActionForEUThread(resume_actions, eu_thread.get());
    if (action && action->state == eStateStepping) {
      eu_thread->PrepareStep();
      // Remember the lane TID the client is stepping so the step-complete stop
      // is attributed to that lane (see HandleZeThreadStopped).
      m_stepping_eu_threads[ze_thread] = action->tid;
    }
  }

  EUThreadIntelGT *cur_eu = GetCurrentEUThread();
  if (cur_eu) {
    [[maybe_unused]] ze_device_thread_t cur_ze = cur_eu->GetZeThread();
  } else {
  }

  // GPU debugging semantics: multiple EU threads hitting the same breakpoint
  // continue together when the user types 'c'.
  bool stepping = !m_stepping_eu_threads.empty();

  // '$c' becomes a default continue action (tid=0, state=Running) meaning
  // "continue all threads"; 'vCont;s:TID' becomes an explicit stepping action.
  size_t num_running = resume_actions.NumActionsWithState(eStateRunning);
  size_t num_stepping_action =
      resume_actions.NumActionsWithState(eStateStepping);

  const ResumeAction *first_action = resume_actions.GetFirst();
  bool has_continue_action = (num_running > 0 || num_stepping_action > 0);
  if (first_action && resume_actions.GetSize() > 0) {
  }

  // Collect ze_threads that we actually resume so we can clean up
  // their lane threads afterwards.
  std::vector<ze_device_thread_t> resumed_ze_threads;

  // When continuing all threads (not stepping), clear the stepping map so
  // threads are not misidentified as "trace stops" on the next breakpoint.
  if (!stepping && has_continue_action) {
    m_stepping_eu_threads.clear();

    // Also clear lane stop reasons/states so IsAnyThreadSteppingCompleted()
    // does not see stale eStopReasonTrace, and LLDB does not see stale
    // stopped lanes that should be running.
    for (auto &t : m_threads) {
      auto *thread = static_cast<ThreadIntelGT *>(t.get());
      if (!thread->IsShadowThread()) {
        thread->SetStopReason(eStopReasonNone, 0);
        thread->SetState(eStateRunning);
      }
    }
  }

  // Check for a specific thread action (tid != LLDB_INVALID_THREAD_ID) rather
  // than the default "all threads" action.
  bool has_any_specific_action = false;
  for (size_t i = 0; i < resume_actions.GetSize(); i++) {
    const ResumeAction *action = &resume_actions.GetFirst()[i];
    if (action->tid != LLDB_INVALID_THREAD_ID) {
      has_any_specific_action = true;
      break;
    }
  }

  for (auto &[ze_thread, eu_thread] : m_stopped_eu_threads) {
    bool is_stepping_thread =
        m_stepping_eu_threads.find(ze_thread) != m_stepping_eu_threads.end();

    // Check if this specific EU has an action.
    const ResumeAction *eu_action =
        FindResumeActionForEUThread(resume_actions, eu_thread.get());

    // When stepping, only resume the stepping EU thread.
    if (stepping && !is_stepping_thread) {
      continue;
    }

    // If a specific thread action was sent, only resume threads that have a
    // matching action so "continue thread 18" doesn't continue all threads.
    if (!stepping && has_any_specific_action && !eu_action) {
      continue;
    }

    // Without a specific action, resume ALL stopped EU threads because all EU
    // threads hit the same breakpoint and should continue together.
    if (!stepping && !has_any_specific_action && !has_continue_action) {
      continue;
    }

    const DeviceSession *ds = GetDeviceSession(eu_thread->GetDeviceIndex());
    if (!ds)
      continue;

    // Suppress the breakpoint on the current instruction so the EU thread can
    // execute past it; PrepareStep already sets this bit for stepping threads.
    if (!is_stepping_thread)
      eu_thread->SuppressCurrentBreakpoint();

    ze_result_t result = zetDebugResume(ds->session, ze_thread);
    if (result != ZE_RESULT_SUCCESS) {
    }

    eu_thread->InvalidateRegsetCache();
    resumed_ze_threads.push_back(ze_thread);
  }

  // A continue resumes the WHOLE device: reset nresumed to nthreads and clear
  // interrupt bookkeeping so the next-stop gather barrier knows there are
  // running threads to wait for. Stepping resumes only one EU thread so is
  // intentionally excluded here.
  if (!stepping && has_continue_action) {
    for (DeviceSession &ds_mut : m_device_sessions) {
      ds_mut.nresumed = ds_mut.nthreads;
      ds_mut.ninterrupts = 0;
    }
  }

  // NOTE: Keep m_threads and m_stopped_eu_threads intact after resume; LLDB
  // may still read from cached thread IDs. Clearing here yields E33/E15
  // errors. ClearOldEUThreads() handles cleanup on the next stop event.

  // ACK any pending MODULE_LOAD events now that LLDB has set breakpoints.
  if (!m_pending_module_acks.empty()) {
    for (auto &pending : m_pending_module_acks)
      ZeAckEvent(pending.session, pending.event);
    m_pending_module_acks.clear();
  } else {
  }

  // If EU threads resumed, tell the plugin to wait for them to stop before
  // reporting the next breakpoint so all threads reach it together.
  if (!resumed_ze_threads.empty() && m_plugin) {
    m_plugin->SetExpectedStoppedThreadCount(resumed_ze_threads.size());
  }

  if (m_plugin)
    m_plugin->TriggerNotifier();

  SetState(StateType::eStateRunning, true);
  return Status();
}

// ---------------------------------------------------------------------------
// Halt / Interrupt
// ---------------------------------------------------------------------------

Status ProcessIntelGT::Halt() {
  StateType prev = GetState();
  if (m_stopped_eu_threads.empty() && m_threads.size() == 1) {
  }
  if (prev == StateType::eStateStopped) {
    return Status();
  }

  // If current thread is the shadow thread, switch to a real GPU thread
  // before sending the stop packet. Shadow threads return E15 for register
  // reads, causing the client to auto-continue instead of stopping.
  {
    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    NativeThreadProtocol *current = GetCurrentThread();
    if (current) {
      ThreadIntelGT *current_intel = static_cast<ThreadIntelGT *>(current);
      if (current_intel->IsShadowThread()) {
        // Find the first non-shadow thread to use as current
        for (const auto &t : m_threads) {
          ThreadIntelGT *intel = static_cast<ThreadIntelGT *>(t.get());
          if (!intel->IsShadowThread()) {
            SetCurrentThreadID(intel->GetID());
            break;
          }
        }
      }
    }
  }

  SetState(StateType::eStateStopped, true);
  return Status();
}

Status ProcessIntelGT::Interrupt() {

  ze_device_thread_t wildcard = ZeWildcardThread();
  for (const DeviceSession &ds : m_device_sessions) {
    ze_result_t result = zetDebugInterrupt(ds.session, wildcard);
    if (result != ZE_RESULT_SUCCESS) {
    }
  }
  return Status();
}

void ProcessIntelGT::PauseAll() {
  // Interrupt every other running thread so the whole workgroup is presented
  // together; already-stopped threads are ignored by the driver. Send once
  // per device (tracked via ninterrupts); the caller drains events until
  // every interrupted thread reports its stop.
  ze_device_thread_t wildcard = ZeWildcardThread();
  for (DeviceSession &ds : m_device_sessions) {
    if (ds.ninterrupts != 0)
      continue; // interrupt-all already outstanding for this device
    ze_result_t result = zetDebugInterrupt(ds.session, wildcard);
    if (result == ZE_RESULT_SUCCESS) {
      ds.ninterrupts++;
    } else if (result == ZE_RESULT_NOT_READY) {
      // Already requested; treat as outstanding so we don't spam interrupts.
      ds.ninterrupts++;
    }
    // ERROR_NOT_AVAILABLE: all threads already stopped/finished; nothing to do.
  }
}

size_t ProcessIntelGT::GetResumedThreadCount() const {
  size_t total = 0;
  for (const DeviceSession &ds : m_device_sessions)
    total += ds.nresumed;
  return total;
}

Status ProcessIntelGT::Detach() {
  SetState(StateType::eStateDetached, true);
  return Status();
}

Status ProcessIntelGT::Signal(int /*signo*/) {
  return Status::FromErrorString("GPU processes do not support signals");
}

Status ProcessIntelGT::Kill() { return Status(); }

// ---------------------------------------------------------------------------
// Memory access
// ---------------------------------------------------------------------------

Status ProcessIntelGT::ReadMemory(addr_t addr, void *buf, size_t size,
                                  size_t &bytes_read) {

  const DeviceSession *ds = GetCurrentDeviceSession();
  if (!ds)
    return Status::FromErrorString("No active device session");

  // Use the current stopped thread when available; the driver needs a
  // specific thread to resolve per-EU-thread scratch VAs and ignores the
  // thread for code/global reads.
  ze_device_thread_t ze_thread = ZeWildcardThread();
  NativeThreadProtocol *current = GetCurrentThread();
  if (current) {
    ThreadIntelGT *t = static_cast<ThreadIntelGT *>(current);
    if (t->GetEUThread())
      ze_thread = t->GetZeThread();
  }

  zet_debug_memory_space_desc_t space_desc{};
  space_desc.stype = ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC;
  space_desc.type = ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT;
  space_desc.address = addr;

  ze_result_t result =
      zetDebugReadMemory(ds->session, ze_thread, &space_desc, size, buf);
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat("zetDebugReadMemory failed: %s",
                                             ZeResultToString(result).data());
  }

  bytes_read = size;
  return Status();
}

Status ProcessIntelGT::WriteMemory(addr_t addr, const void *buf, size_t size,
                                   size_t &bytes_written) {

  const DeviceSession *ds = GetCurrentDeviceSession();
  if (!ds)
    return Status::FromErrorString("No active device session");

  ze_device_thread_t ze_thread = ZeWildcardThread();

  zet_debug_memory_space_desc_t space_desc{};
  space_desc.stype = ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC;
  space_desc.type = ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT;
  space_desc.address = addr;

  ze_result_t result =
      zetDebugWriteMemory(ds->session, ze_thread, &space_desc, size, buf);
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat("zetDebugWriteMemory failed: %s",
                                             ZeResultToString(result).data());
  }

  bytes_written = size;
  return Status();
}

std::vector<AddressSpaceInfo> ProcessIntelGT::GetAddressSpaces() {
  return {
      {"global", (uint64_t)DW_ASPACE_INTELGT::global,
       /*is_thread_specific=*/false},
      {"slm", (uint64_t)DW_ASPACE_INTELGT::slm,
       /*is_thread_specific=*/true},
  };
}

Status ProcessIntelGT::ReadMemoryWithSpace(addr_t addr, uint64_t addr_space,
                                           NativeThreadProtocol *thread,
                                           void *buf, size_t size,
                                           size_t &bytes_read) {

  zet_debug_memory_space_type_t ze_space;
  ze_device_thread_t ze_thread;

  if (addr_space == (uint64_t)DW_ASPACE_INTELGT::slm) {
    ze_space = ZET_DEBUG_MEMORY_SPACE_TYPE_SLM;
    if (!thread)
      return Status::FromErrorString("SLM read requires a stopped thread");
    ThreadIntelGT *t = static_cast<ThreadIntelGT *>(thread);
    ze_thread = t->GetZeThread();
  } else {
    ze_space = ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT;
    // Use the specific thread when available; the driver needs the thread
    // context to resolve per-EU-thread scratch virtual addresses.
    if (thread) {
      ThreadIntelGT *t = static_cast<ThreadIntelGT *>(thread);
      if (t->GetEUThread())
        ze_thread = t->GetZeThread();
      else
        ze_thread = ZeWildcardThread();
    } else {
      NativeThreadProtocol *current = GetCurrentThread();
      if (current) {
        ThreadIntelGT *ct = static_cast<ThreadIntelGT *>(current);
        if (ct->GetEUThread())
          ze_thread = ct->GetZeThread();
        else
          ze_thread = ZeWildcardThread();
      } else {
        ze_thread = ZeWildcardThread();
      }
    }
  }

  const DeviceSession *ds = nullptr;
  if (thread) {
    ThreadIntelGT *t = static_cast<ThreadIntelGT *>(thread);
    ds = GetDeviceSession(t->GetDeviceIndex());
  }
  if (!ds)
    ds = GetCurrentDeviceSession();
  if (!ds)
    return Status::FromErrorString("No active device session");

  zet_debug_memory_space_desc_t space_desc{};
  space_desc.stype = ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC;
  space_desc.type = ze_space;
  space_desc.address = addr;

  ze_result_t result =
      zetDebugReadMemory(ds->session, ze_thread, &space_desc, size, buf);
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat(
        "zetDebugReadMemory (space %u) failed: %s",
        static_cast<unsigned>(ze_space), ZeResultToString(result).data());
  }

  bytes_read = size;
  return Status();
}

// ---------------------------------------------------------------------------
// Software breakpoints
// ---------------------------------------------------------------------------

Status ProcessIntelGT::SetBreakpoint(addr_t addr, uint32_t /*size*/,
                                     bool hardware) {
  if (hardware)
    return Status::FromErrorString(
        "Hardware breakpoints not supported on Intel GPU");

  auto it = m_bp_saved_opcodes.find(addr);
  if (it != m_bp_saved_opcodes.end()) {
    return Status();
  }

  const DeviceSession *ds = GetCurrentDeviceSession();
  if (!ds)
    return Status::FromErrorString("No active device session for breakpoint");

  addr_t exec_addr = TranslateToExecutionAddr(addr);

  uint8_t inst[intelgt::MAX_INST_LENGTH] = {};
  size_t bytes_read = 0;
  Status error =
      ReadMemory(exec_addr, inst, intelgt::MAX_INST_LENGTH, bytes_read);
  if (error.Fail()) {
    return error;
  }

  bool is_compact = (inst[3] & 0x20) != 0;
  uint32_t inst_len =
      is_compact ? intelgt::COMPACT_INST_LENGTH : intelgt::MAX_INST_LENGTH;
  uint32_t bit_offset =
      intelgt::breakpoint_bit_offset(inst, ds->properties.deviceId);

  m_bp_saved_opcodes[addr] = std::vector<uint8_t>(inst, inst + inst_len);

  intelgt::set_inst_bit(inst, bit_offset);

  size_t bytes_written = 0;
  error = WriteMemory(exec_addr, inst, inst_len, bytes_written);
  if (error.Fail()) {
    m_bp_saved_opcodes.erase(addr);
    return error;
  }

  // Readback verification.
  {
    uint8_t rb[intelgt::MAX_INST_LENGTH] = {};
    size_t rb_n = 0;
    Status rb_err = ReadMemory(exec_addr, rb, inst_len, rb_n);
    bool match = !rb_err.Fail() && memcmp(inst, rb, inst_len) == 0;
    if (!match) {
    }
  }

  return Status();
}

Status ProcessIntelGT::RemoveBreakpoint(addr_t addr, bool hardware) {
  if (hardware)
    return Status::FromErrorString(
        "Hardware breakpoints not supported on Intel GPU");

  auto it = m_bp_saved_opcodes.find(addr);
  if (it == m_bp_saved_opcodes.end())
    return Status::FromErrorString("Breakpoint not found");

  const std::vector<uint8_t> &saved = it->second;

  size_t bytes_written = 0;
  Status error = WriteMemory(addr, saved.data(), saved.size(), bytes_written);
  m_bp_saved_opcodes.erase(it);

  if (error.Fail()) {
    return error;
  }

  return Status();
}

// ---------------------------------------------------------------------------
// Thread management
// ---------------------------------------------------------------------------

size_t ProcessIntelGT::UpdateThreads() {
  // Only add the shadow thread if no EU threads exist, so LLDB does not see
  // it in the stopped list when real GPU threads are present.
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);

  // Count non-shadow threads
  size_t eu_thread_count = 0;
  bool has_shadow = false;
  for (const auto &t : m_threads) {
    if (static_cast<ThreadIntelGT *>(t.get())->IsShadowThread()) {
      has_shadow = true;
    } else {
      eu_thread_count++;
    }
  }

  // Only add shadow thread if no EU threads exist
  if (eu_thread_count == 0 && !has_shadow) {
    m_threads.push_back(ThreadIntelGT::CreateShadowThread(*this));
  }

  // Remove shadow thread if EU threads exist
  if (eu_thread_count > 0 && has_shadow) {
    m_threads.erase(
        std::remove_if(
            m_threads.begin(), m_threads.end(),
            [](const std::unique_ptr<NativeThreadProtocol> &t) {
              return static_cast<ThreadIntelGT *>(t.get())->IsShadowThread();
            }),
        m_threads.end());
  }

  // Only set current thread if none has ever been selected; otherwise
  // preserve the TID even if that thread is currently being recreated.
  if (GetCurrentThreadID() == LLDB_INVALID_THREAD_ID) {
    if (!m_threads.empty()) {
      SetCurrentThreadID(m_threads.front()->GetID());
    }
  } else {
  }

  return m_threads.size();
}

// ---------------------------------------------------------------------------
// GetOrAllocateTidBase
//
// Return the stable TID base for a hardware EU thread, allocating simd_width
// TIDs on first sight so every physical lane keeps a fixed slot across stops.
// ---------------------------------------------------------------------------

tid_t ProcessIntelGT::GetOrAllocateTidBase(ze_device_thread_t ze_thread,
                                           uint32_t simd_width) {
  auto it = m_ze_thread_tid_map.find(ze_thread);
  if (it != m_ze_thread_tid_map.end())
    return it->second;

  // Allocate simd_width TIDs so every lane has a stable slot.
  static tid_t s_next_tid = ThreadIntelGT::INTELGT_SHADOW_THREAD_ID + 1;
  tid_t tid_base = s_next_tid;
  s_next_tid += simd_width;
  m_ze_thread_tid_map[ze_thread] = tid_base;
  return tid_base;
}

// ---------------------------------------------------------------------------
// ClearOldEUThreads
//
// Drop old EU threads from m_threads before handling new THREAD_STOPPED
// events so auto-resume does not see stale threads.
// ---------------------------------------------------------------------------

void ProcessIntelGT::ClearOldEUThreads() {
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);

  // Remove all EU lane threads (keep shadow thread)
  m_threads.erase(std::remove_if(m_threads.begin(), m_threads.end(),
                                 [](const auto &t) {
                                   auto *it =
                                       static_cast<ThreadIntelGT *>(t.get());
                                   return !it->IsShadowThread();
                                 }),
                  m_threads.end());

  // Also clear the stopped EU threads map so new stops create fresh EU threads.
  m_stopped_eu_threads.clear();

  // NOTE: Do NOT clear m_stepping_eu_threads here; HandleZeThreadStopped
  // relies on it to distinguish step completions from breakpoint hits.
}

bool ProcessIntelGT::IsAnyThreadSteppingCompleted() const {
  // Check lane threads (not EU threads) for eStopReasonTrace; lane threads
  // are updated correctly per stop while EU threads may hold stale reasons.
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
  for (const auto &t : m_threads) {
    auto *thread = static_cast<ThreadIntelGT *>(t.get());
    if (thread->IsShadowThread())
      continue;

    ThreadStopInfo stop_info;
    std::string desc;
    if (!thread->GetStopReason(stop_info, desc))
      continue;

    if (stop_info.reason == eStopReasonTrace) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// RemoveSteppedThreads
//
// Drop EU threads that completed stepping (eStopReasonTrace) so LLDB sees
// them as no longer stopped and issues vCont;c later.
// ---------------------------------------------------------------------------

void ProcessIntelGT::RemoveSteppedThreads() {
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);

  // Find EU threads with trace stop reason.
  std::vector<ze_device_thread_t> stepped_threads;
  for (const auto &[ze_thread, eu_thread] : m_stopped_eu_threads) {
    ThreadStopInfo stop_info;
    std::string desc;
    if (eu_thread->GetStopReason(stop_info, desc)) {
      if (stop_info.reason == eStopReasonTrace) {
        stepped_threads.push_back(ze_thread);
      }
    }
  }

  // Remove their lane threads from m_threads and EU threads from stopped list.
  for (const ze_device_thread_t &ze_thread : stepped_threads) {
    auto it = m_stopped_eu_threads.find(ze_thread);
    if (it == m_stopped_eu_threads.end())
      continue;

    EUThreadIntelGT *eu_ptr = it->second.get();

    // Remove all lane threads for this EU from m_threads.
    m_threads.erase(
        std::remove_if(
            m_threads.begin(), m_threads.end(),
            [eu_ptr](const std::unique_ptr<NativeThreadProtocol> &t) {
              auto *lane = static_cast<ThreadIntelGT *>(t.get());
              return lane->GetEUThread() == eu_ptr;
            }),
        m_threads.end());

    // Remove EU thread from m_stopped_eu_threads.
    m_stopped_eu_threads.erase(ze_thread);
  }
}

// ---------------------------------------------------------------------------
// HandleZeThreadStopped
//
// On THREAD_STOPPED: create the EU thread, refine the execution mask and
// stop reason from hardware, build lane threads with the focus lane
// inheriting the actual stop reason and siblings sharing it.
// ---------------------------------------------------------------------------

lldb::tid_t ProcessIntelGT::HandleZeThreadStopped(const DeviceSession &ds,
                                                  ze_device_thread_t ze_thread,
                                                  bool is_first_stop) {

  // If we already have an EU thread for this ze_thread (e.g. a step completed
  // while other EUs are still at the original breakpoint), refresh it.
  auto eu_it = m_stopped_eu_threads.find(ze_thread);
  if (eu_it != m_stopped_eu_threads.end()) {
    // Re-read stop reason and execution mask; a single-step may have crossed
    // a divergent branch so the active lane set can differ from before.
    eu_it->second->ReadExecutionMask();
    eu_it->second->ReadStopReason();
    {
      addr_t isa_base = eu_it->second->GetIsaBase();
      if (isa_base != LLDB_INVALID_ADDRESS) {
        uint32_t sw = LookupSimdWidthForIsaBase(isa_base);
        eu_it->second->SetSimdWidth(sw);
      } else {
      }
    }

    // If this EU thread was stepping, attribute the trace stop to the LANE
    // the client requested to step; fall back to the first active lane if
    // that lane diverged away during the step.
    lldb::tid_t preferred_focus = GetCurrentThreadID();
    auto step_it = m_stepping_eu_threads.find(ze_thread);
    if (step_it != m_stepping_eu_threads.end()) {
      eu_it->second->ClearStepBits();
      eu_it->second->SetStopReason(eStopReasonTrace);
      preferred_focus = step_it->second; // the lane TID the client stepped
      m_stepping_eu_threads.erase(step_it);
    }

    // Rebuild lane threads from the fresh execution mask (as on first stop):
    // remove inactive lanes, add new ones. AddLaneThreads assigns the stop
    // reason to preferred_focus if active, else to the first active lane.
    std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
    EUThreadIntelGT *eu = eu_it->second.get();
    m_threads.erase(
        std::remove_if(
            m_threads.begin(), m_threads.end(),
            [eu](const std::unique_ptr<NativeThreadProtocol> &t) {
              return static_cast<ThreadIntelGT *>(t.get())->GetEUThread() == eu;
            }),
        m_threads.end());

    tid_t tid_base =
        GetOrAllocateTidBase(ze_thread, eu_it->second->GetSimdWidth());
    tid_t focus_tid = eu_it->second->AddLaneThreads(*this, m_threads, tid_base,
                                                    preferred_focus);
    return focus_tid;
  }

  // Create a new EUThreadIntelGT; start with the fallback SIMD width and
  // refine below once ReadStopReason() caches the ISA base address.
  auto eu_thread = std::make_shared<EUThreadIntelGT>(
      ze_thread, ds.device_index, ds.session, kFallbackSimdWidth,
      ds.properties.deviceId);

  // Read the execution mask to determine active lanes.
  eu_thread->ReadExecutionMask();

  // Always read the actual stop reason from hardware first.
  eu_thread->ReadStopReason();

  // Refine simd_width using the ISA base cached by ReadStopReason.
  {
    addr_t isa_base = eu_thread->GetIsaBase();
    if (isa_base != LLDB_INVALID_ADDRESS) {
      uint32_t sw = LookupSimdWidthForIsaBase(isa_base);
      eu_thread->SetSimdWidth(sw);
    } else {
    }
  }

  // If this thread was stepping, override the hardware reason with trace.
  auto step_it = m_stepping_eu_threads.find(ze_thread);
  if (step_it != m_stepping_eu_threads.end()) {
    // Single-step completed — override hardware reason with eStopReasonTrace.
    eu_thread->ClearStepBits();
    eu_thread->SetStopReason(eStopReasonTrace);
    m_stepping_eu_threads.erase(step_it);
  }

  // Track the EU thread. nresumed bookkeeping is maintained by the caller
  // (DrainZeEvents) which decrements device->nresumed per event.
  m_stopped_eu_threads[ze_thread] = eu_thread;

  // Look up (or allocate) a stable TID base for this hardware thread.
  tid_t tid_base = GetOrAllocateTidBase(ze_thread, eu_thread->GetSimdWidth());

  // Create lane threads. The focus lane gets the real stop reason; siblings
  // share it so Thread::ShouldStop can engage the step plan on a sibling
  // lane. All lanes share the EU thread and PC (lockstep).
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);

  // Remove shadow thread when adding real EU threads to keep it out of the
  // stopped threads list.
  m_threads.erase(
      std::remove_if(
          m_threads.begin(), m_threads.end(),
          [](const std::unique_ptr<NativeThreadProtocol> &t) {
            return static_cast<ThreadIntelGT *>(t.get())->IsShadowThread();
          }),
      m_threads.end());

  tid_t focus_tid = GetCurrentThreadID();
  tid_t first_tid =
      eu_thread->AddLaneThreads(*this, m_threads, tid_base, focus_tid);

  return first_tid;
}

// ---------------------------------------------------------------------------
// HandleZeThreadUnavailable
// Remove all lane threads belonging to the given ze_device_thread_t.
// ---------------------------------------------------------------------------

void ProcessIntelGT::HandleZeThreadUnavailable(const DeviceSession &ds,
                                               ze_device_thread_t ze_thread) {

  auto eu_it = m_stopped_eu_threads.find(ze_thread);
  if (eu_it == m_stopped_eu_threads.end()) {
    return;
  }

  EUThreadIntelGT *eu = eu_it->second.get();

  // Remove all lane threads that belong to this EU thread.
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
  m_threads.erase(std::remove_if(m_threads.begin(), m_threads.end(),
                                 [eu](const auto &t) {
                                   auto *it =
                                       static_cast<ThreadIntelGT *>(t.get());
                                   return it->GetEUThread() == eu;
                                 }),
                  m_threads.end());

  m_stopped_eu_threads.erase(eu_it);
}

void ProcessIntelGT::HandleZePageFault(uint64_t fault_address) {
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
  for (auto &t : m_threads) {
    ThreadIntelGT *it = static_cast<ThreadIntelGT *>(t.get());
    if (!it->IsShadowThread() && it->GetState() == eStateStopped) {
      it->SetStopReason(eStopReasonException);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Module management
// ---------------------------------------------------------------------------

void ProcessIntelGT::HandleModuleLoad(const zet_debug_event_t &event,
                                      const DeviceSession &dev_session) {

  uint64_t begin = event.info.module.moduleBegin;
  uint64_t end = event.info.module.moduleEnd;
  uint64_t load = event.info.module.load;

  if (begin >= end) {
    return;
  }

  std::string uri =
      llvm::formatv("memory://{0:x}+{1:x}", begin, end - begin).str();

  m_loaded_modules_canonical.insert(uri);

  uint32_t ze_simd_width = 0;
  addr_t pre_linked_base =
      ScanElfModule(begin, end - begin, &dev_session, &ze_simd_width);
  if (pre_linked_base != LLDB_INVALID_ADDRESS && load != 0) {
    ModuleAddrRange range;
    range.pre_linked_base = pre_linked_base;
    range.execution_base = static_cast<addr_t>(load);
    range.size = static_cast<addr_t>(end - begin);
    range.simd_width = ze_simd_width ? ze_simd_width : kFallbackSimdWidth;
    m_module_addr_ranges.push_back(range);
  }

  m_gpu_module_manager.BeginCodeObjectListUpdate();
  for (const std::string &u : m_loaded_modules_canonical)
    m_gpu_module_manager.CodeObjectIsLoaded(u, load);
  m_gpu_module_manager.EndCodeObjectListUpdate();
}

void ProcessIntelGT::HandleModuleUnload(const zet_debug_event_t &event) {

  uint64_t begin = event.info.module.moduleBegin;
  uint64_t end = event.info.module.moduleEnd;

  if (begin >= end)
    return;

  std::string uri =
      llvm::formatv("memory://{0:x}+{1:x}", begin, end - begin).str();

  m_loaded_modules_canonical.erase(uri);

  m_gpu_module_manager.BeginCodeObjectListUpdate();
  for (const std::string &u : m_loaded_modules_canonical)
    m_gpu_module_manager.CodeObjectIsLoaded(u, 0);
  m_gpu_module_manager.EndCodeObjectListUpdate();
}

std::optional<GPUDynamicLoaderResponse>
ProcessIntelGT::GetGPUDynamicLoaderLibraryInfos(
    const GPUDynamicLoaderArgs &args) {
  // Return current state without waiting: HandleModuleLoad has already run
  // before the stop packet is sent, and any race is resolved on the next
  // query.
  GPUDynamicLoaderResponse response;

  llvm::iterator_range<GpuModuleManager::CodeObjectList::const_iterator>
      code_objects = args.full ? m_gpu_module_manager.GetLoadedCodeObjects()
                               : m_gpu_module_manager.GetChangedCodeObjects();

  for (const GpuModuleManager::CodeObject &co : code_objects) {
    GPUDynamicLoaderLibraryInfo info;
    uint64_t mem_addr = 0, mem_size = 0;
    if (sscanf(co.uri.c_str(), "memory://0x%" SCNx64 "+0x%" SCNx64, &mem_addr,
               &mem_size) == 2) {
      info.native_memory_address = static_cast<addr_t>(mem_addr);
      info.native_memory_size = static_cast<addr_t>(mem_size);
    }
    if (co.load_address != 0 && info.native_memory_address.has_value()) {
      addr_t pre_linked_base = LLDB_INVALID_ADDRESS;
      for (const ModuleAddrRange &r : m_module_addr_ranges) {
        if (r.execution_base == static_cast<addr_t>(co.load_address)) {
          pre_linked_base = r.pre_linked_base;
          break;
        }
      }
      if (pre_linked_base != LLDB_INVALID_ADDRESS) {
        info.load_address =
            static_cast<addr_t>(co.load_address) - pre_linked_base;
      } else {
        // pre_linked_base is unknown; reset load_address so the loader falls
        // back to file VAs with zero slide (setting it to the raw execution
        // VA would be used as an offset and yield wrong symbol addresses).
        info.load_address.reset();
      }
    }
    info.load = co.IsLoaded();
    info.pathname = co.uri;
    response.library_infos.push_back(info);
  }

  if (!args.full)
    m_gpu_module_manager.ClearChangedObjectList();

  return response;
}

// ---------------------------------------------------------------------------
// Misc NativeProcessProtocol overrides
// ---------------------------------------------------------------------------

addr_t ProcessIntelGT::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

const ArchSpec &ProcessIntelGT::GetArchitecture() const {
  m_arch = ArchSpec("spirv64-unknown-unknown");
  return m_arch;
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ProcessIntelGT::GetAuxvData() const {
  return nullptr;
}

Status ProcessIntelGT::GetLoadedModuleFileSpec(const char * /*module_path*/,
                                               FileSpec & /*file_spec*/) {
  return Status::FromErrorString("unimplemented");
}

Status ProcessIntelGT::GetFileLoadAddress(const llvm::StringRef & /*file_name*/,
                                          addr_t & /*load_addr*/) {
  return Status::FromErrorString("unimplemented");
}

bool ProcessIntelGT::GetProcessInfo(ProcessInstanceInfo &proc_info) {
  m_process_info.SetProcessID(m_pid);
  m_process_info.SetArchitecture(GetArchitecture());
  proc_info = m_process_info;
  return true;
}

void ProcessIntelGT::HandleNativeProcessExit(const WaitStatus &exit_status) {
  SetExitStatus(exit_status, true);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DeviceSession *
ProcessIntelGT::GetDeviceSession(uint32_t device_index) const {
  for (const DeviceSession &ds : m_device_sessions) {
    if (ds.device_index == device_index)
      return &ds;
  }
  return nullptr;
}

DeviceSession *ProcessIntelGT::GetDeviceSessionMutable(uint32_t device_index) {
  for (DeviceSession &ds : m_device_sessions) {
    if (ds.device_index == device_index)
      return &ds;
  }
  return nullptr;
}

Status ProcessIntelGT::EnsureDeviceRegistersDiscovered(uint32_t device_index) {
  for (DeviceSession &session : m_device_sessions) {
    if (session.device_index == device_index) {
      if (session.registers_discovered)
        return Status();

      Status error = DiscoverDeviceRegisterSets(session);
      if (error.Success()) {
        session.registers_discovered = true;
      }
      return error;
    }
  }
  return Status::FromErrorStringWithFormat(
      "Device %u not found in session list", device_index);
}

// ---------------------------------------------------------------------------
// Register set type constants from ZET_DEBUG_REGSET_TYPE_*.
// ---------------------------------------------------------------------------

static constexpr uint32_t kRegsetGRF = 1;
static constexpr uint32_t kRegsetAddr = 2;
static constexpr uint32_t kRegsetFlag = 3;
static constexpr uint32_t kRegsetCE = 4;
static constexpr uint32_t kRegsetSR = 5;
static constexpr uint32_t kRegsetCR = 6;
static constexpr uint32_t kRegsetTDR = 7;
static constexpr uint32_t kRegsetACC = 8;
static constexpr uint32_t kRegsetMME = 9;
static constexpr uint32_t kRegsetSP = 10;
static constexpr uint32_t kRegsetSBA = 11;
static constexpr uint32_t kRegsetDBG = 12;
static constexpr uint32_t kRegsetFC = 13;
static constexpr uint32_t kRegsetMSG = 14;
static constexpr uint32_t kRegsetModeFlags = 15;
static constexpr uint32_t kRegsetDebugScratch = 16;
static constexpr uint32_t kRegsetThreadScratch = 17;
static constexpr uint32_t kRegsetScalar = 18;

static bool IsSimdWideRegset(uint32_t regset_type) {
  switch (regset_type) {
  case kRegsetGRF:
  case kRegsetAddr:
  case kRegsetACC:
  case kRegsetMME:
  case kRegsetSP:
    return true;
  default:
    return false;
  }
}

static const char *RegsetTypeName(uint32_t type) {
  switch (type) {
  case kRegsetGRF:
    return "GRF";
  case kRegsetAddr:
    return "ADDR";
  case kRegsetFlag:
    return "FLAG";
  case kRegsetCE:
    return "CE";
  case kRegsetSR:
    return "SR";
  case kRegsetCR:
    return "CR";
  case kRegsetTDR:
    return "TDR";
  case kRegsetACC:
    return "ACC";
  case kRegsetMME:
    return "MME";
  case kRegsetSP:
    return "SP";
  case kRegsetSBA:
    return "SBA";
  case kRegsetDBG:
    return "DBG";
  case kRegsetFC:
    return "FC";
  case kRegsetMSG:
    return "MSG";
  case kRegsetModeFlags:
    return "ModeFlags";
  case kRegsetDebugScratch:
    return "DebugScratch";
  case kRegsetThreadScratch:
    return "ThreadScratch";
  case kRegsetScalar:
    return "Scalar";
  default:
    return "UNKNOWN";
  }
}

static constexpr uint32_t kCR0Index = 0;
static constexpr uint32_t kCR0_DWORD2_OFFSET_DISC = 8;

Status ProcessIntelGT::DiscoverDeviceRegisterSets(DeviceSession &session) {
  uint32_t count = 0;
  ze_result_t result =
      zetDebugGetRegisterSetProperties(session.device, &count, nullptr);
  if (result != ZE_RESULT_SUCCESS || count == 0)
    return Status::FromErrorStringWithFormat(
        "zetDebugGetRegisterSetProperties (count) failed: %s",
        ZeResultToString(result).data());

  session.regset_props.resize(count);
  result = zetDebugGetRegisterSetProperties(session.device, &count,
                                            session.regset_props.data());
  if (result != ZE_RESULT_SUCCESS) {
    session.regset_props.clear();
    return Status::FromErrorStringWithFormat(
        "zetDebugGetRegisterSetProperties (data) failed: %s",
        ZeResultToString(result).data());
  }

  // Build register tables.
  session.reg_infos.clear();
  session.reg_sets.clear();
  session.reg_set_registers.clear();
  session.reg_set_regs_ptrs.clear();
  session.reg_names.clear();
  session.reg_locations.clear();
  session.pc_reg_num = UINT32_MAX;
  session.isabase_reg_num = UINT32_MAX;

  uint32_t global_reg_num = 0;
  uint32_t cumulative_byte_offset = 0;

  for (size_t si = 0; si < session.regset_props.size(); ++si) {
    const zet_debug_regset_properties_t &props = session.regset_props[si];
    bool simd_wide = IsSimdWideRegset(props.type);
    uint32_t lane_byte_size = props.byteSize;

    std::vector<uint32_t> set_regs;
    for (uint32_t ri = 0; ri < props.count; ++ri) {
      std::string name;
      switch (props.type) {
      case kRegsetGRF:
        name = "r" + std::to_string(ri);
        break;
      case kRegsetAddr:
        name = "a" + std::to_string(ri);
        break;
      case kRegsetFlag:
        name = "f" + std::to_string(ri);
        break;
      case kRegsetCE:
        name = "ce";
        break;
      case kRegsetSR:
        name = "sr" + std::to_string(ri);
        break;
      case kRegsetCR:
        name = "cr" + std::to_string(ri);
        break;
      case kRegsetTDR:
        name = "tdr" + std::to_string(ri);
        break;
      case kRegsetACC:
        name = "acc" + std::to_string(ri);
        break;
      case kRegsetMME:
        name = "mme" + std::to_string(ri);
        break;
      case kRegsetSP:
        name = (ri == 0) ? "sp" : "sp" + std::to_string(ri);
        break;
      case kRegsetSBA: {
        // SBA register name ordering.
        static const char *sba_names[] = {
            "genstbase",  "sustbase",   "dynbase", "iobase",   "isabase",
            "blsustbase", "blsastbase", "btbase",  "scrbase0", "scrbase1"};
        name = (ri < std::size(sba_names)) ? sba_names[ri]
                                           : "sba" + std::to_string(ri);
        break;
      }
      case kRegsetDBG:
        name = "dbg" + std::to_string(ri);
        break;
      case kRegsetFC:
        name = "fc" + std::to_string(ri);
        break;
      case kRegsetModeFlags:
        name = "mf" + std::to_string(ri);
        break;
      default:
        name = "reg" + std::to_string(global_reg_num);
        break;
      }
      session.reg_names.push_back(name);

      RegisterInfo info{};
      info.name = session.reg_names.back().c_str();
      info.alt_name = nullptr;
      info.byte_size = lane_byte_size;
      info.byte_offset = cumulative_byte_offset;
      info.encoding = simd_wide ? eEncodingVector : eEncodingUint;
      info.format = eFormatHex;
      info.kinds[eRegisterKindLLDB] = global_reg_num;

      // DWARF register number mapping.
      constexpr uint32_t kDwarfCE = 1;
      constexpr uint32_t kDwarfGRFBase = 16;
      constexpr uint32_t kDwarfAddrBase = 272;
      constexpr uint32_t kDwarfFlagBase = 288;
      constexpr uint32_t kDwarfAccBase = 304;
      constexpr uint32_t kDwarfMMEBase = 320;
      uint32_t dwarf_num = LLDB_INVALID_REGNUM;
      switch (props.type) {
      case kRegsetGRF:
        dwarf_num = kDwarfGRFBase + ri;
        break;
      case kRegsetSBA: {
        // DWARF numbers for SBA registers. Indices: 0=genstbase, 1=sustbase,
        // 5=blsustbase, 6=blsastbase, 7=btbase, 8=scrbase0.
        static const uint32_t sba_dwarf[] = {7,
                                             8,
                                             LLDB_INVALID_REGNUM,
                                             LLDB_INVALID_REGNUM,
                                             LLDB_INVALID_REGNUM,
                                             9,
                                             10,
                                             5,
                                             6};
        if (ri < std::size(sba_dwarf))
          dwarf_num = sba_dwarf[ri];
        break;
      }
      case kRegsetAddr:
        dwarf_num = kDwarfAddrBase + ri;
        break;
      case kRegsetFlag:
        dwarf_num = kDwarfFlagBase + ri;
        break;
      case kRegsetACC:
        dwarf_num = kDwarfAccBase + ri;
        break;
      case kRegsetMME:
        dwarf_num = kDwarfMMEBase + ri;
        break;
      case kRegsetCE:
        if (ri == 0)
          dwarf_num = kDwarfCE;
        break;
      default:
        break;
      }
      info.kinds[eRegisterKindDWARF] = dwarf_num;

      info.kinds[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
      info.kinds[eRegisterKindProcessPlugin] = global_reg_num;
      info.flags_type = nullptr;
      info.value_regs = nullptr;
      info.invalidate_regs = nullptr;

      if (props.type == kRegsetCR && ri == kCR0Index) {
        info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
        session.pc_reg_num = global_reg_num;
      }
      if (props.type == kRegsetSBA && ri == 4)
        session.isabase_reg_num = global_reg_num;

      session.reg_infos.push_back(info);

      DeviceSession::RegLocation loc;
      loc.regset_index = static_cast<uint32_t>(si);
      loc.byte_offset = ri * props.byteSize;
      loc.byte_size = lane_byte_size;
      loc.full_byte_size = props.byteSize;
      loc.is_simd_wide = simd_wide;
      session.reg_locations.push_back(loc);

      cumulative_byte_offset += lane_byte_size;
      set_regs.push_back(global_reg_num);
      ++global_reg_num;
    }
    session.reg_set_registers.push_back(std::move(set_regs));
  }

  // Build RegisterSet array.
  for (size_t si = 0; si < session.regset_props.size(); ++si) {
    RegisterSet rs{};
    rs.name = RegsetTypeName(session.regset_props[si].type);
    rs.short_name = nullptr;
    rs.num_registers =
        static_cast<uint32_t>(session.reg_set_registers[si].size());
    rs.registers = nullptr;
    session.reg_sets.push_back(rs);
  }

  // Fix up RegisterSet::registers pointers.
  session.reg_set_regs_ptrs.resize(session.regset_props.size());
  for (size_t si = 0; si < session.regset_props.size(); ++si) {
    session.reg_set_regs_ptrs[si] = session.reg_set_registers[si].data();
    session.reg_sets[si].registers = session.reg_set_regs_ptrs[si];
  }

  // Add synthetic "pc" pseudo-register.
  if (session.pc_reg_num != UINT32_MAX) {
    const DeviceSession::RegLocation &cr0_loc =
        session.reg_locations[session.pc_reg_num];
    session.reg_names.push_back("pc");
    RegisterInfo pc_info{};
    pc_info.name = nullptr;
    pc_info.alt_name = nullptr;
    pc_info.byte_size = 8;
    pc_info.byte_offset = cumulative_byte_offset;
    pc_info.encoding = eEncodingUint;
    pc_info.format = eFormatHex;
    pc_info.kinds[eRegisterKindLLDB] = global_reg_num;
    pc_info.kinds[eRegisterKindDWARF] = 0; // DWARF IP = 0
    pc_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
    pc_info.kinds[eRegisterKindProcessPlugin] = global_reg_num;
    pc_info.flags_type = nullptr;
    pc_info.value_regs = nullptr;
    pc_info.invalidate_regs = nullptr;
    session.reg_infos.push_back(pc_info);
    DeviceSession::RegLocation pc_loc = cr0_loc;
    pc_loc.byte_size = 8;
    pc_loc.is_simd_wide = false;
    session.reg_locations.push_back(pc_loc);
    session.reg_infos[session.pc_reg_num].kinds[eRegisterKindGeneric] =
        LLDB_INVALID_REGNUM;
    uint32_t cr0_set_index = cr0_loc.regset_index;
    session.reg_set_registers[cr0_set_index].push_back(global_reg_num);
    session.reg_sets[cr0_set_index].num_registers =
        static_cast<uint32_t>(session.reg_set_registers[cr0_set_index].size());
    session.reg_set_regs_ptrs[cr0_set_index] =
        session.reg_set_registers[cr0_set_index].data();
    session.reg_sets[cr0_set_index].registers =
        session.reg_set_regs_ptrs[cr0_set_index];
    session.pc_reg_num = global_reg_num;
    ++global_reg_num;
    cumulative_byte_offset += 8; // PC is 64-bit

    // Synthetic "ip" pseudo-register: raw 32-bit instruction pointer from
    // CR0.dword2 (offset within the ISA); pc = isabase + ip.
    session.reg_names.push_back("ip");
    RegisterInfo ip_info{};
    ip_info.name = nullptr;
    ip_info.alt_name = nullptr;
    ip_info.byte_size = 4;
    ip_info.byte_offset = cumulative_byte_offset;
    ip_info.encoding = eEncodingUint;
    ip_info.format = eFormatHex;
    ip_info.kinds[eRegisterKindLLDB] = global_reg_num;
    ip_info.kinds[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    ip_info.kinds[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
    ip_info.kinds[eRegisterKindProcessPlugin] = global_reg_num;
    ip_info.flags_type = nullptr;
    ip_info.value_regs = nullptr;
    ip_info.invalidate_regs = nullptr;
    session.reg_infos.push_back(ip_info);
    DeviceSession::RegLocation ip_loc = cr0_loc;
    ip_loc.byte_size = 4;
    ip_loc.is_simd_wide = false;
    session.reg_locations.push_back(ip_loc);
    session.reg_set_registers[cr0_set_index].push_back(global_reg_num);
    session.reg_sets[cr0_set_index].num_registers =
        static_cast<uint32_t>(session.reg_set_registers[cr0_set_index].size());
    session.reg_set_regs_ptrs[cr0_set_index] =
        session.reg_set_registers[cr0_set_index].data();
    session.reg_sets[cr0_set_index].registers =
        session.reg_set_regs_ptrs[cr0_set_index];
    session.ip_reg_num = global_reg_num;
    ++global_reg_num;
    cumulative_byte_offset += 4;
  }

  // Patch reg_info name pointers (reg_names may have reallocated).
  uint32_t name_idx = 0;
  for (RegisterInfo &ri : session.reg_infos)
    ri.name = session.reg_names[name_idx++].c_str();

  return Status();
}

// ---------------------------------------------------------------------------
// ParseZeInfoMaxSimdWidth
//
// Extract the maximum simd_size across kernels from a .ze_info YAML blob.
// Returns 0 if the section is absent or unparseable.
// ---------------------------------------------------------------------------

static constexpr uint32_t kShtZebinZeInfo = 0xff000011u;

static uint32_t ParseZeInfoMaxSimdWidth(const char *yaml, size_t len) {
  // Line-by-line YAML scan looking for simd_size inside execution_env blocks.
  auto trim = [](llvm::StringRef s) { return s.trim(" \t\r\n"); };

  uint32_t max_simd = 0;
  bool in_execution_env = false;

  llvm::StringRef text(yaml, len);
  while (!text.empty()) {
    auto [line, rest] = text.split('\n');
    text = rest;

    llvm::StringRef t = trim(line);
    if (t.empty() || t.starts_with('#') || t == "---" || t == "...")
      continue;

    int indent =
        static_cast<int>(line.size()) - static_cast<int>(trim(line).size());
    // Count only leading spaces (not tabs, but zeinfo uses spaces).
    indent = 0;
    for (char c : line) {
      if (c == ' ')
        ++indent;
      else
        break;
    }

    if (indent == 4 && t == "execution_env:") {
      in_execution_env = true;
      continue;
    }
    // Any key at indent 4 that isn't execution_env ends that block.
    if (indent == 4 && t.contains(':'))
      in_execution_env = false;

    if (in_execution_env && indent == 6 && t.starts_with("simd_size:")) {
      llvm::StringRef val = trim(t.drop_front(sizeof("simd_size:") - 1));
      uint32_t v = 0;
      if (!val.getAsInteger(10, v) && v > max_simd)
        max_simd = v;
    }
  }
  return max_simd;
}

// ---------------------------------------------------------------------------
// ScanElfModule
//
// Read the zebin ELF from GPU memory: return the minimum VMA of SHF_ALLOC
// sections and (via out_simd_width) the max simd_size from .ze_info. Uses
// the caller-provided DeviceSession for multi-device correctness. On Xe2+
// module_begin is GPU VRAM (not CPU-accessible), so we always read via the
// debug session.
// ---------------------------------------------------------------------------

addr_t ProcessIntelGT::ScanElfModule(addr_t module_begin, addr_t module_size,
                                     const DeviceSession *ds,
                                     uint32_t *out_simd_width) {
  if (out_simd_width)
    *out_simd_width = 0;

  if (!ds || !ds->session) {
    return LLDB_INVALID_ADDRESS;
  }

  // Validate module_size is large enough to contain an ELF header.
  if (module_size < sizeof(llvm::ELF::Elf64_Ehdr)) {
    return LLDB_INVALID_ADDRESS;
  }

  ze_device_thread_t ze_thread = ZeWildcardThread();
  zet_debug_memory_space_desc_t space_desc{};
  space_desc.stype = ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC;
  space_desc.type = ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT;

  llvm::ELF::Elf64_Ehdr ehdr{};
  space_desc.address = module_begin;
  ze_result_t result = zetDebugReadMemory(ds->session, ze_thread, &space_desc,
                                          sizeof(ehdr), &ehdr);
  if (result != ZE_RESULT_SUCCESS) {
    return LLDB_INVALID_ADDRESS;
  }

  if (!ehdr.checkMagic())
    return LLDB_INVALID_ADDRESS;

  // Ensure we are dealing with a 64-bit little-endian ELF before using Elf64_*
  if (ehdr.e_ident[llvm::ELF::EI_CLASS] != llvm::ELF::ELFCLASS64)
    return LLDB_INVALID_ADDRESS;
  if (ehdr.e_ident[llvm::ELF::EI_DATA] != llvm::ELF::ELFDATA2LSB)
    return LLDB_INVALID_ADDRESS;

  if (ehdr.e_shoff == 0 || ehdr.e_shentsize < sizeof(llvm::ELF::Elf64_Shdr) ||
      ehdr.e_shnum == 0)
    return LLDB_INVALID_ADDRESS;

  // Validate section header table fits within module_size and doesn't overflow.
  // Check: e_shoff + (e_shnum * e_shentsize) <= module_size
  uint64_t section_table_size = (uint64_t)ehdr.e_shnum * ehdr.e_shentsize;
  if (section_table_size / ehdr.e_shentsize != ehdr.e_shnum) {
    // Multiplication overflowed.
    return LLDB_INVALID_ADDRESS;
  }
  if (ehdr.e_shoff > module_size ||
      section_table_size > module_size - ehdr.e_shoff) {
    return LLDB_INVALID_ADDRESS;
  }

  addr_t min_vma = LLDB_INVALID_ADDRESS;
  bool scan_complete = true;

  for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
    uint64_t section_offset = ehdr.e_shoff + (uint64_t)i * ehdr.e_shentsize;
    if (section_offset + sizeof(llvm::ELF::Elf64_Shdr) > module_size) {
      scan_complete = false;
      break;
    }

    // Check module_begin + section_offset doesn't overflow addr_t.
    if (section_offset > std::numeric_limits<addr_t>::max() - module_begin) {
      scan_complete = false;
      break;
    }

    llvm::ELF::Elf64_Shdr shdr{};
    space_desc.address = module_begin + section_offset;
    result = zetDebugReadMemory(ds->session, ze_thread, &space_desc,
                                sizeof(shdr), &shdr);
    if (result != ZE_RESULT_SUCCESS) {
      scan_complete = false;
      break;
    }

    if ((shdr.sh_flags & llvm::ELF::SHF_ALLOC) && shdr.sh_addr != 0) {
      if (min_vma == LLDB_INVALID_ADDRESS || shdr.sh_addr < min_vma)
        min_vma = static_cast<addr_t>(shdr.sh_addr);
    }

    // Read the .ze_info section body (sh_type == 0xff000011).
    if (out_simd_width && shdr.sh_type == kShtZebinZeInfo && shdr.sh_size > 0 &&
        shdr.sh_offset + shdr.sh_size <= module_size) {
      uint64_t data_addr = module_begin + shdr.sh_offset;
      if (shdr.sh_offset <= std::numeric_limits<addr_t>::max() - module_begin) {
        std::vector<char> zeinfo_buf(shdr.sh_size);
        space_desc.address = data_addr;
        ze_result_t ze_res =
            zetDebugReadMemory(ds->session, ze_thread, &space_desc,
                               shdr.sh_size, zeinfo_buf.data());
        if (ze_res == ZE_RESULT_SUCCESS) {
          uint32_t sw =
              ParseZeInfoMaxSimdWidth(zeinfo_buf.data(), shdr.sh_size);
          *out_simd_width = sw;
        } else {
        }
      }
    }
  }

  // If scan incomplete, return LLDB_INVALID_ADDRESS even if we found some
  // allocated sections. The lowest-VMA section might be after the error.
  if (!scan_complete) {
    return LLDB_INVALID_ADDRESS;
  }

  return min_vma;
}

uint32_t ProcessIntelGT::LookupSimdWidthForIsaBase(addr_t isa_base) const {
  for (const ModuleAddrRange &r : m_module_addr_ranges) {
    if (isa_base >= r.execution_base && isa_base < r.execution_base + r.size)
      return r.simd_width;
  }
  return kFallbackSimdWidth;
}

addr_t ProcessIntelGT::TranslateToExecutionAddr(addr_t addr) const {
  for (const ModuleAddrRange &r : m_module_addr_ranges) {
    if (addr >= r.pre_linked_base && addr < r.pre_linked_base + r.size) {
      addr_t offset = addr - r.pre_linked_base;
      return r.execution_base + offset;
    }
  }
  return addr;
}

const DeviceSession *ProcessIntelGT::GetCurrentDeviceSession() const {
  NativeThreadProtocol *cur =
      const_cast<ProcessIntelGT *>(this)->GetCurrentThread();
  if (cur) {
    ThreadIntelGT *t = static_cast<ThreadIntelGT *>(cur);
    if (!t->IsShadowThread()) {
      const DeviceSession *ds = GetDeviceSession(t->GetDeviceIndex());
      if (ds)
        return ds;
    }
  }
  if (!m_device_sessions.empty())
    return &m_device_sessions[0];
  return nullptr;
}

ThreadIntelGT *ProcessIntelGT::FindThread(tid_t tid) {
  std::lock_guard<std::recursive_mutex> guard(m_threads_mutex);
  for (auto &t : m_threads) {
    if (t->GetID() == tid)
      return static_cast<ThreadIntelGT *>(t.get());
  }
  return nullptr;
}

ThreadIntelGT *ProcessIntelGT::GetCurrentThreadIntelGT() {
  return static_cast<ThreadIntelGT *>(GetCurrentThread());
}

void ProcessIntelGT::SetLaunchInfo(ProcessLaunchInfo &launch_info) {
  static_cast<ProcessInfo &>(m_process_info) =
      static_cast<const ProcessInfo &>(launch_info);
}

// ---------------------------------------------------------------------------
// GetCurrentEUThread
// EUThreadIntelGT for the current lane thread, or nullptr for shadow/none.
// ---------------------------------------------------------------------------

EUThreadIntelGT *ProcessIntelGT::GetCurrentEUThread() {
  ThreadIntelGT *t = GetCurrentThreadIntelGT();
  if (t && !t->IsShadowThread())
    return t->GetEUThread();
  return nullptr;
}

// ---------------------------------------------------------------------------
// FindResumeActionForEUThread
//
// Find an action matching any lane thread of the EU thread. Stepping wins
// over running so vCont;s:<tid>;c does not stop at a non-stepping lane
// before checking the stepping lane.
// ---------------------------------------------------------------------------

const ResumeAction *
ProcessIntelGT::FindResumeActionForEUThread(const ResumeActionList &actions,
                                            const EUThreadIntelGT *eu) {
  const ResumeAction *fallback = nullptr;
  for (auto &t : m_threads) {
    auto *lane = static_cast<ThreadIntelGT *>(t.get());
    if (lane->GetEUThread() == eu) {
      // Check for a SPECIFIC action for this lane (no default fallback).
      const ResumeAction *a = actions.GetActionForThread(lane->GetID(), false);
      if (a) {
        if (a->state == eStateStepping)
          return a; // Stepping takes priority.
        if (!fallback)
          fallback = a;
      }
    }
  }
  return fallback ? fallback
                  : actions.GetActionForThread(LLDB_INVALID_THREAD_ID, true);
}

// ---------------------------------------------------------------------------
// ProcessManagerIntelGT
// ---------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerIntelGT::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  lldb::pid_t pid = launch_info.GetProcessID();
  auto proc_up = std::make_unique<ProcessIntelGT>(pid, native_delegate,
                                                  m_plugin, m_device_sessions);
  proc_up->SetLaunchInfo(launch_info);
  return proc_up;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
ProcessManagerIntelGT::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  return llvm::createStringError("Unimplemented function");
}

unsigned int
ProcessIntelGT::GetOrAssignKernelInstanceID(uint32_t implicit_args) {
  std::lock_guard<std::mutex> lock(m_kernel_instances_mutex);

  unsigned int num_kernels = m_kernel_instances.size();
  auto it = m_kernel_instances.insert({implicit_args, num_kernels});
  return (*it.first).second;
}
