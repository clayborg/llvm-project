//===-- LLDBServerPluginIntelGT.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginIntelGT.h"
#include "LevelZeroHelpers.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerLLGS.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "ProcessIntelGT.h"
#include "ThreadIntelGT.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/GPUGDBRemotePackets.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Error.h"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <signal.h>
#include <thread>
#include <unistd.h>

// Elapsed time helper — returns ms since first call.
static uint64_t elapsed_ms() {
  static auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------

// Identifier used to tag the GPU-attach trigger breakpoint.
static constexpr uint32_t kZeModuleCreateBpId = 1;

// Symbol whose hit triggers GPU attach. By the time zeModuleCreate fires the
// inferior has finished zeInit()/zeContextCreate(), so our zeInit() runs
// quickly on the background thread without contending with GPU driver init.
static constexpr const char *kZeModuleCreateSymbol = "zeModuleCreate";

// Maximum threads per device used for TID space partitioning when exact
// topology is not yet known.
static constexpr uint64_t kDefaultMaxThreadsPerDevice = 1024 * 1024;

// ---------------------------------------------------------------------------
// LLDBServerPluginIntelGT constructor / destructor
// ---------------------------------------------------------------------------

LLDBServerPluginIntelGT::LLDBServerPluginIntelGT(
    LLDBServerPlugin::GDBServer &native_process, MainLoop &main_loop)
    : LLDBServerPlugin(native_process, main_loop) {
  // Initialize IntelGT log channel

  m_process_manager_up.reset(new ProcessManagerIntelGT(main_loop));
  m_gdb_server.reset(new GDBRemoteCommunicationServerLLGS(
      m_main_loop, *m_process_manager_up, "intelgt-gpu.server"));
  m_gdb_server->SetPlugin(this);

  // Create the wakeup pipe.  The read end is registered with the main loop
  // after zetDebugAttach() succeeds; writing one byte to the write end
  // re-arms the loop to call NativeProcessIsStopping() again.
  if (::pipe(m_notifier_fd) != 0) {
    m_notifier_fd[0] = INVALID_FD;
    m_notifier_fd[1] = INVALID_FD;
  }

  // Do NOT call zeInit() here: Driver::driverInit uses std::call_once, so
  // constructing during inferior GPU init would block ~75 s. zeInit() runs
  // later from BackgroundAttach after the CPU is confirmed running.
}

LLDBServerPluginIntelGT::~LLDBServerPluginIntelGT() {
  StopZeEventPollThread();
  if (m_attach_thread.joinable())
    m_attach_thread.join();
  for (int fd : m_notifier_fd) {
    if (fd != INVALID_FD)
      ::close(fd);
  }
}

// ---------------------------------------------------------------------------
// GetInitializeActions
// ---------------------------------------------------------------------------

GPUActions LLDBServerPluginIntelGT::GetInitializeActions() {
  GPUActions actions = GetNewGPUAction();

  GPUBreakpointByName bp_name;
  bp_name.function_name = kZeModuleCreateSymbol;
  // shlib intentionally unset so the SYCL Unified Runtime path
  // (libur_adapter_level_zero.so) is also matched, not only libze_loader.so.

  GPUBreakpointInfo bp;
  bp.identifier = kZeModuleCreateBpId;
  bp.name_info = std::move(bp_name);
  bp.symbol_names = {kZeModuleCreateSymbol};
  actions.breakpoints.emplace_back(std::move(bp));

  return actions;
}

// ---------------------------------------------------------------------------
// ZE event poll thread
// ---------------------------------------------------------------------------
// Continuously calls zetDebugReadEvent; on an event, wakes the main loop
// via the notifier pipe so the notifier callback can drain and process.

void LLDBServerPluginIntelGT::StartZeEventPollThread() {
  if (m_ze_poll_thread.joinable())
    return;
  m_ze_poll_stop.store(false);
  m_ze_poll_thread = std::thread([this]() {
    while (!m_ze_poll_stop.load()) {
      bool any = false;
      for (const DeviceSession &ds : m_device_sessions) {
        zet_debug_event_t event{};
        // Block up to 50ms — responsive but not busy-spinning.
        ze_result_t result =
            zetDebugReadEvent(ds.session, /*timeout_ms=*/50, &event);
        if (result == ZE_RESULT_SUCCESS) {
          // Push to queue; main loop processes and ACKs appropriately.
          {
            std::lock_guard<std::mutex> g(m_ze_event_queue_mutex);
            m_ze_event_queue.push_back({ds.device_index, ds.session, event});
          }
          any = true;
        } else if (result != ZE_RESULT_NOT_READY) {
          m_ze_poll_stop.store(true);
          break;
        }
      }
      if (any)
        TriggerNotifier(); // wake main loop to process queued events
    }
  });
}

void LLDBServerPluginIntelGT::StopZeEventPollThread() {
  m_ze_poll_stop.store(true);
  if (m_ze_poll_thread.joinable())
    m_ze_poll_thread.join();
}

// ---------------------------------------------------------------------------
// BreakpointWasHit
// ---------------------------------------------------------------------------
// Runs synchronously while the CPU is stopped at zeModuleCreate: zeInit,
// AttachToDevices, CreateGpuProcess, CreateConnection; then returns
// connect_info and the CPU resumes.
// ---------------------------------------------------------------------------

llvm::Expected<GPUPluginBreakpointHitResponse>
LLDBServerPluginIntelGT::BreakpointWasHit(GPUPluginBreakpointHitArgs &args) {

  GPUPluginBreakpointHitResponse response(GetNewGPUAction());

  if (args.breakpoint.identifier != kZeModuleCreateBpId) {
    return response;
  }

  response.disable_bp = true;
  m_zemodulecreate_hit = true;

  if (m_ze_state != ZeState::Uninitialized &&
      m_ze_state != ZeState::Initialized) {
    return response;
  }

  // Everything runs synchronously while the CPU is stopped at zeModuleCreate.
  // zetDebugAttach returns in ~1 ms (waits only for the queued snapshot);
  // MODULE_LOAD arrives later, ACK is held until LLDB sets breakpoints.
  Status error = InitializeLevelZero();
  if (error.Fail()) {
    m_ze_state = ZeState::Error;
    return response;
  }

  error = AttachToDevices();
  if (error.Fail()) {
    m_ze_state = ZeState::Error;
    return response;
  }

  error = CreateGpuProcess();
  if (error.Fail()) {
    m_ze_state = ZeState::Error;
    return response;
  }

  error = InstallNotifierOnMainLoop();
  if (error.Fail()) {
    m_ze_state = ZeState::Error;
    return response;
  }

  // Start the ZE event polling thread: continuously call zetDebugReadEvent,
  // queue events, and wake the main loop via TriggerNotifier so DrainZeEvents
  // processes them on the right thread.
  StartZeEventPollThread();
  TriggerNotifier();

  auto conn_info = CreateConnection();
  if (conn_info) {
    response.actions.connect_info = conn_info;
  }

  response.actions.resume_gpu_process = true;

  return response;
}

// ---------------------------------------------------------------------------
// NativeProcessIsStopping
// ---------------------------------------------------------------------------

std::optional<GPUActions> LLDBServerPluginIntelGT::NativeProcessIsStopping() {
  // Log the CPU stop reason so we can see why the process stopped each time.
  {
    NativeProcessProtocol *cpu = GetNativeProcess();
    NativeThreadProtocol *thread = cpu ? cpu->GetCurrentThread() : nullptr;
    if (thread) {
      ThreadStopInfo si{};
      std::string desc;
      thread->GetStopReason(si, desc);
      static const char *reason_names[] = {
          "Invalid",
          "None",
          "Trace",
          "Breakpoint",
          "Watchpoint",
          "Signal",
          "Exception",
          "Exec",
          "PlanComplete",
          "ThreadExiting",
          "Instrumentation",
          "ProcessorTrace",
          "Fork",
          "VFork",
          "VForkDone",
          "Interrupt",
          "???",
          "HistoryBound",
          "???",
          "DynamicLoader",
      };
      [[maybe_unused]] const char *reason_str =
          (si.reason < sizeof(reason_names) / sizeof(reason_names[0]))
              ? reason_names[si.reason]
              : "???";
    } else {
    }
  }

  if (m_attach_failed.load()) {
    if (m_attach_thread.joinable())
      m_attach_thread.join();
    m_ze_state = ZeState::Error;
    return std::nullopt;
  }

  // If the poll thread delivered a MODULE_LOAD and halted the GPU, set
  // load_libraries=true here so LLDB calls GetGPUDynamicLoaderLibraryInfos.
  if (m_pending_library_load && m_is_connected) {
    m_pending_library_load = false;
    GPUActions actions = GetNewGPUAction();
    actions.load_libraries = true;
    // Do NOT set resume_gpu_process: HandleGPUActions' PrivateResume would
    // deadlock and later poison stop events, making all future GPU stops
    // auto-continue. The shadow-thread MODULE_LOAD stop is unhandled, so
    // LLDB continues on its own via vCont;c -> Resume -> ACK -> BP hit.
    return actions;
  }

  // GPU breakpoint stop: routed through the CPU. Halt the CPU and return
  // wait_for_gpu_process_to_stop so HandleGPUActions waits before prompting.
  if (m_pending_gpu_stop && m_is_connected) {
    m_pending_gpu_stop = false;
    ProcessIntelGT *gpu_proc = GetGPUProcess();
    GPUActions actions = GetNewGPUAction();
    actions.wait_for_gpu_process_to_stop = true;
    if (gpu_proc)
      actions.stop_id = gpu_proc->GetStopID();

    // Auto-resume CPU immediately after delivering GPU stop event.
    // This prevents CPU from staying stuck at SIGINT.
    NativeProcessProtocol *cpu = GetNativeProcess();
    if (cpu && cpu->GetState() == StateType::eStateStopped) {
      ResumeActionList cpu_actions;
      ResumeAction action;
      action.tid = LLDB_INVALID_THREAD_ID;
      action.state = StateType::eStateRunning;
      action.signal = 0; // Don't deliver the SIGINT
      cpu_actions.Append(action);
      Status error = cpu->Resume(cpu_actions);
      if (error.Fail()) {
      } else {
        m_cpu_halted_for_gpu_stop = false;
      }
    }

    return actions;
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// NativeProcessDidExit
// ---------------------------------------------------------------------------

void LLDBServerPluginIntelGT::NativeProcessDidExit(
    const WaitStatus &exit_status) {
  StopZeEventPollThread();

  // Detach all ZE debug sessions.
  for (const DeviceSession &ds : m_device_sessions) {
    if (ds.session) {
      ze_result_t result = zetDebugDetach(ds.session);
      if (result != ZE_RESULT_SUCCESS) {
      }
    }
  }
  m_device_sessions.clear();
  m_ze_state = ZeState::Detached;

  // Notify the GPU process.
  ProcessIntelGT *gpu_proc = GetGPUProcess();
  if (gpu_proc)
    gpu_proc->HandleNativeProcessExit(exit_status);

  // Unregister the notifier.
  m_gpu_event_read_up.reset();
  m_gpu_event_io_obj_sp.reset();

  // Close the notifier pipe.
  for (int &fd : m_notifier_fd) {
    if (fd != INVALID_FD) {
      ::close(fd);
      fd = INVALID_FD;
    }
  }
}

// ---------------------------------------------------------------------------
// GetGPUDynamicLoaderLibraryInfos
// ---------------------------------------------------------------------------

std::optional<GPUDynamicLoaderResponse>
LLDBServerPluginIntelGT::GetGPUDynamicLoaderLibraryInfos(
    const GPUDynamicLoaderArgs &args) {
  ProcessIntelGT *proc = GetGPUProcess();
  if (!proc)
    return std::nullopt;
  return proc->GetGPUDynamicLoaderLibraryInfos(args);
}

// ---------------------------------------------------------------------------
// InitializeLevelZero
// ---------------------------------------------------------------------------

Status LLDBServerPluginIntelGT::InitializeLevelZero() {

  ze_result_t result = zeInit(0);

  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat("zeInit failed: %s",
                                             ZeResultToString(result).data());
  }

  m_ze_state = ZeState::Initialized;
  return Status();
}

// ---------------------------------------------------------------------------
// AttachToDevices
// ---------------------------------------------------------------------------

Status LLDBServerPluginIntelGT::AttachToDevices() {

  uint32_t driver_count = 0;
  ze_result_t result = zeDriverGet(&driver_count, nullptr);
  if (result != ZE_RESULT_SUCCESS || driver_count == 0) {
    return Status::FromErrorString("No Level Zero drivers found");
  }

  std::vector<ze_driver_handle_t> drivers(driver_count);
  result = zeDriverGet(&driver_count, drivers.data());
  if (result != ZE_RESULT_SUCCESS) {
    return Status::FromErrorStringWithFormat("zeDriverGet failed: %s",
                                             ZeResultToString(result).data());
  }

  uint32_t device_index = 0;
  uint64_t tid_base = 2; // TID 1 is shadow thread.

  for (ze_driver_handle_t driver : drivers) {
    uint32_t device_count = 0;
    result = zeDeviceGet(driver, &device_count, nullptr);
    if (result != ZE_RESULT_SUCCESS || device_count == 0)
      continue;

    std::vector<ze_device_handle_t> devices(device_count);
    result = zeDeviceGet(driver, &device_count, devices.data());
    if (result != ZE_RESULT_SUCCESS)
      continue;

    for (ze_device_handle_t device : devices) {
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      zeDeviceGetProperties(device, &props);

      // Only attach to Intel GPU devices.
      if (props.vendorId != 0x8086 || props.type != ZE_DEVICE_TYPE_GPU) {
        continue;
      }

      // Try sub-devices first, then fall back to the parent device.
      uint32_t subdev_count = 0;
      zeDeviceGetSubDevices(device, &subdev_count, nullptr);
      bool attached_to_subdev = false;

      if (subdev_count > 0) {
        std::vector<ze_device_handle_t> subdevices(subdev_count);
        zeDeviceGetSubDevices(device, &subdev_count, subdevices.data());

        for (ze_device_handle_t subdev : subdevices) {
          DeviceSession ds;
          ds.device_index = device_index;
          // Copy parent properties; sub-device may have same deviceId.
          ds.properties = props;
          ds.tid_base = tid_base;
          ds.max_threads = kDefaultMaxThreadsPerDevice;
          // The whole device is debugged, so all EU threads start
          // "resumed" (running). nthreads = nresumed = total.
          ds.nthreads = static_cast<size_t>(props.numSlices) *
                        props.numSubslicesPerSlice * props.numEUsPerSubslice *
                        props.numThreadsPerEU;
          ds.nresumed = ds.nthreads;

          if (AttachToDevice(subdev, device_index, tid_base,
                             kDefaultMaxThreadsPerDevice, ds)) {
            m_device_sessions.push_back(ds);
            tid_base += kDefaultMaxThreadsPerDevice;
            ++device_index;
            attached_to_subdev = true;
          }
        }
      }

      if (!attached_to_subdev) {
        // Fall back to attaching to the parent device.
        DeviceSession ds;
        ds.device_index = device_index;
        ds.properties = props;
        ds.tid_base = tid_base;
        ds.max_threads = kDefaultMaxThreadsPerDevice;
        // Whole device debugged: all EU threads start resumed.
        // nthreads = nresumed = total device threads.
        ds.nthreads = static_cast<size_t>(props.numSlices) *
                      props.numSubslicesPerSlice * props.numEUsPerSubslice *
                      props.numThreadsPerEU;
        ds.nresumed = ds.nthreads;

        if (AttachToDevice(device, device_index, tid_base,
                           kDefaultMaxThreadsPerDevice, ds)) {
          m_device_sessions.push_back(ds);
          tid_base += kDefaultMaxThreadsPerDevice;
          ++device_index;
        }
      }
    }
  }

  if (m_device_sessions.empty()) {
    return Status::FromErrorString(
        "No Intel GPU devices could be attached for debugging");
  }

  m_ze_state = ZeState::Attached;
  return Status();
}

// ---------------------------------------------------------------------------
// AttachToDevice
// ---------------------------------------------------------------------------

bool LLDBServerPluginIntelGT::AttachToDevice(ze_device_handle_t device,
                                             uint32_t device_index,
                                             uint64_t tid_base,
                                             uint64_t max_threads,
                                             DeviceSession &session_out) {

  NativeProcessProtocol *native = GetNativeProcess();
  if (!native) {
    return false;
  }

  zet_debug_config_t config{};
  config.pid = static_cast<uint32_t>(native->GetID());

  zet_debug_session_handle_t session = nullptr;
  ze_result_t result = zetDebugAttach(device, &config, &session);

  if (result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return false;
  }

  if (result == ZE_RESULT_ERROR_NOT_AVAILABLE) {
    return false;
  }

  if (result != ZE_RESULT_SUCCESS) {
    return false;
  }

  session_out.device = device;
  session_out.session = session;
  session_out.device_index = device_index;
  session_out.tid_base = tid_base;
  session_out.max_threads = max_threads;

  return true;
}

// ---------------------------------------------------------------------------
// CreateGpuProcess
// ---------------------------------------------------------------------------

Status LLDBServerPluginIntelGT::CreateGpuProcess() {

  ProcessManagerIntelGT *manager =
      static_cast<ProcessManagerIntelGT *>(m_process_manager_up.get());
  manager->m_plugin = this;
  manager->m_device_sessions = m_device_sessions;

  ProcessLaunchInfo info;
  info.GetFlags().Set(eLaunchFlagStopAtEntry | eLaunchFlagDebug |
                      eLaunchFlagDisableASLR);
  Args args;
  args.AppendArgument("/pretend/path/to/intelgt-gpu");
  info.SetArguments(args, true);
  info.GetEnvironment() = Host::GetEnvironment();
  // Use the CPU process PID as the GPU process PID.
  NativeProcessProtocol *native = GetNativeProcess();
  info.SetProcessID(native ? native->GetID() : 0);

  m_gdb_server->SetLaunchInfo(info);
  Status status = m_gdb_server->LaunchProcess();
  if (status.Success())
    GetGPUProcess()->UpdateThreads();

  return status;
}

// ---------------------------------------------------------------------------
// InstallNotifierOnMainLoop
// ---------------------------------------------------------------------------

Status LLDBServerPluginIntelGT::InstallNotifierOnMainLoop() {
  if (m_notifier_fd[0] == INVALID_FD)
    return Status::FromErrorString("Notifier pipe not created");

  Status error;
  m_gpu_event_io_obj_sp =
      std::make_shared<GPUIOObjectIntelGT>(m_notifier_fd[0]);
  m_gpu_event_read_up = m_main_loop.RegisterReadObject(
      m_gpu_event_io_obj_sp,
      [this](MainLoopBase &) {
        // Drain the wake byte.
        char buf;
        ssize_t n;
        do {
          n = ::read(m_notifier_fd[0], &buf, 1);
        } while (n == -1 && errno == EINTR);

        // Drain ZE events (MODULE_LOAD, THREAD_STOPPED, etc.) while the
        // GPU process is running, independent of CPU stops.
        if (m_ze_state == ZeState::Attached ||
            m_ze_state == ZeState::RuntimeLoaded) {
          ProcessIntelGT *gpu_proc =
              static_cast<ProcessIntelGT *>(m_gdb_server->GetCurrentProcess());
          if (gpu_proc) {
            GPUActions actions = GetNewGPUAction();
            if (DrainZeEvents(actions)) {
              if (actions.load_libraries) {
                // The MODULE_LOAD ACK is held in m_pending_module_acks; GPU
                // execution blocks until Resume() ACKs. Halt() delivers a
                // GPU T packet with load_libraries=true so LLDB loads the
                // ELF and sets breakpoints before the user continues.
                m_pending_library_load = true;
                gpu_proc->Halt(); // GPU T packet with load_libraries=true
              }
            }
          }
        }
      },
      error);
  return error;
}

// ---------------------------------------------------------------------------
// TriggerNotifier / RearmNotifier
// ---------------------------------------------------------------------------

void LLDBServerPluginIntelGT::TriggerNotifier() {
  if (m_notifier_fd[1] == INVALID_FD)
    return;
  char byte = 1;
  ssize_t n;
  do {
    n = ::write(m_notifier_fd[1], &byte, 1);
  } while (n == -1 && errno == EINTR);
}

void LLDBServerPluginIntelGT::RearmNotifier() {
  // Write to the write end to re-arm the main loop.
  TriggerNotifier();
}

// ---------------------------------------------------------------------------
// CreateConnection
// ---------------------------------------------------------------------------

std::optional<GPUPluginConnectionInfo>
LLDBServerPluginIntelGT::CreateConnection() {
  std::lock_guard<std::mutex> guard(m_connect_mutex);

  if (m_is_connected || m_is_listening)
    return std::nullopt;

  m_is_listening = true;

  llvm::Expected<std::unique_ptr<TCPSocket>> sock =
      Socket::TcpListen("localhost:0", 5);
  if (!sock) {
    m_is_listening = false;
    return std::nullopt;
  }

  const uint16_t port = (*sock)->GetLocalPortNumber();

  m_listen_socket = std::move(*sock);

  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> res =
      m_listen_socket->Accept(
          m_main_loop, [this, port](std::unique_ptr<Socket> socket) {
            std::unique_ptr<Connection> conn(
                new ConnectionFileDescriptor(std::move(socket)));
            m_gdb_server->InitializeConnection(std::move(conn));
            m_is_connected = true;
          });
  if (!res) {
  } else {
    m_read_handles = std::move(*res);
  }

  GPUPluginConnectionInfo info;
  info.connect_url = llvm::formatv("connect://localhost:{}", port).str();
  info.synchronous = true;
  // spirv64 matches the GPU ELF's parsed architecture (EM_INTELGT=205).
  info.triple = "spirv64-unknown-unknown";
  // Copy CPU breakpoints (e.g. b xxlgrf.cpp:69) to the GPU target so they
  // are resolved against GPU DWARF when the GPU module loads.
  info.copy_cpu_breakpoints_during_attaching = true;
  return info;
}

// ---------------------------------------------------------------------------
// DrainZeEvents
// ---------------------------------------------------------------------------

bool LLDBServerPluginIntelGT::DrainZeEvents(GPUActions &actions) {
  bool any_change = false;

  ProcessIntelGT *gpu_proc = GetGPUProcess();
  if (!gpu_proc) {
    return false;
  }

  lldb::tid_t first_stopped_tid = LLDB_INVALID_THREAD_ID;
  bool any_thread_stopped = false;
  size_t threads_stopped_this_call = 0;

  // Phase 1: drain the queue filled by the ZE poll thread.
  // Drain ALL queued events before halting the CPU; otherwise other
  // THREAD_STOPPED events queue up as separate breakpoint hits.
  auto drain_once = [&]() -> size_t {
    size_t stops_before = threads_stopped_this_call;
    std::vector<ZeQueuedEvent> queued;
    {
      std::lock_guard<std::mutex> g(m_ze_event_queue_mutex);
      queued.swap(m_ze_event_queue);
    }
    for (auto &qe : queued) {
      // Process the queued event exactly like a live one.
      const zet_debug_event_t &event = qe.event;
      const DeviceSession *ds = nullptr;
      for (const auto &s : m_device_sessions)
        if (s.device_index == qe.device_index) {
          ds = &s;
          break;
        }
      if (!ds)
        continue;

      switch (event.type) {
      case ZET_DEBUG_EVENT_TYPE_INVALID:
        break;

      case ZET_DEBUG_EVENT_TYPE_DETACHED:
        m_ze_state = ZeState::Detached;
        any_change = true;
        break;

      case ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY:
        m_ze_state = ZeState::RuntimeLoaded;
        any_change = true;
        break;

      case ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT:
        gpu_proc->HandleNativeProcessExit(WaitStatus{WaitStatus::Exit, 0});
        m_ze_state = ZeState::Detached;
        any_change = true;
        break;

      case ZET_DEBUG_EVENT_TYPE_MODULE_LOAD: {
        bool need_ack = (event.flags & ZET_DEBUG_EVENT_FLAG_NEED_ACK) != 0;
        gpu_proc->HandleModuleLoad(event, *ds);
        actions.load_libraries = true;
        any_change = true;
        if (need_ack) {
          // Hold the ACK in m_pending_module_acks.  GPU execution stays blocked
          // (NEED_ACK hold in i915) while LLDB loads the GPU ELF and sets BPs.
          // ACK is sent in ProcessIntelGT::Resume() after LLDB continues.
          gpu_proc->m_pending_module_acks.push_back({ds->session, event});
          continue; // skip ZeAckEvent at bottom of loop
        }
        break;
      }

      case ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD:
        gpu_proc->HandleModuleUnload(event);
        // Do not set load_libraries=true for unloads; there is nothing to
        // process and reporting the stop confuses shutdown.
        any_change = true;
        break;

      case ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED: {
        ze_device_thread_t ze_thread = event.info.thread.thread;
        bool was_wildcard = ZeThreadIsWildcard(ze_thread);
        // nresumed/ninterrupts maintenance: specific event decrements
        // nresumed by 1; wildcard drives nresumed to 0 and decrements
        // ninterrupts (wildcard is the response to interrupt-all).
        if (DeviceSession *mds =
                gpu_proc->GetDeviceSessionMutable(qe.device_index)) {
          if (was_wildcard) {
            mds->nresumed = 0;
            if (mds->ninterrupts > 0)
              mds->ninterrupts--;
          } else if (mds->nresumed > 0) {
            mds->nresumed--;
          }
        }
        if (was_wildcard) {
          ze_thread = ze_device_thread_t{0, 0, 0, 0};
        }
        {
          // Clear old EU threads before creating new ones so auto-resume
          // does not see stale threads. Skip clearing on step completion,
          // while waiting for more stops, and mid-batch of a multi-thread
          // stop.
          bool is_first = (first_stopped_tid == LLDB_INVALID_THREAD_ID);
          bool is_step_completion = gpu_proc->IsSteppingThread(ze_thread);
          bool waiting_for_more = (m_expected_stopped_count > 0);
          size_t num_already_stopped = gpu_proc->GetStoppedEUThreadCount();
          bool multi_thread_stop = (num_already_stopped > 0);
          if (is_first && !is_step_completion && !waiting_for_more &&
              !multi_thread_stop) {
            gpu_proc->ClearOldEUThreads();
          }
          lldb::tid_t tid =
              gpu_proc->HandleZeThreadStopped(*ds, ze_thread, is_first);
          if (is_first)
            first_stopped_tid = tid;
          any_thread_stopped = true;
          threads_stopped_this_call++;
          // Decrement expected count as we process each thread
          if (m_expected_stopped_count > 0)
            m_expected_stopped_count--;
          any_change = true;
        }
        break;
      }

      case ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE: {
        ze_device_thread_t ze_thread = event.info.thread.thread;
        bool was_wildcard = ZeThreadIsWildcard(ze_thread);
        gpu_proc->HandleZeThreadUnavailable(*ds, ze_thread);
        // Unavailable thread is no longer resumed: specific decrements
        // nresumed by 1; wildcard drives nresumed to 0 and decrements
        // ninterrupts (wildcard is the response to interrupt-all).
        if (DeviceSession *mds =
                gpu_proc->GetDeviceSessionMutable(qe.device_index)) {
          if (was_wildcard) {
            mds->nresumed = 0;
            if (mds->ninterrupts > 0)
              mds->ninterrupts--;
          } else if (mds->nresumed > 0) {
            mds->nresumed--;
          }
        }
        any_change = true;
        break;
      }

      case ZET_DEBUG_EVENT_TYPE_PAGE_FAULT:
        gpu_proc->HandleZePageFault(event.info.page_fault.address);
        any_change = true;
        break;

      default:
        break;
      }
      ZeAckEvent(ds->session, event);
    }
    return threads_stopped_this_call - stops_before;
  };

  // Normal drain of whatever the poll thread has queued so far.
  drain_once();

  // The poll thread is the SOLE reader of zetDebugReadEvent; do not call
  // it here or events (e.g. THREAD_STOPPED) can be silently discarded.

  // Gather barrier: on a breakpoint, siblings in the same workgroup are
  // not auto-stopped by the driver. Interrupt them and drain until all
  // have settled before halting; otherwise only the first EU's lanes
  // are reported. Skipped for single-step/trace stops.
  if (any_thread_stopped && m_expected_stopped_count == 0 &&
      !gpu_proc->IsAnyThreadSteppingCompleted()) {
    // pause_all(): interrupt every still-running EU thread on devices that
    // have resumed threads and no outstanding interrupt. The Level Zero driver
    // does NOT stop the sibling EU threads of a workgroup on its own.
    gpu_proc->PauseAll();

    // Drain events until nresumed == 0 on every device. The driver
    // answers the wildcard interrupt with a wildcard
    // THREAD_STOPPED/THREAD_UNAVAILABLE event that collapses all idle
    // threads. Deadline is a defensive backstop against a lost event.
    const uint64_t kBarrierDeadlineMs = elapsed_ms() + 2000;
    while (gpu_proc->GetResumedThreadCount() != 0 &&
           elapsed_ms() < kBarrierDeadlineMs) {
      if (drain_once() == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  // Process ALL queued THREAD_STOPPED events before halting; otherwise
  // remaining threads surface as spurious later stops. When
  // m_expected_stopped_count > 0, wait for that many stops first.
  if (any_thread_stopped) {
    // Only change the current thread if the currently selected thread is no
    // longer valid. This preserves the user's thread selection after stepping.
    lldb::tid_t current_tid = gpu_proc->GetCurrentThreadID();
    bool current_thread_just_stopped = false;
    if (current_tid != LLDB_INVALID_THREAD_ID) {
      NativeThreadProtocol *current_thread =
          gpu_proc->GetThreadByID(current_tid);
      if (current_thread)
        current_thread_just_stopped = true;
    }
    if (!current_thread_just_stopped) {
      gpu_proc->SetCurrentThreadID(first_stopped_tid);
    } else {
    }

    // Check if we need to wait for more threads before halting.
    // m_expected_stopped_count is decremented as we process each thread.
    // When it reaches 0, all expected threads have been processed.
    if (m_expected_stopped_count > 0) {
      // Don't halt yet - wait for more THREAD_STOPPED events.
      // When the expected count reaches 0, we'll halt on the next DrainZeEvents
      // call.
    } else if (threads_stopped_this_call > 0) {
      // All expected threads processed (m_expected_stopped_count reached 0)
      // OR no expectation (first breakpoint hit).

      // Trace stops: report the single stepped EU and wait for LLDB.
      // Breakpoint stops: report and auto-resume CPU so LLDB can drive
      // the first step command.
      bool is_trace_stop = gpu_proc->IsAnyThreadSteppingCompleted();

      // Normal stop - report it to LLDB
      gpu_proc->Halt();

      // Auto-resume CPU only for the initial breakpoint stop, once per
      // stop cluster. Trace stops leave CPU control to LLDB.
      if (!is_trace_stop && !m_cpu_halted_for_gpu_stop) {
        m_pending_gpu_stop = true;
        NativeProcessProtocol *cpu = GetNativeProcess();
        if (cpu) {
          cpu->Signal(SIGINT);
          m_cpu_halted_for_gpu_stop = true;
        }
      }
    }
  }

  // Re-arm the notifier pipe so the main loop can deliver more events.
  if (any_change)
    RearmNotifier();

  return any_change;
}
