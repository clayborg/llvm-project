//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H

#include "CUDADebuggerAPI.h"
#include "MainLoopEventNotifier.h"
#include "Plugins/Process/gdb-remote/LLDBServerPlugin.h"
#include "ProcessNVGPU.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <atomic>
#include <thread>

namespace lldb_private::lldb_server {

/// LLDB server plugin for NVIDIA GPU debugging support.
///
/// This effectively orchestrates the initialization of the NVIDIA debugger API
/// and the interaction between the CPU process, the GPU process and the
/// debugger API.
///
/// The debugger API comes up in one of two ways. When lldb launches the
/// application, or attaches before the application has initialized CUDA, the
/// initialization breakpoints bring it up in the call to cuInit that
/// initializes the driver:
///
/// 1. The breakpoint at cuInit's entry has lldb set one where cuInit returns.
/// 2. Before the driver initializes, at its CUDBG_PRE_INIT hook, or at cuInit's
///    entry when libcuda lacks the hook, the plug-in writes who is debugging
///    into libcuda, brings its side of the debugger API up and sets the IPC
///    flag.
/// 3. The driver initializes in the application's first call to cuInit, and
///    later calls do nothing. Initializing, it loads its debug engine, which
///    sets CUDBG_DEBUGGER_INITIALIZED, and once cuInit returns, the plug-in
///    brings the GPU up.
///
/// An application or its libraries can call cuInit again after lldb attached
/// to it. When such a call returns with CUDBG_DEBUGGER_INITIALIZED clear, the
/// driver had initialized before, so the plug-in undoes its setup. Then, as
/// when cuInit is not called again, the driver's attach procedure brings the
/// debugger API up:
///
/// 1. GetInitializeActions asks lldb for libcuda's symbols, which lldb sends
///    (SetRequestedSymbols) once it has loaded libcuda: as soon as it loads the
///    libraries of a process that it attaches to.
/// 2. The driver publishes its attach procedure descriptor only once the
///    application has a CUDA context, so TryLateAttach reads it again each time
///    the native process stops. Once it is valid, the plug-in writes who is
///    attaching into libcuda and a byte to the descriptor, which asks the
///    driver to load its debug engine at a point that it knows to be safe, and
///    has lldb set a breakpoint on CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED
///    before the application runs again.
/// 3. The driver loads its debug engine once the application runs, then calls
///    that function, whose breakpoint brings the debugger API up.
/// 4. The driver replays the contexts and modules that the application already
///    had, which needs the application to run, and reports
///    CUDBG_EVENT_ATTACH_COMPLETE. Once the attach has finished and the GPU
///    runs, the plug-in suspends the devices as an interrupt does, and reports
///    the stop. See AttachState.
class LLDBServerPluginNVGPU : public LLDBServerPlugin {
public:
  /// Constructor for the NVIDIA GPU server plugin.
  ///
  /// \param[in] native_process
  ///     Reference to the GDB server managing the native process.
  ///
  /// \param[in] main_loop
  ///     Reference to the main event loop for handling asynchronous events.
  LLDBServerPluginNVGPU(LLDBServerPlugin::GDBServer &native_process,
                        MainLoop &main_loop);

  /// Stop the GPU MainLoop thread and finalize the debugger API before any
  /// member is destroyed, since that thread and the driver's event callback
  /// both use this object.
  ~LLDBServerPluginNVGPU() override;

  /// Get the name identifier for this plugin.
  ///
  /// \return
  ///     String reference containing the plugin name.
  llvm::StringRef GetPluginName() override;

  /// Get the initialization actions required for this plugin.
  ///
  /// \return
  ///     GPUActions structure containing the initialization steps.
  GPUActions GetInitializeActions() override;

  /// Handle breakpoint hit events from the GPU.
  ///
  /// Processes breakpoint events and determines the appropriate response
  /// action for the debugger.
  ///
  /// \param[in] args
  ///     Arguments containing details about the breakpoint hit.
  ///
  /// \return
  ///     Expected response indicating the action to take, or error if
  ///     the breakpoint could not be processed.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  BreakpointWasHit(GPUPluginBreakpointHitArgs &args) override;

  /// Ask the CUDA driver to attach if SetRequestedSymbols found that it could
  /// not attach yet, and it can now. The driver can attach only once the
  /// application has a CUDA context, which it can create long after cuInit,
  /// which initializes the driver, has run.
  ///
  /// \return
  ///     Breakpoints on the function that the driver calls once it has
  ///     attached, if this asked it to attach.
  std::optional<GPUActions> NativeProcessIsStopping() override;

  void NativeProcessDidExit(const WaitStatus &exit_status) override;

  /// End the GPU MainLoop thread's loop.
  void ClientDidDisconnect() override;

  /// Ask the CUDA driver to attach once libcuda's symbols resolve, if it can,
  /// as it can once the application has a CUDA context. Otherwise the
  /// initialization breakpoints bring the GPU up, as on launch, or
  /// NativeProcessIsStopping asks again.
  ///
  /// \return
  ///     Breakpoints on the function that the driver calls once it has
  ///     attached, if this asked it to attach.
  std::optional<GPUActions>
  SetRequestedSymbols(const std::vector<SymbolValue> &symbols) override;

private:
  /// Create a connection to the GPU process that the client can use.
  ///
  /// Establishes a communication channel between the debugger client and
  /// the GPU debugging infrastructure.
  ///
  /// \param[in] before_accept
  ///     Called once the socket is listening and before the thread that
  ///     accepts the client starts, which waits for a client until the
  ///     plug-in shuts down. A step that can fail after the socket exists
  ///     belongs here. If it returns an error, no thread is started and the
  ///     socket is closed.
  ///
  /// \return
  ///     Expected connection info on success, or error on failure.
  llvm::Expected<GPUPluginConnectionInfo>
  CreateConnection(llvm::function_ref<llvm::Error()> before_accept);

  /// Function used to execute the main loop of the GPU process in an
  /// independent thread.
  ///
  /// This method runs the GPU process event handling loop in a separate
  /// thread to avoid blocking the main debugger execution.
  ///
  /// \param[in] listen_socket_up
  ///     Unique pointer to TCP socket for listening to client connections.
  void AcceptAndMainLoopThread(std::unique_ptr<TCPSocket> listen_socket_up);

  /// Process debugger API events.
  ///
  /// Handles events from the CUDA debugger API, processing them and
  /// taking appropriate action based on event type.
  void OnDebuggerAPIEvent();

  /// \return
  ///     The GPU process, or null once the GPU server has forgotten it, which
  ///     it does on the GPU MainLoop thread when the GPU process exits or lldb
  ///     detaches from it. Other threads may call this only before that thread
  ///     starts or after it ends.
  ProcessNVGPU *GetGPUProcess() const;

  /// Stop using the debugger API after it reports an internal error, a call
  /// into it fails, or the device state it reports cannot be decoded, and stop
  /// the GPU process with \a description. lldb also gets \a description as an
  /// error if it is waiting for a stop reply, and the native process stops too
  /// unless NVGPU_DISABLE_CPU_STOP_ON_GPU_STOP=1.
  ///
  /// \param[in] description
  ///     What went wrong, which the stop shows.
  void ReportDebuggerAPIFault(std::string description);

  /// Drain the event queue until it is empty or a stop is reported, then
  /// acknowledge once, returning how many events were handled. One
  /// notification can cover several events, so draining one at a time would
  /// strand the rest. While the GPU process is stopped, nothing is drained,
  /// for the reasons in the class comment of ProcessNVGPU.
  llvm::Expected<int> DrainSyncEvents();

  /// Stop the GPU if a late attach has finished and the GPU process runs, or
  /// return the error to report as a fault if the driver attached without
  /// replaying anything: see AttachState. This follows each drain, and the
  /// drain runs whenever the GPU process resumes, which an attach that
  /// replayed nothing needs, since it leaves no event to drain.
  llvm::Error StopIfAttachFinished();

  /// Finalize and drop the debugger API along with everything holding a
  /// pointer into it, on destruction, and to undo a setup that failed partway
  /// or that the driver did not pick up.
  void ReleaseDebuggerAPI();

  /// Let go of a debugger API that lldb never connected to serve, without
  /// finalizing it. The driver keeps the application waiting until the events
  /// it raised are acknowledged, and a finalize meanwhile can crash it, as can
  /// one just after an acknowledgment lets the application go on. The
  /// application goes on by itself once lldb-server exits.
  void ReleaseUnservedDebuggerAPI();

  /// Take from the GPU process, if the GPU server still has it, everything
  /// that points into the debugger API or back at this object: the API
  /// table, the event drain notifier and the fault handler.
  void UnhookGPUProcess();

  /// Bring the debugger API up and hook it to the GPU process, without telling
  /// the driver, from a breakpoint hit, which runs on the native server's
  /// thread, the only one that may use \a native_process.
  llvm::Error SetUpDebuggerAPI(SymbolAddressProvider get_symbol_address,
                               llvm::StringRef libcuda_library_name,
                               NativeProcessProtocol &native_process);

  /// Create the connection that lldb debugs the GPU through, once the debugger
  /// API is set up.
  ///
  /// \param[in] before_accept
  ///     See CreateConnection.
  ///
  /// \return
  ///     The actions that have lldb connect.
  llvm::Expected<GPUActions>
  Connect(llvm::function_ref<llvm::Error()> before_accept);

  /// Bring the debugger API up at the breakpoint that ends a late attach, and
  /// create the connection.
  llvm::Expected<GPUActions>
  InitializeAPIForLateAttach(SymbolAddressProvider get_symbol_address,
                             llvm::StringRef libcuda_library_name,
                             NativeProcessProtocol &native_process);

  /// Handle the breakpoint at cuInit's entry: have lldb set a breakpoint where
  /// cuInit returns, and set the debugger API up now if libcuda lacks the
  /// CUDBG_PRE_INIT hook. See the class comment.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  CuInitWasCalled(const GPUPluginBreakpointHitArgs &args,
                  llvm::StringRef libcuda_library_name,
                  NativeProcessProtocol &native_process);

  /// Handle the breakpoint on CUDBG_PRE_INIT, which the call to cuInit that
  /// m_pending_cuinit follows makes before the driver initializes.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  PreInitWasCalled(const GPUPluginBreakpointHitArgs &args,
                   NativeProcessProtocol &native_process);

  /// Handle the breakpoint where the call to cuInit that m_pending_cuinit
  /// follows returns: bring the GPU up if the driver picked up the debugger
  /// API, and undo the setup otherwise.
  llvm::Expected<GPUPluginBreakpointHitResponse>
  CuInitReturned(const GPUPluginBreakpointHitArgs &args,
                 NativeProcessProtocol &native_process);

  /// Set the debugger API up for the call to cuInit that m_pending_cuinit
  /// follows, and set the IPC flag, which the driver picks up if it
  /// initializes in that call.
  llvm::Error SetUpForCuInit(SymbolAddressProvider get_symbol_address,
                             NativeProcessProtocol &native_process);

  /// Ask the driver to attach with the symbols in m_late_attach_symbols, if it
  /// can, on the native server's thread while the native process is stopped.
  ///
  /// \return
  ///     Breakpoints on the function that the driver calls once it has
  ///     attached, if this asked it to attach.
  std::optional<GPUActions> TryLateAttach();

  /// Tell the driver, once the debugger API is up at the end of a late attach,
  /// that this debugger serves it, and find out whether it has state to
  /// replay. See AttachState.
  llvm::Error FinishLateAttach(SymbolAddressProvider get_symbol_address,
                               NativeProcessProtocol &native_process);

  Status m_main_loop_status;

  /// Runs AcceptAndMainLoopThread. Only used on the native server's thread,
  /// which creates the connection and destroys this object.
  std::thread m_gpu_thread;

  /// Set by the destructor so that the GPU thread stops waiting for the client
  /// to connect.
  std::atomic<bool> m_shutting_down = false;

  std::optional<CUDADebuggerAPI> m_cuda_api;
  /// A utility to send debugger api notifications to the main loop.
  std::unique_ptr<MainLoopEventNotifier> m_main_loop_event_notifier_up;

  /// Set once the debugger API is up. Only breakpoint hits and
  /// SetRequestedSymbols use it, and they arrive on the native server's
  /// thread.
  bool m_api_initialized = false;

  /// Whether SetRequestedSymbols has received libcuda's symbols. Only used on
  /// the native server's thread, as is m_late_attach_symbols.
  bool m_libcuda_symbols_received = false;

  /// libcuda's symbols, while the driver cannot attach yet and nothing else has
  /// brought the debugger API up. See NativeProcessIsStopping.
  std::vector<SymbolValue> m_late_attach_symbols;

  /// Whether TryLateAttach has asked the driver to attach, after which only
  /// the breakpoint that ends the attach brings the debugger API up. Only used
  /// on the native server's thread.
  bool m_late_attach_requested = false;

  /// A call to cuInit that the initialization breakpoints follow, from its
  /// entry until it returns. See the class comment.
  struct PendingCuInit {
    /// The thread that called cuInit.
    lldb::tid_t tid;
    /// Where cuInit returns to.
    lldb::addr_t return_address;
    /// The libcuda that cuInit is in.
    std::string libcuda_library_name;
    /// The symbols of the breakpoint at cuInit's entry.
    std::vector<SymbolValue> symbols;
    /// Whether the debugger API is set up for this call.
    bool set_up = false;
  };
  /// Only used on the native server's thread.
  std::optional<PendingCuInit> m_pending_cuinit;

  /// Set once a call to cuInit returned without the driver picking up the
  /// debugger API that the plug-in set up for it, or found the debug engine of
  /// an earlier debugger initialized already, either of which shows that the
  /// driver initialized before lldb attached. From then on, only the driver's
  /// attach procedure brings the debugger API up. Only used on the native
  /// server's thread.
  bool m_driver_initialized_before_attach = false;

  /// Where a late attach stands, once the driver has attached and the
  /// debugger API is up.
  enum class AttachState {
    /// No late attach is under way: lldb launched the application, or the GPU
    /// has stopped since the attach finished.
    None,
    /// The driver replays the contexts and modules that the application
    /// created before the attach, which needs the application to run, and
    /// reports CUDBG_EVENT_ATTACH_COMPLETE when it is done.
    Replaying,
    /// The attach has finished. Once the GPU runs, it stops, so that lldb
    /// shows the threads of the kernels that were already running, unless a
    /// stop with its threads comes first.
    Finished,
    /// The driver attached without replaying anything, although it is asked
    /// to attach only once the application has a CUDA context. It does so
    /// when the application loads modules lazily, since lldb-server does not
    /// request the capabilities for that. Once the GPU runs, the plug-in
    /// reports a fault that says how to start the application instead.
    Unreported,
  };
  /// Set on the native server's thread before the GPU MainLoop thread starts,
  /// and used only on that thread afterwards.
  AttachState m_attach_state = AttachState::None;

  /// Set on the native server's thread when the native process exits, before
  /// the GPU MainLoop thread reports the exit to lldb. From then on, the
  /// debugger API's events are left alone: acknowledging the events of an
  /// application that has exited can crash the debugger library.
  std::atomic<bool> m_native_process_exited = false;
};

} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINNVGPU_H
