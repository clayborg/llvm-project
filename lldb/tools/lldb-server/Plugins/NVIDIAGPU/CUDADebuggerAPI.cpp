//===-- CUDADebuggerAPI.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDADebuggerAPI.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "Utils.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/Support/Error.h"

#include <dlfcn.h>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

#define STRINGIFY_SYMBOL_HELPER(x) #x
#define STRINGIFY_SYMBOL(x) STRINGIFY_SYMBOL_HELPER(x)

namespace Symbols {
static std::string CUDBG_IPC_FLAG_NAME = STRINGIFY_SYMBOL(CUDBG_IPC_FLAG_NAME);
static std::string CUDBG_APICLIENT_PID = STRINGIFY_SYMBOL(CUDBG_APICLIENT_PID);
static std::string CUDBG_APICLIENT_REVISION =
    STRINGIFY_SYMBOL(CUDBG_APICLIENT_REVISION);
static std::string CUDBG_SESSION_ID = STRINGIFY_SYMBOL(CUDBG_SESSION_ID);
static std::string CUDBG_DEBUGGER_CAPABILITIES =
    STRINGIFY_SYMBOL(CUDBG_DEBUGGER_CAPABILITIES);
static std::string CUDBG_INJECTION_PATH = "cudbgInjectionPath";
static std::string CUDBG_GET_API = "cudbgGetAPI";
static std::string CUDBG_CUINIT = "cuInit";
} // namespace Symbols

namespace lldb_private::lldb_server {
void CUDBGAPIDeleter(CUDBGAPI api) {
  if (!api)
    return;

  CUDBGResult res = api->finalize();
  if (res != CUDBG_SUCCESS) {
    Log *log = GetLog(GDBRLog::Plugin);
    LLDB_LOG(log, "Failed to finalize the CUDA Debugger API. {0}",
             cudbgGetErrorString(res));
  }
}
} // namespace lldb_private::lldb_server

static Error VerifyDebuggerCapabilities(CUDADebuggerAPI &api) {
  CUDBGCapabilityFlags supported_capabilities;
  CUDBGResult res =
      api->getSupportedDebuggerCapabilities(&supported_capabilities);
  if (res != CUDBG_SUCCESS)
    return createStringError(
        "Failed to get the GPU debugger supported capabilities. {0}",
        cudbgGetErrorString(res));
  if (!(supported_capabilities & CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS))
    return createStringError(
        "The GPU debugger does not support suspend events");
  if (!(supported_capabilities &
        CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS))
    return createStringError(
        "The GPU debugger does not support skipping context "
        "push/pop events");
  return Error::success();
}

static Error WriteToHostSymbol(const GPUPluginBreakpointHitArgs &bp_args,
                               NativeProcessProtocol &linux_process,
                               llvm::StringRef symbol_name,
                               const uint32_t &value) {
  std::optional<uint64_t> symbol_address = bp_args.GetSymbolValue(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                symbol_name);
  size_t bytes_written = 0;
  linux_process.WriteMemory(*symbol_address, &value, sizeof(value),
                            bytes_written);
  if (bytes_written != sizeof(value))
    return createStringErrorFmt("Failed to write symbol {0}", symbol_name);
  return Error::success();
}

static Error WriteToHostSymbol(const GPUPluginBreakpointHitArgs &bp_args,
                               NativeProcessProtocol &linux_process,
                               llvm::StringRef symbol_name, const char *value) {
  std::optional<uint64_t> symbol_address = bp_args.GetSymbolValue(symbol_name);
  if (!symbol_address)
    return createStringErrorFmt("Couldn't find address for symbol {0}",
                                symbol_name);
  size_t bytes_written = 0;
  size_t value_size = strlen(value) + 1;
  linux_process.WriteMemory(*symbol_address, value, value_size, bytes_written);
  if (bytes_written != value_size)
    return createStringErrorFmt("Failed to write symbol {0}", symbol_name);
  return Error::success();
}

static Error
WriteInitializationSymbolsToHost(const GPUPluginBreakpointHitArgs &bp_args,
                                 NativeProcessProtocol &linux_process,
                                 uint32_t pid, uint32_t session_id,
                                 uint32_t revision) {
  auto write_uint32_t = [&](const std::string &symbol_name,
                            const uint32_t &value) -> Error {
    return WriteToHostSymbol(bp_args, linux_process, symbol_name, value);
  };

  const uint32_t ipc_flag = 1;
  if (Error err = write_uint32_t(Symbols::CUDBG_IPC_FLAG_NAME, ipc_flag))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_PID, pid))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_APICLIENT_REVISION, revision))
    return err;

  if (Error err = write_uint32_t(Symbols::CUDBG_SESSION_ID, session_id))
    return err;

  const uint32_t capabilities =
      CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS |
      CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;
  if (Error err =
          write_uint32_t(Symbols::CUDBG_DEBUGGER_CAPABILITIES, capabilities))
    return err;

  return Error::success();
}

static Error WriteConfigurationToLibcuda(void *libcuda, uint32_t pid,
                                         uint32_t revision,
                                         uint32_t session_id) {
  auto *api_client_pid = reinterpret_cast<uint32_t *>(
      dlsym(libcuda, Symbols::CUDBG_APICLIENT_PID.c_str()));
  if (!api_client_pid)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_APICLIENT_PID,
                                CUDADebuggerAPI::CUDA_API_LIBRARY_NAME);

  auto *api_client_revision = reinterpret_cast<uint32_t *>(
      dlsym(libcuda, Symbols::CUDBG_APICLIENT_REVISION.c_str()));
  if (!api_client_revision)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_APICLIENT_REVISION,
                                CUDADebuggerAPI::CUDA_API_LIBRARY_NAME);

  auto *session_id_ptr = reinterpret_cast<uint32_t *>(
      dlsym(libcuda, Symbols::CUDBG_SESSION_ID.c_str()));
  if (!session_id_ptr)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_SESSION_ID,
                                CUDADebuggerAPI::CUDA_API_LIBRARY_NAME);

  *api_client_pid = pid;
  *api_client_revision = revision;
  *session_id_ptr = session_id;

  return Error::success();
}

static Error
WriteInjectionPathToLibcuda(const GPUPluginBreakpointHitArgs &bp_args,
                            NativeProcessProtocol &linux_process,
                            void *libcuda) {
  auto write_c_str = [&](const std::string &symbol_name,
                         const char *value) -> Error {
    return WriteToHostSymbol(bp_args, linux_process, symbol_name, value);
  };

  if (const auto *path = getenv("CUDBG_INJECTION_PATH")) {
    if (Error err = write_c_str(Symbols::CUDBG_INJECTION_PATH, path))
      return err;

    char *injection_path = reinterpret_cast<char *>(
        dlsym(libcuda, Symbols::CUDBG_INJECTION_PATH.c_str()));
    if (!injection_path)
      return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                  Symbols::CUDBG_INJECTION_PATH,
                                  CUDADebuggerAPI::CUDA_API_LIBRARY_NAME);
    strcpy(injection_path, path);
  }
  return Error::success();
}

static Expected<CUDBGAPI> GetRawAPIInstance(void *libcuda) {
  using CudbgGetAPIFn =
      CUDBGResult (*)(uint32_t, uint32_t, uint32_t, CUDBGAPI *);
  const auto cudbgGetAPI = reinterpret_cast<CudbgGetAPIFn>(
      dlsym(libcuda, Symbols::CUDBG_GET_API.c_str()));
  if (!cudbgGetAPI)
    return createStringErrorFmt("Failed to find symbol {0} in {1}",
                                Symbols::CUDBG_GET_API,
                                CUDADebuggerAPI::CUDA_API_LIBRARY_NAME);

  CUDBGAPI api;
  CUDBGResult res =
      cudbgGetAPI(CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
                  CUDBG_API_VERSION_REVISION, &api);
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("The `cudbgGetAPI` call failed. {0}",
                                cudbgGetErrorString(res));

  return api;
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::InitializeImpl(const GPUPluginBreakpointHitArgs &bp_args,
                                NativeProcessProtocol &linux_process) {
  Log *log = GetLog(GDBRLog::Plugin);
  LLDB_LOG(log, "CUDADebuggerAPI::Initialize()");

  const uint32_t pid = getpid();
  const uint32_t session_id = 0;
  const uint32_t revision = CUDBG_API_VERSION_REVISION;

  if (Error err = WriteInitializationSymbolsToHost(bp_args, linux_process, pid,
                                                   session_id, revision))
    return err;

  void *libcuda = dlopen(CUDA_API_LIBRARY_NAME, RTLD_LAZY);
  if (!libcuda)
    return createStringErrorFmt("Failed to dlopen {0}", CUDA_API_LIBRARY_NAME);

  if (Error err =
          WriteConfigurationToLibcuda(libcuda, pid, revision, session_id))
    return err;

  if (Error err = WriteInjectionPathToLibcuda(bp_args, linux_process, libcuda))
    return err;

  Expected<CUDBGAPI> api_or = GetRawAPIInstance(libcuda);
  if (!api_or)
    return api_or.takeError();

  CUDADebuggerAPI api(*api_or);

  CUDBGResult res = api->initialize();
  if (res != CUDBG_SUCCESS)
    return createStringErrorFmt("The `CUDBGAPI.initialize` call failed. {0}",
                                cudbgGetErrorString(res));

  if (Error err = VerifyDebuggerCapabilities(api))
    return err;

  return api;
}

Expected<CUDADebuggerAPI>
CUDADebuggerAPI::Initialize(const GPUPluginBreakpointHitArgs &args,
                            NativeProcessProtocol &linux_process) {
  Expected<CUDADebuggerAPI> api = InitializeImpl(args, linux_process);
  if (!api)
    return createStringErrorFmt(
        "Failed to initialize the CUDA Debugger API. {0}",
        llvm::toString(api.takeError()));
  return api;
}

GPUBreakpointInfo CUDADebuggerAPI::GetInitializationBreakpointInfo() {
  GPUBreakpointInfo bp;
  bp.name_info = {CUDA_API_LIBRARY_NAME, Symbols::CUDBG_CUINIT};
  bp.symbol_names.push_back(Symbols::CUDBG_IPC_FLAG_NAME);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_PID);
  bp.symbol_names.push_back(Symbols::CUDBG_APICLIENT_REVISION);
  bp.symbol_names.push_back(Symbols::CUDBG_SESSION_ID);
  bp.symbol_names.push_back(Symbols::CUDBG_DEBUGGER_CAPABILITIES);
  return bp;
}