/*
 * Copyright 2007-2025 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*---------------------------- Files Information -----------------------------*/

/**
 * \file cudadebugger.h
 * \brief Header file for the CUDA Debugger API.
 */


/*------------------------------ Section Order -------------------------------*/

/** \defgroup GENERAL General */
/** \defgroup INIT    Initialization */
/** \defgroup EXEC    Device Execution Control */
/** \defgroup BP      Breakpoints */
/** \defgroup READ    Device State Inspection*/
/** \defgroup WRITE   Device State Alteration */
/** \defgroup GRID    Grid Properties */
/** \defgroup DEV     Device Properties */
/** \defgroup DWARF   DWARF Utilities */


/*------------------------------- Events -------------------------------------*/

/** \defgroup EVENT Events

One of those events will create a CUDBGEvent:
\arg the elf image of the current kernel has been loaded and the
     addresses within its DWARF sections have been relocated (and can
     now be used to set breakpoints),
\arg a device breakpoint has been hit.

When a CUDBGEvent is created, the debugger is notified by calling the
callback functions registered with setNotifyNewEventCallback() after
the API struct initialization. It is up to the debugger to decide what
method is best to be notified. The debugger API routines cannot be
called from within the callback function or the routine will return an
error.

Upon notification the debugger is responsible for handling the
CUDBGEvents in the event queue by using CUDBGAPI_st::getNextEvent(), and for
acknowledging the debugger API that the event has been handled by
calling CUDBGAPI_st::acknowledgeEvent(). In the case of an event raised by the
device itself, such as a breakpoint being hit, the event queue will
be empty. It is the responsibility of the debugger to inspect the
hardware any time a CUDBGEvent is received.


Example:
\code
CUDBGEvent event;
CUDBGResult res;
for (res = cudbgAPI->getNextEvent(&event);
     res == CUDBG_SUCCESS && event.kind != CUDBG_EVENT_INVALID;
     res = cudbgAPI->getNextEvent(&event)) {
    switch (event.kind)
        {
        case CUDBG_EVENT_ELF_IMAGE_LOADED:
            //...
            break;
        default:
            error(...);
        }
    }
\endcode

See cuda-tdep.c and cuda-linux-nat.c files in the cuda-gdb source code
for a more detailed example on how to use CUDBGEvent.

*/


/*-------------------------------- Main Page ---------------------------------*/

/** \mainpage Introduction

This document describes the API for the set routines and data
structures available in the CUDA library to any debugger.

Starting with 3.0, the CUDA debugger API includes several major changes, of
which only few are directly visible to end-users:
\arg Performance is greatly improved, both with respect to
     interactions with the debugger and the performance of
     applications being debugged.
\arg The format of cubins has changed to ELF and, as a consequence,
     most restrictions on debug compilations have been lifted. More
     information about the new object format is included below.

The debugger API has significantly changed, reflected in the CUDA-GDB
sources.

\section API Debugger API

The CUDA Debugger API was developed with the goal of adhering to the following
principles:

\arg Policy free
\arg Explicit
\arg Axiomatic
\arg Extensible
\arg Machine oriented

Being explicit is another way of saying that we minimize the
assumptions we make. As much as possible the API reflects machine
state, not internal state.

There are two major "modes" of the devices: stopped or running. We
switch between these modes explicitly with suspendDevice and
resumeDevice, though the machine may suspend on its own accord, for
example when hitting a breakpoint.

Only when stopped, can we query the machine's state. Warp state
includes which function is it running, which block, which lanes are
valid, etc.

\section ELF ELF and DWARF

CUDA applications are compiled in ELF binary format.

DWARF device information is obtained through a CUDBGEvent of type
CUDBG_EVENT_ELF_IMAGE_LOADED. This means that the information is not available
until runtime, after the CUDA driver has loaded.

DWARF device information contains physical addresses for all
device memory regions except for code memory.  The address class field
(DW_AT_address_class) is set for all device variables, and is used to
indicate the memory segment type (ptxStorageKind).  The physical addresses must be
accessed using several segment-specific API calls:

For memory reads, see:
\arg CUDBGAPI_st::readCodeMemory()
\arg CUDBGAPI_st::readConstMemory()
\arg CUDBGAPI_st::readGenericMemory()
\arg CUDBGAPI_st::readParamMemory()
\arg CUDBGAPI_st::readSharedMemory()
\arg CUDBGAPI_st::readLocalMemory()
\arg CUDBGAPI_st::readTextureMemory()
\arg CUDBGAPI_st::readGlobalMemory()

For memory writes, see:
\arg CUDBGAPI_st::writeGenericMemory()
\arg CUDBGAPI_st::writeParamMemory()
\arg CUDBGAPI_st::writeSharedMemory()
\arg CUDBGAPI_st::writeLocalMemory()
\arg CUDBGAPI_st::writeGlobalMemory()

Access to code memory requires a virtual address. This virtual address is
embedded for all device code sections in the device ELF image. See the API
call:
\arg CUDBGAPI_st::readVirtualPC()

Here is a typical DWARF entry for a device variable located in memory:

\code
<2><321>: Abbrev Number: 18 (DW_TAG_formal_parameter)
     DW_AT_decl_file   : 27
     DW_AT_decl_line   : 5
     DW_AT_name        : res
     DW_AT_type        : <2c6>
     DW_AT_location    : 9 byte block: 3 18 0 0 0 0 0 0 0       (DW_OP_addr: 18)
     DW_AT_address_class: 7
\endcode

The above shows that variable 'res' has an address class of 7
(ptxParamStorage). Its location information shows it is located at
address 18 within the parameter memory segment.

Local variables are no longer spilled to local memory by default. The
DWARF now contains variable-to-register mapping and liveness
information for all variables.  It can be the case that variables are
spilled to local memory, and this is all contained in the DWARF
information which is ULEB128 encoded (as a DW_OP_regx stack operation
in the DW_AT_location attribute).

Here is a typical DWARF entry for a variable located in a local
register:

\code
 <3><359>: Abbrev Number: 20 (DW_TAG_variable)
     DW_AT_decl_file   : 27
     DW_AT_decl_line   : 7
     DW_AT_name        : c
     DW_AT_type        : <1aa>
     DW_AT_location    : 7 byte block: 90 b9 e2 90 b3 d6 4      (DW_OP_regx: 160631632185)
     DW_AT_address_class: 2
\endcode

This shows variable 'c' has address class 2 (ptxRegStorage) and its location
can be found by decoding the ULEB128 value, DW_OP_regx: 160631632185.
See cuda-tdep.c in the cuda-gdb source drop for information on decoding
this value and how to obtain which physical register holds this variable
during a specific device PC range. Access to physical registers liveness
information requires a 0-based physical PC. See the API call:
\arg CUDBGAPI_st::readPC()

\section abi31 ABI Support

ABI support is handled through the following thread API calls.
\arg CUDBGAPI_st::readCallDepth()
\arg CUDBGAPI_st::readReturnAddress()
\arg CUDBGAPI_st::readVirtualReturnAddress()

The return address is not accessible on the local
stack and the API call must be used to access its value.

For more information, please refer to the ABI documentation titled "Fermi
ABI: Application Binary Interface".

\section exceptions31 Exception Reporting

Some kernel exceptions are reported as device events and accessible via the API
call:
\arg CUDBGAPI_st::readLaneException()

The reported exceptions are listed in the CUDBGException_t enum type.
Each prefix, (Device, Warp, Lane), refers to the precision of the exception.
That is, the lowest known execution unit that is responsible for the origin of
the exception. All lane errors are precise; the exact instruction and lane that
caused the error are known. Warp errors are typically within a few instructions
of where the actual error occurred, but the exact lane within the warp is not
known. On device errors, we _may_ know the _kernel_ that caused it.
Explanations about each exception type can be found in the documentation of the
struct.

Exception reporting is only supported on Fermi (sm_20 or greater).

*/


/*------------------------------- Includes -----------------------------------*/

#ifndef CUDADEBUGGER_H
#define CUDADEBUGGER_H

#include <stdint.h>
#include <stdlib.h>

#if defined(_MSC_VER) && _MSC_VER < 1800
// old MSVC does not support stdbool.h
typedef unsigned char bool;
#undef false
#undef true
#define false 0
#define true  1
#else
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* OS-agnostic _CUDBG_INLINE */
#if defined(_WIN32)
#define _CUDBG_INLINE __inline
#else
#define _CUDBG_INLINE inline
#endif


/*--------------------------------- API Version ------------------------------*/

/**
 * \brief Major release version number.
 * This matches the major version of the CUDA driver exposing this API.
 * \ingroup GENERAL
 */
#define CUDBG_API_VERSION_MAJOR    13
/**
 * \brief Minor release version number.
 * This matches the minor version of the CUDA driver exposing this API.
 * \ingroup GENERAL
 */
#define CUDBG_API_VERSION_MINOR    4
/**
 * \brief API revision number.
 * This number is incremented every time changes are made to the API.
 * \ingroup GENERAL
 */
#define CUDBG_API_VERSION_REVISION 193


/*---------------------------------- Constants -------------------------------*/

/**
 * \brief Maximum number of supported devices.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_DEVICES       64
/**
 * \brief Maximum number of SMs per device.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_SMS           256
/**
 * \brief Maximum number of warps per SM.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_WARPS         64
/**
 * \brief Maximum number of lanes per warp.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_LANES         32
/**
 * \brief Maximum number of convergence barriers per warp.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_WARP_BARRIERS 16
/**
 * \brief Maximum length of a single CUDA log message.
 * \ingroup GENERAL
 */
#define CUDBG_MAX_LOG_LEN       256


/*----------------------- Thread/Block Coordinates Types ---------------------*/

/**
 * \brief 2-dimensional coordinates for threads, blocks, etc.
 * \note DEPRECATED: Use CuDim3 instead.
 * \ingroup GENERAL
 */
typedef struct {
    /** \brief X coordinate. */
    uint32_t x;
    /** \brief Y coordinate. */
    uint32_t y;
} CuDim2;
/**
 * \brief 3-dimensional coordinates for threads, blocks, etc.
 * \ingroup GENERAL
 */
typedef struct {
    /** \brief X coordinate. */
    uint32_t x;
    /** \brief Y coordinate. */
    uint32_t y;
    /** \brief Z coordinate. */
    uint32_t z;
} CuDim3;


/*--------------------- Memory Segments (as used in DWARF) -------------------*/

/**
 * \brief Memory segments for DWARF.
 * \note DEPRECATED: This enum is no longer used since the API methods that use it have been
 * deprecated.
 * \ingroup READ
 */
typedef enum {
    ptxUNSPECIFIEDStorage,
    ptxCodeStorage,
    ptxRegStorage,
    ptxSregStorage,
    ptxConstStorage,
    ptxGlobalStorage,
    ptxLocalStorage,
    ptxParamStorage,
    ptxSharedStorage,
    ptxSurfStorage,
    ptxTexStorage,
    ptxTexSamplerStorage,
    ptxGenericStorage,
    ptxIParamStorage,
    ptxOParamStorage,
    ptxFrameStorage,
    ptxURegStorage,
    ptxMAXStorage
} ptxStorageKind;


/*--------------------------- Debugger System Calls --------------------------*/

/**
 * \brief Name of the global variable containing the IPC flag.
 * This variable is set by the API client to indicate that it is ready to attach to a running CUDA
 * application.
 * \ingroup GENERAL
 */
#define CUDBG_IPC_FLAG_NAME            cudbgIpcFlag
/**
 * \brief Name of the global variable containing the RPC enabled flag.
 * This variable is only used by debuggers that use the RPCD interface (e.g. CUDA-GDB).
 * \ingroup GENERAL
 */
#define CUDBG_RPC_ENABLED              cudbgRpcEnabled
/**
 * \brief Name of the global variable containing the API client PID.
 * This variable is set by the API client to identify itself.
 * \ingroup GENERAL
 */
#define CUDBG_APICLIENT_PID            cudbgApiClientPid
/**
 * \brief Name of the global variable containing the debug engine initialized flag.
 * This variable is set by the debug engine to indicate that it has been initialized.
 * \ingroup GENERAL
 */
#define CUDBG_DEBUGGER_INITIALIZED     cudbgDebuggerInitialized
/**
 * \brief Name of the global variable containing the API client revision.
 * This variable is set by the API client to identify the API revision it wants to use.
 * \ingroup GENERAL
 */
#define CUDBG_APICLIENT_REVISION       cudbgApiClientRevision
/**
 * \brief Name of the global variable containing the session ID.
 * This variable is set by the API client to identify the session.
 * \ingroup GENERAL
 */
#define CUDBG_SESSION_ID               cudbgSessionId
/**
 * \brief Name of the global variable containing the attach handler available flag.
 * This variable is set by the debug engine to indicate that the attach handler is available.
 * \ingroup GENERAL
 */
#define CUDBG_ATTACH_HANDLER_AVAILABLE cudbgAttachHandlerAvailable
/**
 * \brief Name of the global variable containing the enable launch blocking flag.
 * This variable is set by the API client to enable blocking launches of CUDA kernels.
 * \ingroup GENERAL
 */
#define CUDBG_ENABLE_LAUNCH_BLOCKING   cudbgEnableLaunchBlocking
/**
 * \brief Name of the global variable containing the resume for attach detach flag.
 * This variable is set by the debug engine to indicate that the API client should resume the CUDA
 * application (including the CPU) to complete the attach or detach procedure.
 * \ingroup GENERAL
 */
#define CUDBG_RESUME_FOR_ATTACH_DETACH cudbgResumeForAttachDetach

/**
 * \brief Name of the global variable containing the debug engine capabilities bitmask.
 * This variable is set by the API client to indicate the requested capabilities.
 * See the CUDBGCapabilityFlags enum for the list of available capabilities.
 * \ingroup GENERAL
 */
#define CUDBG_DEBUGGER_CAPABILITIES cudbgDebuggerCapabilities

/**
 * \brief Name of the global variable containing the external debug engine in use flag.
 * Can be read to detect whether the external debug engine implementation (libcudadebugger.so) is
 * used or not.
 * \note Since CUDA 13.1, the external debug engine implementation is always used.
 * \ingroup GENERAL
 */
#define CUDBG_USE_EXTERNAL_DEBUGGER cudbgUseExternalDebugger

/* The following global variables are no longer supported:
 * #define CUDBG_DETACH_SUSPENDED_DEVICES_MASK cudbgDetachSuspendedDevicesMask
 * #define CUDBG_ENABLE_INTEGRATED_MEMCHECK    cudbgEnableIntegratedMemcheck
 * #define CUDBG_ENABLE_PREEMPTION_DEBUGGING   cudbgEnablePreemptionDebugging
 */

/**
 * \brief Name of the global variable containing the debug engine attach procedure initiation FD.
 * If this FD is not -1, write a single byte (any value) to it to initiate the CUDA debug engine
 * attach procedure. After the request was received and the attach procedure has finished, the
 * CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED function is called.
 * \ingroup GENERAL
 */
#define CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD cudbgInitiateDebuggerAttachProcedureFd

/**
 * \brief Debug engine capability flags.
 * Clients should request the capabilities they want by setting the CUDBG_DEBUGGER_CAPABILITIES
 * global variable and then checking the supported capabilities by calling the
 * getSupportedDebuggerCapabilities() API method and adjusting their behavior accordingly.
 * Capabilities requested but not supported by the debug engine will be ignored and should not be
 * relied upon.
 * \ingroup GENERAL
 */
typedef enum {
    /** \brief No capabilities. */
    CUDBG_DEBUGGER_CAPABILITY_NONE                                       = 0,
    /** \brief Lazy function loading.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will enable CUDBG_EVENT_FUNCTIONS_LOADED events to be sent.
     * This capability should not be requested until the API client is prepared to handle these
     * events. */
    CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING                      = (1 << 0),
    /** \brief Suspend events.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will enable CUDBG_EVENT_ALL_DEVICES_SUSPENDED events to be sent.
     * This capability should not be requested until the API client is prepared to handle these
     * events. */
    CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS                             = (1 << 1),
    /** \brief Report exceptions in exited warps.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will enable reporting of exceptions in exited warps.
     * This capability should not be requested until the API client is prepared to handle such
     * situations. */
    CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS          = (1 << 2),
    /** \brief No context push/pop events.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will disable the CUDBG_EVENT_CONTEXT_PUSH and
     * CUDBG_EVENT_CONTEXT_POP events. This capability should be requested if the push/pop events
     * are not used by the API client. */
    CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS                 = (1 << 3),
    /** \brief Enable CUDA logs.
     * Dynamic flag: can be changed after initialization.
     * Requesting this capability will enable CUDA log capture by the debug engine and cause
     * CUDBG_EVENT_CUDA_LOGS_AVAILABLE and CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED events to be
     * sent. This capability should not be requested until the API client is prepared to handle
     * these events. */
    CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS                           = (1 << 4),
    /** \brief Collect CPU call stack for kernel launches.
     * Dynamic flag: can be changed after initialization.
     * Requesting this capability will enable collection of CPU call stack for kernel launches.
     * This capability should not be requested if the API client does not plan on calling the
     * readCPUCallStack() API. */
    CUDBG_DEBUGGER_CAPABILITY_COLLECT_CPU_CALL_STACK_FOR_KERNEL_LAUNCHES = (1 << 5),
    /** \brief Flush printf on suspend.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will enable flushing of the CUDA printf output on suspend.
     * This capability should generally be requested. */
    CUDBG_DEBUGGER_CAPABILITY_FLUSH_PRINTF_ON_SUSPEND                    = (1 << 6),
    /** \brief Enable break on launch feature.
     * Static flag: cannot be changed after initialization.
     * Requesting this capability will enable the break on launch feature. It may impact launch
     * performance for light workloads. This capability should be requested if break on launch is
     * used. */
    CUDBG_DEBUGGER_CAPABILITY_BREAK_ON_LAUNCH                            = (1 << 7),
} CUDBGCapabilityFlags;


/*--------------- Internal Breakpoint Entries for Error Reporting ------------*/

/**
 * \brief Name of the global function that's called to report a driver API error.
 * The API client can set a breakpoint on this function to be notified about driver API errors.
 * \ingroup GENERAL
 */
#define CUDBG_REPORT_DRIVER_API_ERROR                  cudbgReportDriverApiError
/**
 * \brief Name of the global variable containing the driver API error flag.
 * This variable is set by the API client to indicate which driver API errors should be reported.
 * See the CUDBGReportDriverApiErrorFlags enum for the list of available flags.
 * \ingroup GENERAL
 */
#define CUDBG_REPORT_DRIVER_API_ERROR_FLAGS            cudbgReportDriverApiErrorFlags
/**
 * \brief Name of the global variable containing the code of the driver API error being reported.
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_CODE           cudbgReportedDriverApiErrorCode
/**
 * \brief Name of the global variable containing the size of the name of the function in which the
 * driver API error occurred.
 *
 * This variable is set by the debug engine when a driver API error is
 * reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE cudbgReportedDriverApiErrorFuncNameSize
/**
 * \brief Name of the global variable containing the address of the name of the function in which
 * the driver API error occurred.
 *
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR cudbgReportedDriverApiErrorFuncNameAddr
/**
 * \brief Name of the global variable containing the driver API error source.
 * This variable is set by the debug engine when a driver API error is reported.
 * See the CUDBGReportedDriverApiErrorSource enum for the list of available sources.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE         cudbgReportedDriverApiErrorSource
/**
 * \brief Name of the global variable containing the size of the driver API error name being
 * reported.
 *
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_NAME_SIZE      cudbgReportedDriverApiErrorNameSize
/**
 * \brief Name of the global variable containing the address where the driver API error name is
 * stored.
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_NAME_ADDR      cudbgReportedDriverApiErrorNameAddr
/**
 * \brief Name of the global variable containing the size of the driver API error string being
 * reported.
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_STRING_SIZE    cudbgReportedDriverApiErrorStringSize
/**
 * \brief Name of the global variable containing the address where the driver API error string is
 * stored.
 * This variable is set by the debug engine when a driver API error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_API_ERROR_STRING_ADDR    cudbgReportedDriverApiErrorStringAddr

/**
 * \brief Name of the global function that's called to report an internal driver error.
 * The API client can set a breakpoint on this function to be notified about internal driver errors.
 * \ingroup GENERAL
 */
#define CUDBG_REPORT_DRIVER_INTERNAL_ERROR        cudbgReportDriverInternalError
/**
 * \brief Name of the global variable containing the driver internal error code.
 * This variable is set by the debug engine when an internal driver error is reported.
 * \ingroup GENERAL
 */
#define CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE cudbgReportedDriverInternalErrorCode


/*--------- Internal Breakpoint Entry for Debugger Session Initialization ----*/

/**
 * \brief Name of the global function that's called to report the completion of the attach
 * procedure.
 * The API client can set a breakpoint on this function to be notified about the
 * completion of the attach procedure. This function is called after the attach is done after
 * signalling CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD.
 * \ingroup GENERAL
 */
#define CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED cudbgReportAttachProcedureFinished

/**
 * \brief Name of the pre-init global function.
 * The API client can set a breakpoint on this function to be notified before the CUDA driver starts
 * its initialization.
 * \ingroup GENERAL
 */
#define CUDBG_PRE_INIT cudbgPreInit


/*----------------------------- API Return Types -----------------------------*/

/**
 * \brief Result values of all the API routines.
 * \ingroup GENERAL
 */
typedef enum {
    /** \brief The API call executed successfully. */
    CUDBG_SUCCESS                           = 0x0000,
    /** \brief Error type not listed below. */
    CUDBG_ERROR_UNKNOWN                     = 0x0001,
    /** \brief Cannot copy all the queried data into the buffer argument. */
    CUDBG_ERROR_BUFFER_TOO_SMALL            = 0x0002,
    /** \brief Function cannot be found in the CUDA kernel. */
    CUDBG_ERROR_UNKNOWN_FUNCTION            = 0x0003,
    /** \brief Wrong use of arguments (NULL pointer, illegal value,....). */
    CUDBG_ERROR_INVALID_ARGS                = 0x0004,
    /** \brief The API has not yet been properly initialized. */
    CUDBG_ERROR_UNINITIALIZED               = 0x0005,
    /** \brief Invalid block or thread coordinates were provided. */
    CUDBG_ERROR_INVALID_COORDINATES         = 0x0006,
    /** \brief Invalid memory segment requested. */
    CUDBG_ERROR_INVALID_MEMORY_SEGMENT      = 0x0007,
    /** \brief Requested address (+size) is not within proper segment boundaries. */
    CUDBG_ERROR_INVALID_MEMORY_ACCESS       = 0x0008,
    /** \brief Memory is not mapped and cannot be mapped. */
    CUDBG_ERROR_MEMORY_MAPPING_FAILED       = 0x0009,
    /** \brief A debug engine internal error occurred. */
    CUDBG_ERROR_INTERNAL                    = 0x000a,
    /** \brief Specified device cannot be found. */
    CUDBG_ERROR_INVALID_DEVICE              = 0x000b,
    /** \brief Specified sm cannot be found. */
    CUDBG_ERROR_INVALID_SM                  = 0x000c,
    /** \brief Specified warp cannot be found. */
    CUDBG_ERROR_INVALID_WARP                = 0x000d,
    /** \brief Specified lane cannot be found. */
    CUDBG_ERROR_INVALID_LANE                = 0x000e,
    /** \brief The requested operation is not allowed when the device is suspended. */
    CUDBG_ERROR_SUSPENDED_DEVICE            = 0x000f,
    /** \brief Device is running and not suspended. */
    CUDBG_ERROR_RUNNING_DEVICE              = 0x0010,
    /** \brief Reserved error code. */
    CUDBG_ERROR_RESERVED_0                  = 0x0011,
    /** \brief Address is out-of-range. */
    CUDBG_ERROR_INVALID_ADDRESS             = 0x0012,
    /** \brief The requested API is not available. */
    CUDBG_ERROR_INCOMPATIBLE_API            = 0x0013,
    /** \brief The API could not be initialized. */
    CUDBG_ERROR_INITIALIZATION_FAILURE      = 0x0014,
    /** \brief The specified grid is not valid. */
    CUDBG_ERROR_INVALID_GRID                = 0x0015,
    /** \brief The event queue is empty and there is no event left to be processed. */
    CUDBG_ERROR_NO_EVENT_AVAILABLE          = 0x0016,
    /** \brief Some devices were excluded because they have a watchdog associated with them. */
    CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED    = 0x0017,
    /** \brief All devices were exclude because they have a watchdog associated with them. */
    CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED     = 0x0018,
    /** \brief Specified attribute does not exist or is incorrect. */
    CUDBG_ERROR_INVALID_ATTRIBUTE           = 0x0019,
    /** \brief No function calls have been made on the device. */
    CUDBG_ERROR_ZERO_CALL_DEPTH             = 0x001a,
    /** \brief Specified call level is invalid. */
    CUDBG_ERROR_INVALID_CALL_LEVEL          = 0x001b,
    /** \brief Communication error between the debug engine and the application. */
    CUDBG_ERROR_COMMUNICATION_FAILURE       = 0x001c,
    /** \brief Specified context cannot be found. */
    CUDBG_ERROR_INVALID_CONTEXT             = 0x001d,
    /** \brief Requested address was not originally allocated from device memory (most likely
       visible in system memory). */
    CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM   = 0x001e,
    /** \brief Requested address is not mapped and cannot be unmapped. */
    CUDBG_ERROR_MEMORY_UNMAPPING_FAILED     = 0x001f,
    /** \brief The display driver is incompatible with the API. */
    CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER = 0x0020,
    /** \brief The specified module is not valid. */
    CUDBG_ERROR_INVALID_MODULE              = 0x0021,
    /** \brief The specified lane is not inside a device syscall. */
    CUDBG_ERROR_LANE_NOT_IN_SYSCALL         = 0x0022,
    /** \brief Reserved error code. */
    CUDBG_ERROR_RESERVED_1                  = 0x0023,
    /** \brief Some environment variable's value is invalid. */
    CUDBG_ERROR_INVALID_ENVVAR_ARGS         = 0x0024,
    /** \brief Error while allocating resources from the OS. */
    CUDBG_ERROR_OS_RESOURCES                = 0x0025,
    /** \brief Error while forking the debug engine process. */
    CUDBG_ERROR_FORK_FAILED                 = 0x0026,
    /** \brief No CUDA capable device was found. */
    CUDBG_ERROR_NO_DEVICE_AVAILABLE         = 0x0027,
    /** \brief Attaching to the CUDA program is not possible. */
    CUDBG_ERROR_ATTACH_NOT_POSSIBLE         = 0x0028,
    /** \brief The resumeWarpsUntilPC() API is not possible, use resumeDevice() or singleStepWarp()
       instead. */
    CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE    = 0x0029,
    /** \brief Specified warp mask is zero, or contains invalid warps. */
    CUDBG_ERROR_INVALID_WARP_MASK           = 0x002a,
    /** \brief Specified device pointer cannot be resolved to a GPU unambiguously because it is
       valid on more than one GPU. */
    CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS    = 0x002b,
    /** \brief Debug API entry point called from within a debug API callback. */
    CUDBG_ERROR_RECURSIVE_API_CALL          = 0x002c,
    /** \brief The requested data is missing. */
    CUDBG_ERROR_MISSING_DATA                = 0x002d,
    /** \brief Attempted operation is not supported. */
    CUDBG_ERROR_NOT_SUPPORTED               = 0x002e,
    /** \brief The current breakpoint state conflicts with the requested operation */
    CUDBG_ERROR_BREAKPOINT_STATE_CONFLICT   = 0x002f,
} CUDBGResult;

static const char* CUDBGResultNames[] = {
    "CUDBG_SUCCESS",
    "CUDBG_ERROR_UNKNOWN",
    "CUDBG_ERROR_BUFFER_TOO_SMALL",
    "CUDBG_ERROR_UNKNOWN_FUNCTION",
    "CUDBG_ERROR_INVALID_ARGS",
    "CUDBG_ERROR_UNINITIALIZED",
    "CUDBG_ERROR_INVALID_COORDINATES",
    "CUDBG_ERROR_INVALID_MEMORY_SEGMENT",
    "CUDBG_ERROR_INVALID_MEMORY_ACCESS",
    "CUDBG_ERROR_MEMORY_MAPPING_FAILED",
    "CUDBG_ERROR_INTERNAL",
    "CUDBG_ERROR_INVALID_DEVICE",
    "CUDBG_ERROR_INVALID_SM",
    "CUDBG_ERROR_INVALID_WARP",
    "CUDBG_ERROR_INVALID_LANE",
    "CUDBG_ERROR_SUSPENDED_DEVICE",
    "CUDBG_ERROR_RUNNING_DEVICE",
    "CUDBG_ERROR_RESERVED_0",
    "CUDBG_ERROR_INVALID_ADDRESS",
    "CUDBG_ERROR_INCOMPATIBLE_API",
    "CUDBG_ERROR_INITIALIZATION_FAILURE",
    "CUDBG_ERROR_INVALID_GRID",
    "CUDBG_ERROR_NO_EVENT_AVAILABLE",
    "CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED",
    "CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED",
    "CUDBG_ERROR_INVALID_ATTRIBUTE",
    "CUDBG_ERROR_ZERO_CALL_DEPTH",
    "CUDBG_ERROR_INVALID_CALL_LEVEL",
    "CUDBG_ERROR_COMMUNICATION_FAILURE",
    "CUDBG_ERROR_INVALID_CONTEXT",
    "CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM",
    "CUDBG_ERROR_MEMORY_UNMAPPING_FAILED",
    "CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER",
    "CUDBG_ERROR_INVALID_MODULE",
    "CUDBG_ERROR_LANE_NOT_IN_SYSCALL",
    "CUDBG_ERROR_RESERVED_1",
    "CUDBG_ERROR_INVALID_ENVVAR_ARGS",
    "CUDBG_ERROR_OS_RESOURCES",
    "CUDBG_ERROR_FORK_FAILED",
    "CUDBG_ERROR_NO_DEVICE_AVAILABLE",
    "CUDBG_ERROR_ATTACH_NOT_POSSIBLE",
    "CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE",
    "CUDBG_ERROR_INVALID_WARP_MASK",
    "CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS",
    "CUDBG_ERROR_RECURSIVE_API_CALL",
    "CUDBG_ERROR_MISSING_DATA",
    "CUDBG_ERROR_NOT_SUPPORTED",
    "CUDBG_ERROR_BREAKPOINT_STATE_CONFLICT",
};

/**
 * \brief Returns the string representation of a result value.
 * It is preferred to use this function instead of indexing CUDBGResultNames directly
 * as future versions of the old API methods might start returning new result values.
 * \ingroup GENERAL
 * \param error The result value to get the string representation of.
 * \return The string representation of the result value or "*UNDEFINED*" if the result value is not
 * recognized.
 */
static _CUDBG_INLINE const char* cudbgGetErrorString(CUDBGResult error)
{
    if (((unsigned)error) * sizeof(char*) >= sizeof(CUDBGResultNames))
    {
        return "*UNDEFINED*";
    }
    return CUDBGResultNames[(unsigned)error];
}


/*------------------------ API Error Reporting Flags -------------------------*/

/**
 * \brief API error reporting flags.
 * \ingroup GENERAL
 */
typedef enum {
    /** \brief No flags set (default behavior). */
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_NONE               = 0x0000,
    /** \brief When set, cudaErrorNotReady/cuErrorNotReady will not be reported. */
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_SUPPRESS_NOT_READY = (1U << 0),
} CUDBGReportDriverApiErrorFlags;

/**
 * \brief Driver API error source.
 * \ingroup GENERAL
 */
typedef enum {
    /** \brief No error/source. */
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_NONE    = 0x000,
    /** \brief The error originates from the CUDA Driver API. */
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_DRIVER  = 0x001,
    /** \brief The error originates from the CUDA Runtime API. */
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_RUNTIME = 0x002,
} CUDBGReportedDriverApiErrorSource;


/*------------------------------ Grid Attributes -----------------------------*/

/**
 * \brief Queryable grid attributes.
 * \ingroup GRID
 */
typedef enum {
    /** \brief Whether the launch is synchronous (blocking) or not. */
    CUDBG_ATTR_GRID_LAUNCH_BLOCKING = 0x000,
    /** \brief The id of the host thread that launched the grid. */
    CUDBG_ATTR_GRID_TID             = 0x001,
} CUDBGAttribute;

/**
 * \brief Grid attribute-value pair.
 * \ingroup GRID
 */
typedef struct {
    /** \brief The attribute to query. */
    CUDBGAttribute attribute;
    /** \brief The value of the attribute. */
    uint64_t value;
} CUDBGAttributeValuePair;

/**
 * \brief Grid status.
 * \ingroup GRID
 */
typedef enum {
    /** \brief An invalid grid ID was passed, or an error occurred during status lookup. */
    CUDBG_GRID_STATUS_INVALID,
    /** \brief The grid was launched but is not running on the HW yet. */
    CUDBG_GRID_STATUS_PENDING,
    /** \brief The grid is currently running on the HW. */
    CUDBG_GRID_STATUS_ACTIVE,
    /** \brief The grid is on the device, doing a join. */
    CUDBG_GRID_STATUS_SLEEPING,
    /** \brief The grid has finished executing. */
    CUDBG_GRID_STATUS_TERMINATED,
    /** \brief The grid is either QUEUED or TERMINATED. */
    CUDBG_GRID_STATUS_UNDETERMINED,
} CUDBGGridStatus;


/*------------------------------- Kernel Types -------------------------------*/

/**
 * \brief Kernel types.
 * \ingroup GRID
 */
typedef enum {
    /** \brief Unknown kernel type. Fall-back value. */
    CUDBG_KNL_TYPE_UNKNOWN     = 0x000,
    /** \brief System kernel, launched by the CUDA driver (cudaMemset, ...). */
    CUDBG_KNL_TYPE_SYSTEM      = 0x001,
    /** \brief Application kernel, launched by the application (user-defined or libraries). */
    CUDBG_KNL_TYPE_APPLICATION = 0x002,
} CUDBGKernelType;


/*--------------------------- Elf Image Properties ---------------------------*/

/**
 * \brief ELF Image Properties.
 * \note This enum is no longer used.
 */
typedef enum {
    /** \brief ELF image contains system kernels, launched by the CUDA driver. */
    CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM = 0x001,
} CUDBGElfImageProperties;


/*-------------------------- Physical Register Types -------------------------*/

/**
 * \brief Physical register types.
 * \ingroup DWARF
 */
typedef enum {
    /** \brief The physical register is invalid. */
    REG_CLASS_INVALID         = 0x000,
    /** \brief The physical register is a condition code register. Unused. */
    REG_CLASS_REG_CC          = 0x001,
    /** \brief The physical register is a predicate register. Unused. */
    REG_CLASS_REG_PRED        = 0x002,
    /** \brief The physical register is an address register. Unused. */
    REG_CLASS_REG_ADDR        = 0x003,
    /** \brief The physical register is a 16-bit register. Unused. */
    REG_CLASS_REG_HALF        = 0x004,
    /** \brief The physical register is a 32-bit register. */
    REG_CLASS_REG_FULL        = 0x005,
    /** \brief The content of the physical register has been spilled to memory. */
    REG_CLASS_MEM_LOCAL       = 0x006,
    /** \brief The content of the physical register has been spilled to the local stack (ABI only).
     */
    REG_CLASS_LMEM_REG_OFFSET = 0x007,
    /** \brief The physical register is a uniform predicate register. */
    REG_CLASS_UREG_PRED       = 0x009,
    /** \brief The physical register is a 16-bit uniform register. */
    REG_CLASS_UREG_HALF       = 0x00a,
    /** \brief The physical register is a 32-bit uniform register. */
    REG_CLASS_UREG_FULL       = 0x00b,
    /** \brief The physical register is a temp register spill.
     * Temp register spill 0 is RPC.LO and 1 is RPC.HI. */
    REG_CLASS_TEMP_REG_SPILL  = 0x00c,
} CUDBGRegClass;


/*---------------------------- Application Events ----------------------------*/

/**
 * \enum CUDBGEventKind
 * \brief Kinds of events sent by the debug engine to the API client.
 * \ingroup EVENT
 */
typedef enum {
    /** \brief Invalid event. */
    CUDBG_EVENT_INVALID                     = 0x000,
    /** \brief The ELF image for a CUDA source module is available. */
    CUDBG_EVENT_ELF_IMAGE_LOADED            = 0x001,
    /** \brief A CUDA kernel is about to be launched.
        \note DEPRECATED: This event is unreliable and will be removed in a future release. */
    CUDBG_EVENT_KERNEL_READY                = 0x002,
    /** \brief A CUDA kernel has terminated.
        \note DEPRECATED: This event is unreliable and will be removed in a future release. */
    CUDBG_EVENT_KERNEL_FINISHED             = 0x003,
    /** \brief An internal error occurred. The API may be unstable. */
    CUDBG_EVENT_INTERNAL_ERROR              = 0x004,
    /** \brief A CUDA context has been pushed. */
    CUDBG_EVENT_CTX_PUSH                    = 0x005,
    /** \brief A CUDA context has been popped. */
    CUDBG_EVENT_CTX_POP                     = 0x006,
    /** \brief A CUDA context has been created. */
    CUDBG_EVENT_CTX_CREATE                  = 0x007,
    /** \brief A CUDA context has been, popped if pushed, then destroyed. */
    CUDBG_EVENT_CTX_DESTROY                 = 0x008,
    /** \brief A timeout event is sent at regular interval. This event can safely be ignored.
     * Only sent by the classic backend. */
    CUDBG_EVENT_TIMEOUT                     = 0x009,
    /** \brief The attach process has completed and debugging of device code may start. */
    CUDBG_EVENT_ATTACH_COMPLETE             = 0x00a,
    /** \brief The detach process has completed. */
    CUDBG_EVENT_DETACH_COMPLETE             = 0x00b,
    /** \brief The ELF image for CUDA kernel(s) no longer available */
    CUDBG_EVENT_ELF_IMAGE_UNLOADED          = 0x00c,
    /** \brief A group of functions/kernels have been loaded
     * Will only be sent if the debug engine capability
     * CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING is set. */
    CUDBG_EVENT_FUNCTIONS_LOADED            = 0x00d,
    /** \brief All CUDA devices have been suspended due to a breakpoint hit or an exception.
     * Does not get sent for GPU events that result in synchronous API method calls, such as
     * singleStepWarp or resumeWarpsUntilPC. Will only be sent if the debug engine capability
     * CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS is set.*/
    CUDBG_EVENT_ALL_DEVICES_SUSPENDED       = 0x00e,
    /** \brief (Async or Sync) CUDA Logs are available for the debugger to consume.
     * After receiving this event, debuggers should drain all available log entries by repeatedly
     * calling consumeCudaLogs until no more logs are available. For asynchronous CUDA logs, this
     * event is only sent for the first log message that's generated after the client has read all
     * logs with consumeCudaLogs. For synchronous CUDA logs, this event is sent for each log message
     * that's generated and the emitting thread will wait for the client to acknowledge the event.
     * See \ref CUDBGAPI_st::setCudaLogRules for more details.
     * It is not sent by default, and can be enabled via the capability
     * CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS. */
    CUDBG_EVENT_CUDA_LOGS_AVAILABLE         = 0x00f,
    /** \brief (Sync) CUDA Logs buffer has reached an implementation-defined threshold.
     * The client should call consumeCudaLogs to avoid excessive log buildup.
     * New logs will still be collected even if consumeCudaLogs is not called. */
    CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED = 0x010,
    /** \brief (Sync) Asynchronous single step operation has been completed.
     * Call singleStepWarp or resumeWarpsUntilPC with CUDBG_SINGLE_STEP_FLAGS_NON_BLOCKING to enable
     * this event. This event is only sent if the single step operation was completed. If it's
     * interrupted by a breakpoint or an exception, a CUDBG_EVENT_ALL_DEVICES_SUSPENDED event will
     * be sent instead. */
    CUDBG_EVENT_SINGLE_STEP_COMPLETE        = 0x011,
    /** \brief (Async) The CUDA log ruleset has changed.
     * This event is sent after the new ruleset has been applied. The client can call consumeCudaLogs to drain
     * the log buffer, after which it's guaranteed that the new logs will only match the ruleset with the
     * index larger than or equal to the ruleset index of this event. This allows the clients to stop tracking
     * the old rulesets. */
    CUDBG_EVENT_CUDA_LOGS_RULESET_CHANGED   = 0x012,
} CUDBGEventKind;


/*------------------------------- Kernel Origin ------------------------------*/

/**
 * \brief Kernel origin.
 * \ingroup GRID
 */
typedef enum {
    /** \brief The kernel was launched from the CPU. */
    CUDBG_KNL_ORIGIN_CPU = 0x000,
    /** \brief The kernel was launched from the GPU. */
    CUDBG_KNL_ORIGIN_GPU = 0x001,
} CUDBGKernelOrigin;


/*------------------------ Kernel Launch Notify Mode -------------------------*/

/**
 * \brief Kernel launch notification mode.
 * \ingroup GRID
 */
typedef enum {
    /** \brief Kernel launches generate launch notification events. */
    CUDBG_KNL_LAUNCH_NOTIFY_EVENT = 0x000,
    /** \brief Kernel launches do not generate any notification. */
    CUDBG_KNL_LAUNCH_NOTIFY_DEFER = 0x001,
} CUDBGKernelLaunchNotifyMode;


/*---------------------- Application Event Queue Type ------------------------*/

/**
 * \brief Application event queue type.
 * \ingroup EVENT
 */
typedef enum {
    /** \brief Synchronous event queue. */
    CUDBG_EVENT_QUEUE_TYPE_SYNC  = 0,
    /** \brief Asynchronous event queue. */
    CUDBG_EVENT_QUEUE_TYPE_ASYNC = 1,
} CUDBGEventQueueType;


/*------------------------------ Elf Image Type ------------------------------*/

/**
 * \brief CUDA ELF image type.
 * \ingroup DWARF
 */
typedef enum {
    /** \brief Non-relocated ELF image. */
    CUDBG_ELF_IMAGE_TYPE_NONRELOCATED = 0,
    /** \brief Relocated ELF image. */
    CUDBG_ELF_IMAGE_TYPE_RELOCATED    = 1,
} CUDBGElfImageType;


/*------------------------------ Code Address --------------------------------*/

/**
 * \brief Describes which adjusted code address is to be returned.
 * \ingroup BP
 */
typedef enum {
    /** \brief Get the adjusted previous code address. */
    CUDBG_ADJ_PREVIOUS_ADDRESS = 0x000,
    /** \brief Get the adjusted next code address. */
    CUDBG_ADJ_CURRENT_ADDRESS  = 0x001,
    /** \brief Get the adjusted current code address. */
    CUDBG_ADJ_NEXT_ADDRESS     = 0x002,
} CUDBGAdjAddrAction;


/**
 * \brief Single step operation type (API method that was used to start the single step operation)
 * \ingroup EXEC
 */
typedef enum {
    /** \brief Invalid single step operation type. */
    CUDBG_SINGLE_STEP_TYPE_INVALID               = 0,
    /** \brief The singleStepWarp() API method was used to start the single step operation. */
    CUDBG_SINGLE_STEP_TYPE_SINGLE_STEP_WARP      = 1,
    /** \brief The resumeWarpsUntilPC() API method was used to start the single step operation. */
    CUDBG_SINGLE_STEP_TYPE_RESUME_WARPS_UNTIL_PC = 2,
} CUDBGSingleStepType;


/*---------------------------- Single Step Flags -----------------------------*/

/**
 * \brief Single step flags.
 * \ingroup EXEC
 */
typedef enum {
    /** \brief Default behavior. */
    CUDBG_SINGLE_STEP_FLAGS_NONE                       = 0,
    /** \brief Disable optimized warp barrier stepping.
     * Do not step over warp-wide barriers using a breakpoint and resume,
     * instead perform a single step and return. Passing this flag in means
     * that the API client plans to repeat the singleStepWarp() call until
     * the warp barrier is stepped over. This gives a more precise exception
     * information if an exception is encountered by the diverged threads
     * while stepping.
     * This flag is only valid for the singleStepWarp() API method,
     * resumeWarpsUntilPC() always steps over warp-wide barriers.
     */
    CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS = (1U << 0),
    /** \brief Don't block on the stepping operations, instead return early and send an event once
       stepping is done */
    CUDBG_SINGLE_STEP_FLAGS_NON_BLOCKING               = (1U << 1),
} CUDBGSingleStepFlags;

/**
 * \brief Event information container for API version 3.0.
 * \note DEPRECATED: Use CUDBGEvent instead.
 * For documentation of individual fields, see the documentation of the CUDBGEvent struct instead.
 * \ingroup EVENT
 * \sa CUDBGEvent
 */
typedef struct {
    CUDBGEventKind kind;
    union cases30_st {
        struct elfImageLoaded30_st {
            char* relocatedElfImage;
            char* nonRelocatedElfImage;
            uint32_t size;
        } elfImageLoaded;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelReady;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelFinished;
    } cases;
} CUDBGEvent30;

/**
 * \brief Event information container for API version 3.2.
 * \note DEPRECATED: Use CUDBGEvent instead.
 * For documentation of individual fields, see the documentation of the CUDBGEvent struct instead.
 * \ingroup EVENT
 * \sa CUDBGEvent
 */
typedef struct {
    CUDBGEventKind kind;
    union cases32_st {
        struct elfImageLoaded32_st {
            char* relocatedElfImage;
            char* nonRelocatedElfImage;
            uint32_t size;
            uint32_t dev;
            uint64_t context;
            uint64_t module;
        } elfImageLoaded;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady32_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelReady;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished32_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
    } cases;
} CUDBGEvent32;

/**
 * \brief Event information container for API version 4.2.
 * \note DEPRECATED: Use CUDBGEvent instead.
 * For documentation of individual fields, see the documentation of the CUDBGEvent struct instead.
 * \ingroup EVENT
 * \sa CUDBGEvent
 */
typedef struct {
    CUDBGEventKind kind;
    union cases42_st {
        struct elfImageLoaded42_st {
            char* relocatedElfImage;
            char* nonRelocatedElfImage;
            uint32_t size32;
            uint32_t dev;
            uint64_t context;
            uint64_t module;
            uint64_t size;
        } elfImageLoaded;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady42_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3 gridDim;
            CuDim3 blockDim;
            CUDBGKernelType type;
        } kernelReady;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished42_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
    } cases;
} CUDBGEvent42;

/**
 * \brief Event information container for API version 5.0.
 * \note DEPRECATED: Use CUDBGEvent instead.
 * For documentation of individual fields, see the documentation of the CUDBGEvent struct instead.
 * \ingroup EVENT
 * \sa CUDBGEvent
 */
typedef struct {
    CUDBGEventKind kind;
    union cases50_st {
        struct elfImageLoaded50_st {
            char* relocatedElfImage;
            char* nonRelocatedElfImage;
            uint32_t size32;
            uint32_t dev;
            uint64_t context;
            uint64_t module;
            uint64_t size;
        } elfImageLoaded;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady50_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3 gridDim;
            CuDim3 blockDim;
            CUDBGKernelType type;
        } kernelReady;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished50_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError50_st {
            CUDBGResult errorType;
        } internalError;
    } cases;
} CUDBGEvent50;

/**
 * \brief Event information container for API version 5.5.
 * \note DEPRECATED: Use CUDBGEvent instead.
 * For documentation of individual fields, see the documentation of the CUDBGEvent struct instead.
 * \ingroup EVENT
 * \sa CUDBGEvent
 */
typedef struct {
    CUDBGEventKind kind;
    union cases55_st {
        struct elfImageLoaded55_st {
            char* relocatedElfImage;
            char* nonRelocatedElfImage;
            uint32_t size32;
            uint32_t dev;
            uint64_t context;
            uint64_t module;
            uint64_t size;
        } elfImageLoaded;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady55_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3 gridDim;
            CuDim3 blockDim;
            CUDBGKernelType type;
            /** \brief Reserved deprecated field kept for ABI compatibility.
                \note DEPRECATED: Since CUDA 13.4, this field is no longer used.
                Used to be ID of the parent grid (in case of a device-launched CDP grid). */
            uint64_t reserved0;
            uint64_t gridId64;
            CUDBGKernelOrigin origin;
        } kernelReady;
        /** \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished55_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            uint64_t gridId64;
        } kernelFinished;
        struct contextPush55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError55_st {
            CUDBGResult errorType;
        } internalError;
    } cases;
} CUDBGEvent55;

/**
 * \struct CUDBGEvent
 * \brief Event information container.
 * \ingroup EVENT
 */
#pragma pack(push, 1)
typedef struct {
    CUDBGEventKind kind;
    /** \brief Information for each type of event. */
    union cases_st {
        /** \brief Information about the loaded ELF image. */
        struct elfImageLoaded_st {
            /** \brief Device index of the loaded module. */
            uint32_t dev;
            /** \brief Context handle of the loaded module. */
            uint64_t context;
            /** \brief Loaded module handle. */
            uint64_t module;
            /** \brief Size of the ELF image (64-bit). */
            uint64_t size;
            /** \brief ELF image handle. */
            uint64_t handle;
            /** \brief ELF image properties. */
            uint32_t properties;
        } elfImageLoaded;
        /** \brief Information about the ELF image about to be unloaded. */
        struct elfImageUnloaded_st {
            /** \brief Device index of the module being unloaded. */
            uint32_t dev;
            /** \brief Context handle of the module being unloaded. */
            uint64_t context;
            /** \brief Module handle of the module being unloaded. */
            uint64_t module;
            /** \brief Size of the ELF image (64-bit). */
            uint64_t size;
            /** \brief ELF image handle. */
            uint64_t handle;
        } elfImageUnloaded;
        /** \brief Information about the kernel ready to be launched.
            \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelReady_st {
            /** \brief Device index of the kernel. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the kernel (Linux only). */
            uint32_t tid;
            /** \brief Grid ID of the kernel. */
            uint64_t gridId;
            /** \brief Context handle of the kernel. */
            uint64_t context;
            /** \brief Module handleof the kernel. */
            uint64_t module;
            /** \brief Function handle of the kernel. */
            uint64_t function;
            /** \brief Entry address of the kernel. */
            uint64_t functionEntry;
            /** \brief Grid dimensions of the kernel. */
            CuDim3 gridDim;
            /** \brief Block dimensions of the kernel. */
            CuDim3 blockDim;
            /** \brief Type of the kernel: system or application. */
            CUDBGKernelType type;
            /** \brief Reserved deprecated field kept for ABI compatibility.
                \note DEPRECATED: Since CUDA 13.4, this field is no longer used.
                Used to be ID of the parent grid (in case of a device-launched CDP grid). */
            uint64_t reserved0;
            /** \brief Origin of the kernel (CPU or GPU). */
            CUDBGKernelOrigin origin;
        } kernelReady;
        /** \brief Information about the kernel that just terminated.
            \note DEPRECATED: This event is unreliable and will be removed in a future release. */
        struct kernelFinished_st {
            /** \brief Device index of the kernel. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the kernel (Linux only). */
            uint32_t tid;
            /** \brief Context handle of the kernel. */
            uint64_t context;
            /** \brief Module handle of the kernel. */
            uint64_t module;
            /** \brief Function handle of the kernel. */
            uint64_t function;
            /** \brief Entry address of the kernel. */
            uint64_t functionEntry;
            /** \brief Grid ID of the kernel. */
            uint64_t gridId;
        } kernelFinished;
        /** \brief Information about the context being pushed. */
        struct contextPush_st {
            /** \brief Device index of the context. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the context (Linux only). */
            uint32_t tid;
            /** \brief Context handle of the context being pushed. */
            uint64_t context;
        } contextPush;
        /** \brief Information about the context being popped. */
        struct contextPop_st {
            /** \brief Device index of the context. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the context (Linux only). */
            uint32_t tid;
            /** \brief Context handle of the context being popped. */
            uint64_t context;
        } contextPop;
        /** \brief Information about the context being created. */
        struct contextCreate_st {
            /** \brief Device index of the context. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the context (Linux only). */
            uint32_t tid;
            /** \brief Context handle of the context being created. */
            uint64_t context;
        } contextCreate;
        /** \brief Information about the context being destroyed. */
        struct contextDestroy_st {
            /** \brief Device index of the context. */
            uint32_t dev;
            /** \brief Host thread id (or LWP id) of the thread hosting the context (Linux only). */
            uint32_t tid;
            /** \brief Context handle of the context being destroyed. */
            uint64_t context;
        } contextDestroy;
        /** \brief Information about internal erros. */
        struct internalError_st {
            /** \brief Type of the internal error. */
            CUDBGResult errorType;
        } internalError;
        /** \brief Information about the functions being lazily loaded. */
        struct functionsLoaded_st {
            /** \brief Device index of the module containing the functions. */
            uint32_t dev;
            /** \brief Functions count. */
            uint32_t count;
            /** \brief Context handle of the module containing the functions. */
            uint64_t context;
            /** \brief Module handle of the module containing the functions. */
            uint64_t module;
        } functionsLoaded;
        /** \brief Information about an event which forced all devices to be suspended. */
        struct allDevicesSuspended_st {
            /** \brief Device bitmask. This mask has bits set for devices with any warps that hit a
             * breakpoint. */
            uint64_t brokenDevicesMask;
            /** \brief Device bitmask. This mask has bits set for devices with any warps that hit an
             * exception. */
            uint64_t faultedDevicesMask;
        } allDevicesSuspended;
        /** \brief Information about a single step operation that has completed. */
        struct singleStepComplete_st {
            /** \brief Device that was stepping */
            uint32_t dev;
            /** \brief SM that was stepping */
            uint32_t sm;
            /** \brief Context that was stepping */
            uint64_t context;
            /** \brief Warps that were requested to be stepped */
            uint64_t originalWarpMask;
            /** \brief Warps that ended up being stepped (only contains warps from the original SM,
             * other SMs could be stepped and are not reported) */
            uint64_t finalWarpMask;
            /** \brief Device bitmask. This mask has bits set for devices with any warps that hit a
             * breakpoint during the step. */
            uint64_t brokenDevicesMask;
            /** \brief Device bitmask. This mask has bits set for devices with any warps that hit an
             * exception during the step. */
            uint64_t faultedDevicesMask;
            /** \brief Which single step method has initiated stepping */
            CUDBGSingleStepType type;
        } singleStepComplete;
        /** \brief Information about the CUDA log ruleset being changed. */
        struct cudaLogsRulesetChanged_st {
            /** \brief The index of the newly applied ruleset. */
            uint32_t rulesetIndex;
        } cudaLogsRulesetChanged;
    } cases;
} CUDBGEvent;
#pragma pack(pop)

/**
 * \brief Callback data passed to callback set with setNotifyNewEventCallback40 function.
 * \note DEPRECATED: Use CUDBGEventCallbackData instead.
 * \ingroup EVENT
 */
typedef struct {
    /** \brief Host thread id of the context generating the event. Zero if not available. */
    uint32_t tid;
} CUDBGEventCallbackData40;

/**
 * \brief Callback data passed to callback set with setNotifyNewEventCallback41 function.
 * \note DEPRECATED: Use CUDBGEventCallbackData instead.
 * \ingroup EVENT
 */
typedef struct {
    /** \brief Host thread id of the context generating the event. Zero if not available. */
    uint32_t tid;
    /** \brief A boolean notifying the debugger that the debug API timed while waiting for a reponse
     * from the debugger to a previous event. It is up to the debugger to decide what to do in
     * response to a timeout. */
    uint32_t timeout;
} CUDBGEventCallbackData41;

/**
 * \brief Callback data passed to callback set with setNotifyNewEventCallback function.
 * \ingroup EVENT
 */
typedef struct {
    /** \brief User data passed to the callback. */
    void* userData;
    /** \brief Host thread id of the context generating the event. Zero if not available. */
    uint32_t tid;
} CUDBGEventCallbackData;

#pragma pack(push, 1)
/**
 * \brief Grid info.
 * \note DEPRECATED: Use CUDBGGridInfo instead.
 * For documentation of individual fields, see the documentation of the CUDBGGridInfo struct
 * instead.
 * \ingroup GRID
 * \sa CUDBGGridInfo
 */
typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3 gridDim;
    CuDim3 blockDim;
    CUDBGKernelType type;
    /** \brief Reserved deprecated field kept for ABI compatibility.
        \note DEPRECATED: Since CUDA 13.4, this field is no longer used.
        Used to be ID of the parent grid (in case of a device-launched CDP grid). */
    uint64_t reserved0;
    CUDBGKernelOrigin origin;
} CUDBGGridInfo55;

/**
 * \brief Grid info.
 * \note DEPRECATED: Use CUDBGGridInfo instead.
 * For documentation of individual fields, see the documentation of the CUDBGGridInfo struct
 * instead.
 * \ingroup GRID
 * \sa CUDBGGridInfo
 */
typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3 gridDim;
    CuDim3 blockDim;
    CUDBGKernelType type;
    /** \brief Reserved deprecated field kept for ABI compatibility.
        \note DEPRECATED: Since CUDA 13.4, this field is no longer used.
        Used to be ID of the parent grid (in case of a device-launched CDP grid). */
    uint64_t reserved0;
    CUDBGKernelOrigin origin;
    CuDim3 clusterDim;
} CUDBGGridInfo120;

/**
 * \brief Information about a CUDA grid.
 * \ingroup GRID
 */
typedef struct {
    /** \brief Index of the device this grid is running on. */
    uint32_t dev;
    /** \brief Grid ID of this grid. */
    uint64_t gridId64;
    /** \brief Thread ID of the host thread that launched this grid. */
    uint32_t tid;
    /** \brief Context handle of the context this grid belongs to. */
    uint64_t context;
    /** \brief Module handle of the module this grid belongs to. */
    uint64_t module;
    /** \brief Function handle of the function corresponding to this grid. */
    uint64_t function;
    /** \brief Entry address of the function corresponding to this grid. */
    uint64_t functionEntry;
    /** \brief Grid dimensions. */
    CuDim3 gridDim;
    /** \brief Block dimensions. */
    CuDim3 blockDim;
    /** \brief Grid type: system or application. */
    CUDBGKernelType type;
    /** \brief Reserved deprecated field kept for ABI compatibility.
        \note DEPRECATED: Since CUDA 13.4, this field is no longer used.
        Used to be ID of the parent grid (in case of a device-launched CDP grid). */
    uint64_t reserved0;
    /** \brief Origin of this grid: CPU or GPU. */
    CUDBGKernelOrigin origin;
    /** \brief Number of blocks in the cluster. */
    CuDim3 clusterDim;
    /** \brief Preferred number of blocks in the cluster. */
    CuDim3 preferredClusterDim;
} CUDBGGridInfo;
#pragma pack(pop)

/**
 * \brief Information about a lazily loaded function.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Section index of a loaded function. */
    uint64_t sectionIndex;
    /** \brief Address of a loaded function. */
    uint64_t address;
} CUDBGLoadedFunctionInfo;
#pragma pack(pop)

/**
 * \brief Function type of the function called to notify debugger of the presence of a new event in
 * the event queue.
 * \note DEPRECATED: Use CUDBGNotifyNewEventCallback instead.
 * \ingroup EVENT
 * \param data - pointer to the event callback data.
 */
typedef void (*CUDBGNotifyNewEventCallback31)(void* data);
/**
 * \brief Function type of the function called to notify debugger of the presence of a new event in
 * the event queue.
 * \note DEPRECATED: Use CUDBGNotifyNewEventCallback instead.
 * \ingroup EVENT
 * \param data - pointer to the event callback data.
 */
typedef void (*CUDBGNotifyNewEventCallback40)(CUDBGEventCallbackData40* data);
/**
 * \brief Function type of the function called to notify debugger of the presence of a new event in
 * the event queue.
 * \note DEPRECATED: Use CUDBGNotifyNewEventCallback instead.
 * \ingroup EVENT
 * \param data - pointer to the event callback data.
 */
typedef void (*CUDBGNotifyNewEventCallback41)(CUDBGEventCallbackData41* data);
/**
 * \brief Function type of the function called to notify debugger of the presence of a new event in
 * the event queue.
 * \ingroup EVENT
 * \param data - pointer to the event callback data.
 */
typedef void (*CUDBGNotifyNewEventCallback)(CUDBGEventCallbackData* data);


/*------------------------------ CUDBGException ------------------------------*/

/**
 * \brief Harwdare Exception Types.
 */
typedef enum {
    /** \brief Reported if we do not know what exception the chip has hit.
     * Global error. */
    CUDBG_EXCEPTION_UNKNOWN                         = 0xFFFFFFFFU, // Force sizeof(CUDBGException_t)==4
    /** \brief Reported when there is no exception on the chip.
     * No error. */
    CUDBG_EXCEPTION_NONE                            = 0,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_0                      = 1,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_1                      = 2,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_2                      = 3,
    /** \brief Reported when any lane in a warp executes an illegal instruction.
     * Warp error: invalid branch target, invalid opcode, misaligned/oor reg, invalid immediates,
     * etc. */
    CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION        = 4,
    /** \brief Reported when any lane in a warp accesses memory that is out of range.
     * Warp error: lmem_lo/hi, shared, and va accesses. */
    CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS       = 5,
    /** \brief Reported when any lane in a warp accesses memory that is misaligned.
     * Warp error: lmem_lo/hi, shared, and va accesses. */
    CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS         = 6,
    /** \brief Reported when any lane in a warp executes an instruction that accesses a memory space
     * that is not permitted for that instruction. Warp error. */
    CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE      = 7,
    /** \brief Reported when any lane in a warp advances its PC beyond the valid address.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_INVALID_PC                 = 8,
    /** \brief Reported when any lane in a warp hits (uncommon) stack issues.
     * Warp error: stack error or api stack overflow. */
    CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW    = 9,
    /** \brief Reported when MMU detects an error.
     * Global error: L1 error status field is set in the global esr -- for the most part this
     * catches errors SM couldn't catch with oor address detection. */
    CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS          = 10,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_3                      = 11,
    /** \brief Reported when any lane in a warp asserts.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_ASSERT                     = 12,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_4                      = 13,
    /** \brief Reported when any lane in a warp encounters an illegal address.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS            = 14,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_5                      = 15,
    /** \brief DEPRECATED: Since CUDA 13.1, this exception is no longer reported. */
    CUDBG_EXCEPTION_RESERVED_6                      = 16,
    /** \brief Reported when any lane in a cluster encounters an out of range address.
     * Cluster error. */
    CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS    = 17,
    /** \brief Reported when any lane in a cluster encounters a block not present error.
     * Cluster error. */
    CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT       = 18,
    /** \brief Reported when any lane in a warp encounters a stack canary error.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_STACK_CANARY               = 19,
    /** \brief Reported when any lane in a warp encounters a tmem access check error.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_TMEM_ACCESS_CHECK          = 20,
    /** \brief Reported when any lane in a warp encounters a tmem leak error.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_TMEM_LEAK                  = 21,
    /** \brief Reported when any lane in a warp encounters a call requires newer driver error.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_CALL_REQUIRES_NEWER_DRIVER = 22,
    /** \brief Reported when any lane in a warp has a target address not aligned with PC size in
     * bytes. Warp error. */
    CUDBG_EXCEPTION_WARP_MISALIGNED_PC              = 23,
    /** \brief Reported when any lane in a warp advances its PC beyond the valid Virtual Address
     * space. Warp error. */
    CUDBG_EXCEPTION_WARP_PC_OVERFLOW                = 24,
    /** \brief Reported when any lane in a warp commits a misaligned read/write to a register.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_MISALIGNED_REG             = 25,
    /** \brief Reported when any lane in a warp executes an instruction that has an illegal
     * instruction encoding. Warp error. */
    CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_ENCODING     = 26,
    /** \brief Reported when any lane in a warp executes an instruction that has an illegal
     * instruction parameter. Warp error. */
    CUDBG_EXCEPTION_WARP_ILLEGAL_INSTR_PARAM        = 27,
    /** \brief Reported when any lane in a warp tries to access a register which is out of range.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_REGISTER      = 28,
    /** \brief Reported when any lane in a warp encounters an invalid constant address error.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_INVALID_CONST_ADDR_LDC     = 29,
    /** \brief Reported when a memory transaction from this warp incurs a MMU fault.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_MMU_FAULT                  = 30,
    /** \brief Reported by mbarrier for cases where the transcation barrier is in the locked state
     * or when there are more arrivals than the expected arrival count. Warp error. */
    CUDBG_EXCEPTION_WARP_ARRIVE                     = 31,
    /** \brief Reported when cluster access result in uncorrectable error.
     * Cluster error. */
    CUDBG_EXCEPTION_CLUSTER_POISON                  = 32,
    /** \brief Reported when a any lane in a warp exceeds API stack limit.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_API_STACK_ERROR            = 33,
    /** \brief Reported when a block is not present during various operations (cluster operations,
     * barrier synchronization, etc). Cluster error. */
    CUDBG_EXCEPTION_WARP_BLOCK_NOT_PRESENT          = 34,
    /** \brief Reported when there is a stack overflow within the warp.
     * Warp error. */
    CUDBG_EXCEPTION_WARP_USER_STACK_OVERFLOW        = 35,
    /** \brief Reported when a TMA syscall error occurs.
     * Warp error. */
    CUDBG_EXCEPTION_TMA_SYSCALL                     = 36,
} CUDBGException_t;


/*-------------------------------- UVM Enums ---------------------------------*/

/**
 * \brief UVM memory access type.
 * \note This enum is currently not used.
 */
typedef enum {
    CUDBG_UVM_MEMORY_ACCESS_TYPE_UNKNOWN  = 0xFFFFFFFFU,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_INVALID  = 0,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_READ     = 1,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_WRITE    = 2,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_ATOMIC   = 3,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_PREFETCH = 4,
} CUDBGUvmMemoryAccessType_t;

/**
 * \brief UVM fault type.
 * \note This enum is currently not used.
 */
typedef enum {
    CUDBG_UVM_FAULT_TYPE_UNKNOWN              = 0xFFFFFFFFU,
    CUDBG_UVM_FAULT_TYPE_INVALID              = 0,
    CUDBG_UVM_FAULT_TYPE_INVALID_PDE          = 1,
    CUDBG_UVM_FAULT_TYPE_INVALID_PTE          = 2,
    CUDBG_UVM_FAULT_TYPE_WRITE                = 3,
    CUDBG_UVM_FAULT_TYPE_ATOMIC               = 4,
    CUDBG_UVM_FAULT_TYPE_INVALID_PDE_SIZE     = 5,
    CUDBG_UVM_FAULT_TYPE_LIMIT_VIOLATION      = 6,
    CUDBG_UVM_FAULT_TYPE_UNBOUND_INST_BLOCK   = 7,
    CUDBG_UVM_FAULT_TYPE_PRIV_VIOLATION       = 8,
    CUDBG_UVM_FAULT_TYPE_PITCH_MASK_VIOLATION = 9,
    CUDBG_UVM_FAULT_TYPE_WORK_CREATION        = 10,
    CUDBG_UVM_FAULT_TYPE_UNSUPPORTED_APERTURE = 11,
    CUDBG_UVM_FAULT_TYPE_COMPRESSION_FAILURE  = 12,
    CUDBG_UVM_FAULT_TYPE_UNSUPPORTED_KIND     = 13,
    CUDBG_UVM_FAULT_TYPE_REGION_VIOLATION     = 14,
    CUDBG_UVM_FAULT_TYPE_POISON               = 15,
} CUDBGUvmFaultType_t;

/**
 * \brief UVM fatal error reason.
 * \note This enum is currently not used.
 */
typedef enum {
    CUDBG_UVM_FATAL_REASON_UNKNOWN             = 0xFFFFFFFFU,
    CUDBG_UVM_FATAL_REASON_INVALID             = 0,
    CUDBG_UVM_FATAL_REASON_INVALID_ADDRESS     = 1,
    CUDBG_UVM_FATAL_REASON_INVALID_PERMISSIONS = 2,
    CUDBG_UVM_FATAL_REASON_INVALID_FAULT_TYPE  = 3,
    CUDBG_UVM_FATAL_REASON_OUT_OF_MEMORY       = 4,
    CUDBG_UVM_FATAL_REASON_INTERNAL_ERROR      = 5,
    CUDBG_UVM_FATAL_REASON_INVALID_OPERATION   = 6,
} CUDBGUvmFatalReason_t;


/*------------------------------- Warp State ---------------------------------*/

#pragma pack(push, 1)
/**
 * \brief Lane state (state of a single thread).
 * \ingroup READ
 */
typedef struct {
    /** \brief (VA) PC of the thread. */
    uint64_t virtualPC;
    /** \brief Thread index of the thread. */
    CuDim3 threadIdx;
    /** \brief Exception of the thread (if any). */
    CUDBGException_t exception;
} CUDBGLaneState;

/**
 * \brief Warp state for API version 6.0.
 * \note DEPRECATED: Use CUDBGWarpState instead.
 * For documentation of individual fields, see the documentation of the CUDBGWarpState struct
 * instead.
 * \ingroup READ
 * \sa CUDBGWarpState
 */
typedef struct {
    uint64_t gridId;
    uint64_t errorPC;
    CuDim3 blockIdx;
    uint32_t validLanes;
    uint32_t activeLanes;
    uint32_t errorPCValid;
    CUDBGLaneState lane[32];
} CUDBGWarpState60;

/**
 * \brief Warp state for API version 12.0.
 * \note DEPRECATED: Use CUDBGWarpState instead.
 * For documentation of individual fields, see the documentation of the CUDBGWarpState struct
 * instead.
 * \ingroup READ
 * \sa CUDBGWarpState
 */
typedef struct {
    uint64_t gridId;
    uint64_t errorPC;
    CuDim3 blockIdx;
    uint32_t validLanes;
    uint32_t activeLanes;
    uint32_t errorPCValid;
    CUDBGLaneState lane[32];
    CuDim3 clusterIdx;
} CUDBGWarpState120;

/**
 * \brief Warp state for API version 12.7.
 * \note DEPRECATED: Use CUDBGWarpState instead.
 * For documentation of individual fields, see the documentation of the CUDBGWarpState struct
 * instead.
 * \ingroup READ
 * \sa CUDBGWarpState
 */
typedef struct {
    uint64_t gridId;
    uint64_t errorPC;
    CuDim3 blockIdx;
    uint32_t validLanes;
    uint32_t activeLanes;
    uint32_t errorPCValid;
    CUDBGLaneState lane[32];
    CuDim3 clusterIdx;
    CuDim3 clusterDim;
    uint32_t clusterExceptionTargetBlockIdxValid;
    CuDim3 clusterExceptionTargetBlockIdx;
} CUDBGWarpState127;

/**
 * \brief Warp state information.
 * \ingroup READ
 */
typedef struct {
    /** \brief Grid ID of the grid running in the warp. */
    uint64_t gridId;
    /** \brief Error PC of the warp (if any). */
    uint64_t errorPC;
    /** \brief Block index of the block containing the warp. */
    CuDim3 blockIdx;
    /** \brief Lane mask of valid threads in the warp. */
    uint32_t validLanes;
    /** \brief Lane mask of active threads in the warp. */
    uint32_t activeLanes;
    /** \brief Whether the error PC is valid. */
    uint32_t errorPCValid;
    /** \brief State of the lanes (threads) in the warp. */
    CUDBGLaneState lane[32];
    /** \brief Cluster index of the cluster containing the warp. */
    CuDim3 clusterIdx;
    /** \brief Cluster dimensions of the cluster containing the warp.
     * Can be different for different warps in the same grid. */
    CuDim3 clusterDim;
    /** \brief Whether the cluster exception target block index is valid. */
    uint32_t clusterExceptionTargetBlockIdxValid;
    /** \brief Cluster exception target block index of the the warp. */
    CuDim3 clusterExceptionTargetBlockIdx;
    /** \brief Lane mask of threads in a syscall. */
    uint32_t inSyscallLanes;
} CUDBGWarpState;

/**
 * \brief Warp resources.
 * These resources can change at runtime between suspends.
 * \ingroup READ
 */
typedef struct {
    /** \brief Shared memory size used by the warp. */
    uint32_t sharedMemSize;
    /** \brief Number of registers used by the warp. */
    uint32_t numRegisters;
} CUDBGWarpResources;
#pragma pack(pop)

/**
 * \brief Memory information.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Start address of the memory region. */
    uint64_t startAddress;
    /** \brief Size of the memory region. */
    uint64_t size;
} CUDBGMemoryInfo;
#pragma pack(pop)


/*------------------------ Batched device info support -----------------------*/

/**
 * \brief Device info query type.
 * \ingroup READ
 */
/* uint32_t sized enum */
typedef enum {
    /** \brief Request state information for all valid SMs/Warps/Lanes. */
    CUDBG_RESPONSE_TYPE_FULL,

    /** \brief Request state information for all changed SMs/Warps/Lanes since the last call.
     * It's safe to always use this type, the API will respond with the full information when
     * necessary. */
    CUDBG_RESPONSE_TYPE_UPDATE,

    /** \brief Unknown device info query type. */
    /* Force sizeof(CUDBGDeviceInfoQueryType_t)==4 */
    CUDBG_RESPONSE_TYPE_UNKNOWN = 0xFFFFFFFFU,
} CUDBGDeviceInfoQueryType_t;

/**
 * \brief Device-level attributes.
 * \ingroup READ
 */
/* uint32_t sized enum */
typedef enum {
    /** \brief Mask of updated SMs reported by this response.
     * Optional: Yes, assume all 1's if absent.
     * Size: Number of SMs-sized bitmask, rounded up to be divisible by 8. */
    CUDBG_DEVICE_ATTRIBUTE_SM_UPDATE_MASK    = 0,
    /** \brief Mask of SMs with any valid warp.
     * Optional: No, always returned by the API.
     * Size: Number of SMs-sized bitmask, rounded up to be divisible by 8. */
    CUDBG_DEVICE_ATTRIBUTE_SM_ACTIVE_MASK    = 1,
    /** \brief Mask of SMs with any warps with exceptions.
     * Optional: Yes, assume all 0's if absent.
     * Size: Number of SMs-sized bitmask, rounded up to be divisible by 8. */
    CUDBG_DEVICE_ATTRIBUTE_SM_EXCEPTION_MASK = 2,

    /** \brief Device attributes count. */
    CUDBG_DEVICE_ATTRIBUTE_COUNT,
} CUDBGDeviceInfoAttribute_t;

/**
 * \brief SM-level attributes.
 * \ingroup READ
 */
/* uint32_t sized enum */
typedef enum {
    /** \brief Mask of updated warps reported by this response.
     * Optional: Yes, assume all 1's if absent.
     * Size: uint64_t. */
    CUDBG_SM_ATTRIBUTE_WARP_UPDATE_MASK = 0,

    /** \brief SM attributes count. */
    CUDBG_SM_ATTRIBUTE_COUNT,
} CUDBGSMInfoAttribute_t;

/**
 * \brief Warp-level attributes.
 * \ingroup READ
 */
/* uint32_t sized enum */
typedef enum {
    /** \brief Mask of updated lanes reported by this response.
     * Optional: Yes, assume all 1's if absent.
     * Size: uint32_t. */
    CUDBG_WARP_ATTRIBUTE_LANE_UPDATE_MASK                   = 0,
    /** \brief Signals whether the attribute flags field is present on the lane level for this warp.
     * Optional: Yes, assume no lane attributes for this warp if absent.
     * Size: 0 (doesn't have an associated warp-level field). */
    CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES                    = 1,
    /** \brief CUDBGException_t for this warp.
     * Optional: Yes, assume CUDBG_EXCEPTION_NONE if absent.
     * Size: uint32_t. */
    CUDBG_WARP_ATTRIBUTE_EXCEPTION                          = 2,
    /** \brief Error PC for this warp.
     * Optional: Yes, assume no error PC is available if absent.
     * Size: uint64_t. */
    CUDBG_WARP_ATTRIBUTE_ERRORPC                            = 3,
    /** \brief Cluster index for this warp.
     * Optional: Yes if warp is not in a cluster.
     * Size: CuDim3. */
    CUDBG_WARP_ATTRIBUTE_CLUSTERIDX                         = 4,
    /** \brief Cluster dimensions for this warp.
     * Optional: Yes if warp is not in a cluster.
     * Size: CuDim3. */
    CUDBG_WARP_ATTRIBUTE_CLUSTERDIM                         = 5,
    /** \brief For cluster exceptions, this represents the target block index handling
     * cluster requests.
     * Optional: Yes, assume no block index is available if absent.
     * Size: CuDim3. */
    CUDBG_WARP_ATTRIBUTE_CLUSTER_EXCEPTION_TARGET_BLOCK_IDX = 6,
    /** \brief Lane mask showing threads that are in a syscall.
     * Optional: Yes, use readSyscallCallDepth() if this attribute is not present.
     * Size: uint32_t. */
    CUDBG_WARP_ATTRIBUTE_IN_SYSCALL_LANES                   = 7,
    /** \brief Breakpoint handle of a broken warp (of type CUDBGBreakpointHandle)
     * Optional: Yes, only present for broken warps
     * Size: CUDBGBreakpointHandle */
    CUDBG_WARP_ATTRIBUTE_HIT_BREAKPOINT_HANDLE              = 8,

    /** \brief Warp attributes count. */
    CUDBG_WARP_ATTRIBUTE_COUNT,
} CUDBGWarpInfoAttribute_t;

/**
 * \brief Lane-level (thread-level) attributes.
 * \ingroup READ
 */
/* uint32_t sized enum */
typedef enum {
    /** \brief Lane (thread) attributes count. */
    CUDBG_LANE_ATTRIBUTE_COUNT,
} CUDBGLaneInfoAttribute_t;

/**
 * \brief Sizes of the various structs returned by the batched device update APIs.
 * No explicit version field - implied by debugAPI major.minor.revision.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Required buffer size. */
    uint32_t requiredBufferSize;

    /** \brief Device info size. */
    uint32_t deviceInfoSize;
    /** \brief Device info attribute sizes. */
    uint32_t deviceInfoAttributeSizes[32];

    /** \brief SM info size. */
    uint32_t smInfoSize;
    /** \brief SM info attribute sizes. */
    uint32_t smInfoAttributeSizes[32];

    /** \brief Warp info size. */
    uint32_t warpInfoSize;
    /** \brief Warp info attribute sizes. */
    uint32_t warpInfoAttributeSizes[32];

    /** \brief Lane (thread) info size. */
    uint32_t laneInfoSize;
    /** \brief Lane (thread) info attribute sizes. */
    uint32_t laneInfoAttributeSizes[32];
} CUDBGDeviceInfoSizes;
#pragma pack(pop)

/** \brief Device-level information.
 * This is the first element in the deviceInfoBuffer, and is always present.
 * getDeviceInfo() takes a deviceId as input, so no need to explicitly pass it back here.
 * Only "valid & updated" SMs/Warps/Lanes are included in the buffer, which allows us to determine
 * indexes without having to encode an explicit ID field in the following buffer datastructures.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Response type. */
    CUDBGDeviceInfoQueryType_t responseType;

    /** \brief Bitmask of CUDBGDeviceInfoAttribute_t enums for a device. */
    uint32_t deviceAttributeFlags;

    /* This struct is not extensible. New elements are added as attributes instead. */
    /* Attributes matching the flags bitmask above are appended after this struct in the buffer. */
} CUDBGDeviceInfo;
#pragma pack(pop)

/**
 * \brief SM-level information.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Valid warps mask. */
    uint64_t warpValidMask;
    /** \brief Broken warps mask. */
    uint64_t warpBrokenMask;

    /** \brief Bitmask of CUDBGSmInfoAttribute_t enums for a SM. */
    uint32_t smAttributeFlags;

    /* This struct is not extensible. New elements are added as attributes instead. */
    /* Attributes matching the flags bitmask above are appended after this struct in the buffer. */
} CUDBGSMInfo;
#pragma pack(pop)

/**
 * \brief Warp-level information.
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Grid ID. */
    uint64_t gridId;

    /** \brief Block index. */
    CuDim3 blockIdx;
    /** \brief Base thread index (index of the first thread in the warp).
     * Indices of all other threads in the warp can be calculated by monotonically increasing
     * the coordinates of the base thread index and wrapping around the block dimensions. */
    CuDim3 baseThreadIdx;

    /** \brief Valid lanes (threads) mask. */
    uint32_t validLanes;
    /** \brief Active lanes (threads) mask. */
    uint32_t activeLanes;

    /** \brief Bitmask of CUDBGWarpInfoAttribute_t enums for warps and their lanes. */
    uint32_t warpAttributeFlags;

    /* This struct is not extensible. New elements are added as attributes instead. */
    /* Attributes matching the flags bitmask above are appended after this struct in the buffer. */
} CUDBGWarpInfo;
#pragma pack(pop)

/* Represents a Lane */
#pragma pack(push, 1)
typedef struct {
    /** \brief (VA) PC of the thread. */
    uint64_t virtualPC;

    /* Optional: present only if CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES bit
       is set in CUDBGWarpInfo::warpAttributeFlags. Any additional data is
       appended here after this.

       uint32_t laneAttributeFlags;
     */
} CUDBGLaneInfo;
#pragma pack(pop)


/*------------------------ Coredump/snapshot support -------------------------*/

/**
 * \brief Coredump generation flags.
 * \ingroup READ
 */
typedef enum {
    /** \brief Default flags. */
    CUDBG_COREDUMP_DEFAULT_FLAGS                = 0,
    /** \brief Skip dumping non-relocated ELF images. */
    CUDBG_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = (1 << 0),
    /** \brief Skip dumping global memory. */
    CUDBG_COREDUMP_SKIP_GLOBAL_MEMORY           = (1 << 1),
    /** \brief Skip dumping shared memory. */
    CUDBG_COREDUMP_SKIP_SHARED_MEMORY           = (1 << 2),
    /** \brief Skip dumping local memory. */
    CUDBG_COREDUMP_SKIP_LOCAL_MEMORY            = (1 << 3),
    /* The value used to be SKIP_ABORT, but it's impossible to change this behavior.  */
    /* DEPRECATED_VALUE_DO_NOT_USE              = (1 << 4), */
    /** \brief Skip dumping constant bank memory. */
    CUDBG_COREDUMP_SKIP_CONSTBANK_MEMORY        = (1 << 5),
    /** \brief Compress the coredump with gzip. */
    CUDBG_COREDUMP_GZIP_COMPRESS                = (1 << 6),
    /** \brief Only include data for contexts which have encountered an exception.
     *
     * If this flag is used and there are no faulted contexts on any device, then generateCoredump()
     * method will return CUDBG_ERROR_INVALID_CONTEXT error code.
     * Contexts that have warps at breakpoints count as faulted.
     * */
    CUDBG_COREDUMP_FAULTED_CONTEXTS_ONLY        = (1 << 7),

    /** \brief Lightweight flags. */

    /* clang-format off */
    CUDBG_COREDUMP_LIGHTWEIGHT_FLAGS = CUDBG_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES
                                       | CUDBG_COREDUMP_SKIP_GLOBAL_MEMORY
                                       | CUDBG_COREDUMP_SKIP_SHARED_MEMORY
                                       | CUDBG_COREDUMP_SKIP_LOCAL_MEMORY
                                       | CUDBG_COREDUMP_SKIP_CONSTBANK_MEMORY
    /* clang-format on */
} CUDBGCoredumpGenerationFlags;


/*---------------- CBU (Convergence Barrier Unit) state --------------------- */

/**
 * \brief Thread state in the CBU (Convergence Barrier Unit).
 * \ingroup READ
 */
typedef enum {
    /* Force sizeof(CUDBGCbuThreadState)==4 */
    CUDBG_CBU_THREAD_STATE_INVALID = 0xFFFFFFFFU,
    CUDBG_CBU_THREAD_STATE_EXITED  = 0,
    CUDBG_CBU_THREAD_STATE_READY,
    CUDBG_CBU_THREAD_STATE_YIELDED,
    CUDBG_CBU_THREAD_STATE_SLEEP,
    CUDBG_CBU_THREAD_STATE_SLEEPYIELD,
    CUDBG_CBU_THREAD_STATE_READYATNEXT,
    CUDBG_CBU_THREAD_STATE_BLOCKEDPLUS,
    CUDBG_CBU_THREAD_STATE_BLOCKEDALL,
    CUDBG_CBU_THREAD_STATE_BLOCKEDCOLLECTIVE,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB0,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB1,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB2,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB3,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB4,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB5,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB6,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB7,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB8,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB9,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB10,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB11,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB12,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB13,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB14,
    CUDBG_CBU_THREAD_STATE_BLOCKEDB15,
} CUDBGCbuThreadState;

/**
 * \brief Warp state in the CBU (Convergence Barrier Unit).
 * \ingroup READ
 */
#pragma pack(push, 1)
typedef struct {
    /** \brief Active threads mask. */
    uint32_t activeMask;
    /** \brief Exited threads mask. */
    uint32_t exitedMask;
    /** \brief Mask of threads that are part of a collective operation. */
    uint32_t collectiveMask;
    /** \brief Convergence barrier participation masks. */
    uint32_t barrierMasks[CUDBG_MAX_WARP_BARRIERS];
    /** \brief Thread state for each warp lane. */
    CUDBGCbuThreadState threadState[CUDBG_MAX_LANES];
} CUDBGCbuWarpState;
#pragma pack(pop)

/**
 * \brief CUDA Log severity level.
 * \ingroup READ
 */
typedef enum {
    /** \brief Invalid log level. */
    CUDBG_CUDA_LOG_LEVEL_INVALID = 0xFFFFFFFFU,
    /** \brief Error log level, matches CU_LOG_LEVEL_ERROR. */
    CUDBG_CUDA_LOG_LEVEL_ERROR   = 0,
    /** \brief Warning log level, matches CU_LOG_LEVEL_WARNING. */
    CUDBG_CUDA_LOG_LEVEL_WARNING = 1,
} CUDBGCudaLogLevel;

/**
 * \brief CUDA barrier scope.
 * \ingroup READ
 */
typedef enum {
    /** \brief Invalid barrier scope. */
    CUDBG_BARRIER_SCOPE_INVALID    = 0xFFFFFFFFU,
    /** \brief No barrier. */
    CUDBG_BARRIER_SCOPE_NONE       = 0,
    /** \brief Warp-wide barrier. */
    CUDBG_BARRIER_SCOPE_WARP       = 1,
    /** \brief Warp group-wide barrier. */
    CUDBG_BARRIER_SCOPE_WARP_GROUP = 2,
    /** \brief Block-wide barrier. */
    CUDBG_BARRIER_SCOPE_BLOCK      = 3,
    /** \brief Cluster-wide barrier. */
    CUDBG_BARRIER_SCOPE_CLUSTER    = 4,
    /** \brief Kernel-wide barrier. */
    CUDBG_BARRIER_SCOPE_KERNEL     = 5,
} CUDBGBarrierScope;

#pragma pack(push, 1)
/**
 * \brief CUDA Log Message for API version 12.9.
 * \note DEPRECATED: Use CUDBGCudaLogMessage instead.
 * For documentation of individual fields, see the documentation of the CUDBGCudaLogMessage struct
 * instead.
 * \ingroup READ
 * \sa CUDBGCudaLogMessage
 */
typedef struct {
    /** \brief The timestamp the log message was received at, in nanoseconds since the Unix epoch.
     */
    uint64_t unixTimestampNs;
    /** \brief The OS thread ID of the thread that generated the log message.
     * \note This is a *possibly truncated* native OS thread ID. Use CUDBGCudaLogMessage instead
     * for the full value. */
    uint32_t osThreadId;
    /** \brief The log severity level of the message. */
    CUDBGCudaLogLevel logLevel;
    /** \brief The log message string. */
    char message[CUDBG_MAX_LOG_LEN];
} CUDBGCudaLogMessage129;
#pragma pack(pop)

#pragma pack(push, 1)
/**
 * \brief CUDA Log Message.
 * \ingroup READ
 */
typedef struct {
    /** \brief The timestamp the log message was received at, in nanoseconds since the Unix epoch.
     */
    uint64_t unixTimestampNs;
    /** \brief The native OS thread ID of the thread that generated the log message. */
    uint64_t osThreadId;
    /** \brief The log severity level of the message. */
    CUDBGCudaLogLevel logLevel;
    /** \brief The index of the ruleset that matched the log message. */
    uint32_t matchedRulesetIndex;
    /** \brief The rule index of the rule that matched the log message. */
    uint32_t matchedRuleIndex;
    /** \brief The log message string. */
    char message[CUDBG_MAX_LOG_LEN];
} CUDBGCudaLogMessage;
#pragma pack(pop)

/**
 * \brief CUDA Log level filter.
 * \ingroup READ
 */
typedef enum {
    /** \brief Any log level */
    CUDBG_CUDA_LOG_LEVEL_FILTER_ANY     = 0xFFFFFFFFU,
    /** \brief Error log level */
    CUDBG_CUDA_LOG_LEVEL_FILTER_ERROR   = 0,
    /** \brief Warning log level */
    CUDBG_CUDA_LOG_LEVEL_FILTER_WARNING = 1,
} CUDBGCudaLogLevelFilter;

/**
 * \brief CUDA Log filtering rule action.
 * \ingroup READ
 */
typedef enum {
    /** \brief Exclude the log message */
    CUDBG_CUDA_LOG_RULE_ACTION_EXCLUDE    = 0xFFFFFFFFU,
    /** \brief Send the log message asynchronously */
    CUDBG_CUDA_LOG_RULE_ACTION_SEND_ASYNC = 0,
    /** \brief Send the log message synchronously */
    CUDBG_CUDA_LOG_RULE_ACTION_SEND_SYNC  = 1,
} CUDBGCudaLogRuleAction;

#pragma pack(push, 1)
/**
 * \brief CUDA Log filtering rule.
 * \ingroup READ
 */
typedef struct {
    /** \brief Action to take for the log message */
    CUDBGCudaLogRuleAction action;
    /** \brief A particular log level (or any) */
    CUDBGCudaLogLevelFilter logLevelFilter;
    /** \brief A particular OS thread ID or for any thread if 0.
     * The user should only set this to TID values seen via consumeCudaLogs(). */
    uint64_t osThreadIdFilter;
    /** \brief A particular message content or any if NULL
     * \note This field accepts C++ modified ECMAScript regex syntax (the std::regex default) */
    const char* messageFilterRegex;
} CUDBGCudaLogRule;
#pragma pack(pop)

/**
 * \brief GPU debugger breakpoint handle.
 * Valid breakpoint handles will grow from 1. Handle values of removed breakpoints can be reused in
 * some cases. It is not guaranteed that new (non-reused) breakpoint handles will always be
 * increasing by 1, there can be gaps. A valid handle always corresponds to a single inserted
 * breakpoint, and each breakpoint can only have one handle. Handles with the highest bit set are
 * reserved for special purposes and can correspond to multiple breakpoints.
 * \ingroup BP
 */
typedef uint64_t CUDBGBreakpointHandle;
/**
 * \brief Invalid breakpoint handle.
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_INVALID              (0ULL)
/**
 * \brief Special breakpoint handle - hard-coded trap.
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_TRAP                 (0xFFFFFFFFFFFFFFFFULL)
/**
 * \brief Special breakpoint handle - breakpoint inserted with setBreakpoint (handle-less).
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_LEGACY               (0xFFFFFFFFFFFFFFFEULL)
/**
 * \brief Special breakpoint handle - internal breakpoint (e.g. used internally for stepping)
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_INTERNAL             (0xFFFFFFFFFFFFFFFDULL)
/**
 * \brief Break On Launch breakpoint handle - this breakpoint can be enabled/disabled with
 * enableBreakpoint/disableBreakpoint. It is hit once every time a kernel is launched. It can be hit
 * by one or more warps, but only once. Only supports CUDA Kernel launches (both host-side and
 * device-side launches).
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_BREAK_ON_LAUNCH      (0xFFFFFFFFFFFFFFFCULL)
/**
 * \brief Special breakpoint handle - represents all non-special breakpoints that were inserted by
 * the API user. This handle can be used to enable, disable, and remove all added breakpoints in a
 * single call.
 *
 * It can only be used as a handle in \ref CUDBGAPI_st::removeBreakpoint,
 * \ref CUDBGAPI_st::enableBreakpoint, \ref CUDBGAPI_st::disableBreakpoint functions.
 *
 * If this handle is used and there are no breakpoints added by the user, the aforementioned API
 * functions do not fail, instead the operation acts on no breakpoints, so calling e.g.
 * CUDBGAPI_st::removeBreakpoint with this handle is fine if there are no breakpoints added by the
 * user. They may still fail for other reasons.
 *
 * \note The set of breakpoints represented by this handle does not include the
 * CUDBG_BREAKPOINT_HANDLE_BREAK_ON_LAUNCH breakpoint.
 *
 * \ingroup BP
 */
#define CUDBG_BREAKPOINT_HANDLE_ALL_USER_BREAKPOINTS (0xFFFFFFFFFFFFFFFBULL)

/*---------------------------------- Exports ---------------------------------*/

/**
 * \brief CUDA Debugger API methods.
 * \ingroup GENERAL
 */
typedef const struct CUDBGAPI_st* CUDBGAPI;

/**
 * \brief Get the API associated with the major/minor/revision version numbers.
 *
 * \ingroup GENERAL
 *
 * \param major - the major version number
 * \param minor - the minor version number
 * \param rev   - the revision version number
 * \param api   - the pointer to the API
 *
 * \return CUDBG_ERROR_INVALID_ARGS,
 * \return CUDBG_SUCCESS,
 * \return CUDBG_ERROR_INCOMPATIBLE_API
 *
 * \sa cudbgGetAPIVersion
 */
CUDBGResult cudbgGetAPI(uint32_t major, uint32_t minor, uint32_t rev, CUDBGAPI* api);

/**
 * \brief Get the API version supported by the CUDA driver.
 *
 * \ingroup GENERAL
 *
 * \param major - the major version number
 * \param minor - the minor version number
 * \param rev   - the revision version number
 *
 * \return CUDBG_ERROR_INVALID_ARGS,
 * \return CUDBG_SUCCESS
 *
 * \sa cudbgGetAPI
 */
CUDBGResult cudbgGetAPIVersion(uint32_t* major, uint32_t* minor, uint32_t* rev);

/**
 * \brief Empty global function that gets called before CUDA driver initialization.
 * The API client can set a breakpoint on this function to be notified before the CUDA driver starts
 * its initialization.
 * \ingroup GENERAL
 */
void CUDBG_PRE_INIT();

/**
 * \brief Initialize the CUDA Debugger API.
 * Remotely called by the API client during the attach procedure to initialize the debugger API.
 * \ingroup GENERAL
 */
void cudbgApiInit(uint32_t arg);
/**
 * \brief Attach to a CUDA application.
 * Remotely called by the API client during the attach procedure to attach to the CUDA application.
 * \ingroup GENERAL
 */
void cudbgApiAttach(void);
/**
 * \brief Detach from a CUDA application.
 * Remotely called by the API client during the detach procedure to detach from the CUDA
 * application.
 * \ingroup GENERAL
 */
void cudbgApiDetach(void);
/**
 * \brief Report a driver API error.
 * The API client can set a breakpoint on this function to be notified about driver API errors.
 * \ingroup GENERAL
 */
void CUDBG_REPORT_DRIVER_API_ERROR(void);
/**
 * \brief Report a driver internal error.
 * The API client can set a breakpoint on this function to be notified about driver internal errors.
 * \ingroup GENERAL
 */
void CUDBG_REPORT_DRIVER_INTERNAL_ERROR(void);

extern uint32_t CUDBG_IPC_FLAG_NAME;
extern uint32_t CUDBG_RPC_ENABLED;
extern uint32_t CUDBG_APICLIENT_PID;
extern uint32_t CUDBG_I_AM_DEBUGGER;
extern uint32_t CUDBG_DEBUGGER_INITIALIZED;
extern uint32_t CUDBG_APICLIENT_REVISION;
extern uint32_t CUDBG_SESSION_ID;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_CODE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR;
extern uint32_t CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_NAME_SIZE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_NAME_ADDR;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_STRING_SIZE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_STRING_ADDR;
extern uint64_t CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE;
extern uint32_t CUDBG_ATTACH_HANDLER_AVAILABLE;
extern uint32_t CUDBG_ENABLE_LAUNCH_BLOCKING;
extern uint32_t CUDBG_RESUME_FOR_ATTACH_DETACH;
extern uint32_t CUDBG_REPORT_DRIVER_API_ERROR_FLAGS;
extern uint32_t CUDBG_DEBUGGER_CAPABILITIES;
extern int32_t CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD;

/**
 * \brief Report that the attach procedure has finished.
 * The API client can set a breakpoint on this function to be notified that the attach procedure has
 * finished.
 * \ingroup GENERAL
 */
void CUDBG_REPORT_ATTACH_PROCEDURE_FINISHED(void);

/* Deprecated:
 * CUDBGResult cudbgMain(int apiClientPid, uint32_t apiClientRevision, int sessionId,
 *                       int attachState, int attachEventInitialized, int writeFd, int detachFd,
 *                       int attachStubInUse, int enablePreemptionDebugging);
 *
 * extern uint32_t CUDBG_DETACH_SUSPENDED_DEVICES_MASK;
 *
 * extern uint32_t CUDBG_ENABLE_INTEGRATED_MEMCHECK;
 *
 * extern uint32_t CUDBG_ENABLE_PREEMPTION_DEBUGGING;
 */

/**
 * \brief CUDA debugger API methods.
 */
struct CUDBGAPI_st {
    /* Initialization */

    /**
     * \fn CUDBGAPI_st::initialize
     * \brief Initialize the API.
     *
     * setNotifyNewEventCallback() and getSupportedDebuggerCapabilities() can be called before
     * initialize().
     * If no CUDA devices are detected on the system, CUDBG_ERROR_NO_DEVICE_AVAILABLE is returned.
     *
     * Since CUDA 3.0.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_DEVICE_AVAILABLE
     *
     * \sa finalize
     */
    CUDBGResult (*initialize)(void);

    /**
     * \fn CUDBGAPI_st::finalize
     * \brief Finalize the API, shutting down the debugging session.
     *
     * Since CUDA 3.0.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa initialize
     */
    CUDBGResult (*finalize)(void);

    /* Device Execution Control */

    /**
     * \fn CUDBGAPI_st::suspendDevice
     * \brief Suspends a running CUDA device.
     *
     * Using this method is discouraged, use suspendAllDevices() instead to avoid race conditions.
     * The device has to be suspended in order to execute most operations on it.
     * CUDBG_ERROR_SUSPENDED_DEVICE is returned if the device is already suspended.
     *
     * Since CUDA 3.0.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_SUSPENDED_DEVICE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa suspendAllDevices
     */
    CUDBGResult (*suspendDevice)(uint32_t dev);

    /**
     * \fn CUDBGAPI_st::resumeDevice
     * \brief Resume a suspended CUDA device.
     *
     * Using this method is discouraged, use resumeAllDevices() instead to avoid race conditions.
     * This method has no effect if the device is already running.
     *
     * Since CUDA 3.0.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa resumeAllDevices
     */
    CUDBGResult (*resumeDevice)(uint32_t dev);

    /**
     * \fn CUDBGAPI_st::singleStepWarp40
     * \brief Single step an individual warp on a suspended CUDA device.
     *
     * Behaves like singleStepWarp41 without the output warpMask parameter.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 4.1: Use \ref singleStepWarp instead.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa singleStepWarp
     */
    CUDBGResult (*singleStepWarp40)(uint32_t dev, uint32_t sm, uint32_t wp);

    /* Breakpoints */

    /**
     * \fn CUDBGAPI_st::setBreakpoint31
     * \brief Set a breakpoint at the given instruction address.
     *
     * Behaves like setBreakpoint but tries to automatically find a device for the given address.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.2: Use \ref setBreakpoint instead.
     *
     * \ingroup BP
     *
     * \param[in] addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setBreakpoint
     */
    CUDBGResult (*setBreakpoint31)(uint64_t addr);

    /**
     * \fn CUDBGAPI_st::unsetBreakpoint31
     * \brief Unset a breakpoint at the given instruction address.
     *
     * Behaves like unsetBreakpoint but tries to automatically find a device for the given address.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.2: Use \ref unsetBreakpoint instead.
     *
     * \ingroup BP
     *
     * \param[in] addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa unsetBreakpoint
     */
    CUDBGResult (*unsetBreakpoint31)(uint64_t addr);

    /* Device State Inspection */

    /**
     * \fn CUDBGAPI_st::readGridId50
     * \brief Read the CUDA grid index running on a valid warp.
     *
     * Behaves like readGridId but truncates the grid ID to 32bit. This is incompatible with some
     * grid IDs like those used by the OptiX applications.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 5.5: Use \ref readGridId instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] gridId - the returned CUDA grid index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readGridId
     */
    CUDBGResult (*readGridId50)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t* gridId);

    /**
     * \fn CUDBGAPI_st::readBlockIdx32
     * \brief Read the two-dimensional CUDA block index running on a valid warp.
     *
     * Behaves like readBlockIdx but doesn't return the z dimension.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref readBlockIdx instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] blockIdx - the returned CUDA block index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readBlockIdx
     */
    CUDBGResult (*readBlockIdx32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2* blockIdx);

    /**
     * \fn CUDBGAPI_st::readThreadIdx
     * \brief Read the CUDA thread index running on valid thread.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] threadIdx - the returned CUDA thread index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readThreadIdx)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3* threadIdx);

    /**
     * \fn CUDBGAPI_st::readBrokenWarps
     * \brief Read the bitmask of warps that are at a breakpoint on a given SM.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[out] brokenWarpsMask - the returned bitmask of broken warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readBrokenWarps)(uint32_t dev, uint32_t sm, uint64_t* brokenWarpsMask);

    /**
     * \fn CUDBGAPI_st::readValidWarps
     * \brief Read the bitmask of valid warps on a given SM.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[out] validWarpsMask - the returned bitmask of valid warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidLanes
     */
    CUDBGResult (*readValidWarps)(uint32_t dev, uint32_t sm, uint64_t* validWarpsMask);

    /**
     * \fn CUDBGAPI_st::readValidLanes
     * \brief Read the lane bitmask of valid threads on a given warp.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] validLanesMask - the returned bitmask of valid threads
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidWarps
     */
    CUDBGResult (*readValidLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t* validLanesMask);

    /**
     * \fn CUDBGAPI_st::readActiveLanes
     * \brief Read the lane bitmask of active threads on a valid warp.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] activeLanesMask - the returned bitmask of active threads
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readActiveLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t* activeLanesMask);

    /**
     * \fn CUDBGAPI_st::readCodeMemory
     * \brief Read content at address in the code memory segment.
     *
     * It is generally not necessary to call this function - instead, the same memory could be read
     * from the ELF module images received from the API.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readCodeMemory)(uint32_t dev, uint64_t addr, void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readConstMemory129
     * \brief Read content at address in the constant memory segment.
     *
     * Behaves exactly like readGlobalMemory.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 13.0: Use \ref readGlobalMemory instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readGlobalMemory
     */
    CUDBGResult (*readConstMemory129)(uint32_t dev, uint64_t addr, void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readGlobalMemory31
     * \brief Read content at address in the global memory segment.
     *
     * Behaves like readGenericMemory() with sm, wp, ln == 0. This makes this method not at all
     * useful.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.2: Use \ref readGlobalMemory instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readGlobalMemory
     */
    CUDBGResult (*readGlobalMemory31)(uint32_t dev, uint64_t addr, void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readParamMemory
     * \brief Read content at address in the param memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readParamMemory)(uint32_t dev,
                                   uint32_t sm,
                                   uint32_t wp,
                                   uint64_t addr,
                                   void* buf,
                                   uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readSharedMemory
     * \brief Read content at address in the shared memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readTextureMemory
     */
    CUDBGResult (*readSharedMemory)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    uint64_t addr,
                                    void* buf,
                                    uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readLocalMemory
     * \brief Read content at address in the local memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readLocalMemory)(uint32_t dev,
                                   uint32_t sm,
                                   uint32_t wp,
                                   uint32_t ln,
                                   uint64_t addr,
                                   void* buf,
                                   uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readRegister
     * \brief Read content of a hardware register.
     *
     * Note that warps can dynamically change the number of used registers at runtime,
     * readWarpResources() could be used to query that.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] regno - register index
     * \param[out] val - the returned value of the register
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readRegister)(uint32_t dev,
                                uint32_t sm,
                                uint32_t wp,
                                uint32_t ln,
                                uint32_t regno,
                                uint32_t* val);

    /**
     * \fn CUDBGAPI_st::readPC
     * \brief Read the PC offset on the given active thread.
     *
     * The returned PC offset is from the start of the current function. If a function can't be
     * found, the full virtual address is returned.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] pc - the returned PC
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readVirtualPC
     */
    CUDBGResult (*readPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t* pc);

    /**
     * \fn CUDBGAPI_st::readVirtualPC
     * \brief Read the PC (virtual address) on the given active thread.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] pc - the returned PC
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readPC
     */
    CUDBGResult (*readVirtualPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t* pc);

    /**
     * \fn CUDBGAPI_st::readLaneStatus
     * \brief Read the status of the given thread.
     *
     * For specific error values, use readLaneException.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] error - true if there is an error
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readLaneStatus)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, bool* error);

    /* Device State Alteration */

    /**
     * \fn CUDBGAPI_st::writeGlobalMemory31
     * \brief Write to an address in global memory
     *
     * This method is unsupported on Hopper and later architectures. Use newer methods:
     * writeGlobalMemory or writeGenericMemory.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.2: Use \ref writeGlobalMemory instead.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa writeGlobalMemory
     */
    CUDBGResult (*writeGlobalMemory31)(uint32_t dev, uint64_t addr, const void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeParamMemory
     * \brief Write to an address in param memory
     *
     * The destination address range must be within param memory.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writeGenericMemory
     * \sa writeGlobalMemory
     * \sa writeLocalMemory
     * \sa writeSharedMemory
     */
    CUDBGResult (*writeParamMemory)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    uint64_t addr,
                                    const void* buf,
                                    uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeSharedMemory
     * \brief Write to an address in shared memory
     *
     * The destination address range must be within shared memory.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writeGenericMemory
     * \sa writeGlobalMemory
     * \sa writeLocalMemory
     * \sa writeParamMemory
     */
    CUDBGResult (*writeSharedMemory)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint64_t addr,
                                     const void* buf,
                                     uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeLocalMemory
     * \brief Write to an address in local memory
     *
     * The destination address range must be within local memory.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writeGenericMemory
     * \sa writeGlobalMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     */
    CUDBGResult (*writeLocalMemory)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    uint32_t ln,
                                    uint64_t addr,
                                    const void* buf,
                                    uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeRegister
     * \brief Write to a hardware register
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] regno - register number
     * \param[in] val - value
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writePredicates
     * \sa writeUniformPredicates
     * \sa writeUniformRegister
     */
    CUDBGResult (*writeRegister)(uint32_t dev,
                                 uint32_t sm,
                                 uint32_t wp,
                                 uint32_t ln,
                                 uint32_t regno,
                                 uint32_t val);

    /* Grid Properties */

    /**
     * \fn CUDBGAPI_st::getGridDim32
     * \brief Get the dimensions of the given grid.
     *
     * Behaves like getGridDim but doesn't return the z dimension.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref getGridDim instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] gridDim - the returned number of blocks in the grid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridDim
     */
    CUDBGResult (*getGridDim32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2* gridDim);

    /**
     * \fn CUDBGAPI_st::getBlockDim
     * \brief Get the dimensions of the given block.
     *
     * Since CUDA 3.0.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] blockDim - the returned number of threads in the block
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getClusterDim
     * \sa getGridDim
     */
    CUDBGResult (*getBlockDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3* blockDim);

    /**
     * \fn CUDBGAPI_st::getTID
     * \brief Get the ID of the Linux thread hosting the CUDA context active at the given
     * coordinates.
     *
     * This returns a Linux thread ID.
     *
     * Since CUDA 3.0.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] tid - the returned thread id
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridAttributes
     */
    CUDBGResult (*getTID)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t* tid);

    /**
     * \fn CUDBGAPI_st::getElfImage32
     * \brief Get the relocated or non-relocated ELF image and size for the grid on the given
     * device.
     *
     * Behaves like getElfImage but will truncate the image size for cubins larger than 4GiB.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref getElfImage instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] relocated - set to true to specify the relocated ELF image, false otherwise
     * \param[out] elfImage - pointer to the ELF image
     * \param[out] size - size of the ELF image (32 bits)
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getElfImage
     * \sa getElfImageByHandle
     */
    CUDBGResult (*getElfImage32)(uint32_t dev,
                                 uint32_t sm,
                                 uint32_t wp,
                                 bool relocated,
                                 void** elfImage,
                                 uint32_t* size);

    /* Device Properties */

    /**
     * \fn CUDBGAPI_st::getDeviceType
     * \brief Get the string description of the device.
     *
     * Returns CUDBG_ERROR_BUFFER_TOO_SMALL if the provided buffer is not large enough.
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] buf - the destination buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getDeviceName
     * \sa getSMType
     */
    CUDBGResult (*getDeviceType)(uint32_t dev, char* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::getSmType
     * \brief Get the SM type of the device.
     *
     * Returns CUDBG_ERROR_BUFFER_TOO_SMALL if the provided buffer is not large enough.
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] buf - the destination buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getDeviceName
     * \sa getDeviceType
     */
    CUDBGResult (*getSmType)(uint32_t dev, char* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::getNumDevices
     * \brief Get the number of installed CUDA devices.
     *
     * This value is constant within a single session.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[out] numDev - the returned number of devices
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     */
    CUDBGResult (*getNumDevices)(uint32_t* numDev);

    /**
     * \fn CUDBGAPI_st::getNumSMs
     * \brief Get the total number of SMs on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numSMs - the returned number of SMs
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     */
    CUDBGResult (*getNumSMs)(uint32_t dev, uint32_t* numSMs);

    /**
     * \fn CUDBGAPI_st::getNumWarps
     * \brief Get the number of warps per SM on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numWarps - the returned number of warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     */
    CUDBGResult (*getNumWarps)(uint32_t dev, uint32_t* numWarps);

    /**
     * \fn CUDBGAPI_st::getNumLanes
     * \brief Get the number of lanes per warp on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numLanes - the returned number of lanes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     */
    CUDBGResult (*getNumLanes)(uint32_t dev, uint32_t* numLanes);

    /**
     * \fn CUDBGAPI_st::getNumRegisters
     * \brief Get the maximum number of registers per lane on the device.
     *
     * This value is constant within a single session for a given device.
     * Note that the actual number of registers can change per warp, use readWarpResources() to
     * query that number dynamically.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numRegs - the returned number of registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     * \sa readWarpResources
     */
    CUDBGResult (*getNumRegisters)(uint32_t dev, uint32_t* numRegs);

    /* DWARF-related routines */

    /**
     * \fn CUDBGAPI_st::getPhysicalRegister30
     * \brief Get the physical register number(s) assigned to a virtual register name at a given PC,
     * if it's live at that PC.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.1: Do not use.
     *
     * \ingroup DWARF
     *
     * \param[in] pc - Program counter
     * \param[in] reg - virtual register index
     * \param[out] buf - physical register name(s)
     * \param[in] sz - the physical register name buffer size
     * \param[out] numPhysRegs - number of physical register names returned
     * \param[out] regClass - the class of the physical registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getPhysicalRegister30)(uint64_t pc,
                                         char* reg,
                                         uint32_t* buf,
                                         uint32_t sz,
                                         uint32_t* numPhysRegs,
                                         CUDBGRegClass* regClass);

    /**
     * \fn CUDBGAPI_st::disassemble
     * \brief Disassemble instruction at instruction address.
     *
     * This method does not guarantee any specific output format and its result should be treated as
     * plain text.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param[in] dev - device index
     * \param[in] addr - instruction address
     * \param[out] instSize - instruction size (32 or 64 bits)
     * \param[out] buf - disassembled instruction buffer
     * \param[in] sz - disassembled instruction buffer size
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*disassemble)(uint32_t dev, uint64_t addr, uint32_t* instSize, char* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::isDeviceCodeAddress55
     * \brief Determine whether a virtual address resides within device code.
     *
     * Behaves exactly like isDeviceCodeAddress.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref isDeviceCodeAddress instead.
     *
     * \ingroup DWARF
     *
     * \param[in] addr - virtual address
     * \param[out] isDeviceAddress - true if address resides within device code
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa isDeviceCodeAddress
     */
    CUDBGResult (*isDeviceCodeAddress55)(uintptr_t addr, bool* isDeviceAddress);

    /**
     * \fn CUDBGAPI_st::lookupDeviceCodeSymbol
     * \brief Determines whether a symbol represents a function in device code and returns its
     * virtual address.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param[in] symName - symbol name
     * \param[out] symFound - set to true if the symbol is found
     * \param[out] symAddr - the symbol virtual address if found
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*lookupDeviceCodeSymbol)(char* symName, bool* symFound, uintptr_t* symAddr);

    /* Events */

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback31
     * \brief Provides the API with the function to call to notify the debugger of a new application
     * or device event.
     *
     * Behaves like setNotifyNewEventCallback but doesn't return the host thread ID from which the
     * event originates.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.2: Use \ref setNotifyNewEventCallback instead.
     *
     * \ingroup EVENT
     *
     * \param[in] callback - the callback function
     * \param[in] data - a pointer to be passed to the callback when called
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setNotifyNewEventCallback
     */
    CUDBGResult (*setNotifyNewEventCallback31)(CUDBGNotifyNewEventCallback31 callback, void* data);

    /**
     * \fn CUDBGAPI_st::getNextEvent30
     * \brief Copies the next available event in the event queue into 'event' and removes it from
     * the queue.
     *
     * Behaves like getNextEvent but only for SYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.1: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextEvent30)(CUDBGEvent30* event);

    /**
     * \fn CUDBGAPI_st::acknowledgeEvent30
     * \brief Inform the debugger API that synchronous events have been processed.
     *
     * Behaves exactly like acknowledgeSyncEvents (the event parameter is ignored).
     *
     * Since CUDA 3.0.
     *
     * \note DEPRECATED in CUDA 3.1: Use \ref acknowledgeSyncEvents instead.
     *
     * \ingroup EVENT
     *
     * \param[in] event - pointer to the event that has been processed
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE
     *
     * \sa acknowledgeSyncEvents
     */
    CUDBGResult (*acknowledgeEvent30)(CUDBGEvent30* event);

    /* 3.1 Extensions */

    /**
     * \fn CUDBGAPI_st::getGridAttribute
     * \brief Get the value of a grid attribute.
     *
     * See CUDBGAttribute for the list of available attributes.
     *
     * Since CUDA 3.1.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] attr - the attribute
     * \param[out] value - the returned value of the attribute
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_ATTRIBUTE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridAttributes
     */
    CUDBGResult (*getGridAttribute)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    CUDBGAttribute attr,
                                    uint64_t* value);

    /**
     * \fn CUDBGAPI_st::getGridAttributes
     * \brief Get several grid attribute values in a single API call.
     *
     * See CUDBGAttribute for the list of available attributes.
     *
     * Since CUDA 3.1.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] pairs - array of attribute/value pairs
     * \param[in] numPairs - the number of attribute/values pairs in the array
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_ATTRIBUTE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridAttribute
     */
    CUDBGResult (*getGridAttributes)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     CUDBGAttributeValuePair* pairs,
                                     uint32_t numPairs);

    /**
     * \fn CUDBGAPI_st::getPhysicalRegister40
     * \brief Get the physical register number(s) assigned to a virtual register name at a given PC,
     * if it's live at that PC.
     *
     * Instead, the PTX to SASS mappings can be read from the cubin directly.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 4.1: Do not use.
     *
     * \ingroup DWARF
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] pc - Program counter
     * \param[in] reg - virtual register index
     * \param[out] buf - physical register name(s)
     * \param[in] sz - the physical register name buffer size
     * \param[out] numPhysRegs - number of physical register names returned
     * \param[out] regClass - the class of the physical registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getPhysicalRegister40)(uint32_t dev,
                                         uint32_t sm,
                                         uint32_t wp,
                                         uint64_t pc,
                                         char* reg,
                                         uint32_t* buf,
                                         uint32_t sz,
                                         uint32_t* numPhysRegs,
                                         CUDBGRegClass* regClass);

    /**
     * \fn CUDBGAPI_st::readLaneException
     * \brief Read the exception type for a given thread.
     *
     * Since CUDA 3.1.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] exception - the returned exception type
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readLaneException)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint32_t ln,
                                     CUDBGException_t* exception);

    /**
     * \fn CUDBGAPI_st::getNextEvent32
     * \brief Copies the next available event in the event queue into 'event' and removes it from
     * the queue.
     *
     * Behaves like getNextEvent but only for SYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextEvent32)(CUDBGEvent32* event);

    /**
     * \fn CUDBGAPI_st::acknowledgeEvents42
     * \brief Inform the debugger API that synchronous events have been processed.
     *
     * Behaves exactly like acknowledgeSyncEvents.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 5.0: Use \ref acknowledgeSyncEvents instead.
     *
     * \ingroup EVENT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE
     *
     * \sa acknowledgeSyncEvents
     */
    CUDBGResult (*acknowledgeEvents42)(void);

    /* 3.1 - ABI */

    /**
     * \fn CUDBGAPI_st::readCallDepth32
     * \brief Read the call depth (number of calls) for a given warp.
     *
     * Behaves like readCallDepth() for the active thread group.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref readCallDepth instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCallDepth
     */
    CUDBGResult (*readCallDepth32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t* depth);

    /**
     * \fn CUDBGAPI_st::readReturnAddress32
     * \brief Read the return address (offset) for a call level.
     *
     * Behaves like readReturnAddress() for the active thread group.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref readReturnAddress instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] level - the specified call level
     * \param[out] ra - the returned return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readReturnAddress
     */
    CUDBGResult (*readReturnAddress32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t* ra);

    /**
     * \fn CUDBGAPI_st::readVirtualReturnAddress32
     * \brief Read the virtual return address for a call level.
     *
     * Behaves like readVirtualReturnAddress for the active thread group.
     *
     * Since CUDA 3.1.
     *
     * \note DEPRECATED in CUDA 4.0: Use \ref readVirtualReturnAddress instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] level - the specified call level
     * \param[out] ra - the returned virtual return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readVirtualReturnAddress32)(uint32_t dev,
                                              uint32_t sm,
                                              uint32_t wp,
                                              uint32_t level,
                                              uint64_t* ra);

    /* 3.2 Extensions */

    /**
     * \fn CUDBGAPI_st::readGlobalMemory55
     * \brief Read content at address in the global memory segment.
     *
     * Behaves exactly like readGenericMemory().
     *
     * Since CUDA 3.2.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref readGlobalMemory instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readGlobalMemory
     */
    CUDBGResult (*readGlobalMemory55)(uint32_t dev,
                                      uint32_t sm,
                                      uint32_t wp,
                                      uint32_t ln,
                                      uint64_t addr,
                                      void* buf,
                                      uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeGlobalMemory55
     * \brief Write to an address in global memory
     *
     * Use newer methods: writeGlobalMemory or writeGenericMemory.
     *
     * Since CUDA 3.2.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref writeGlobalMemory instead.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa writeGlobalMemory
     */
    CUDBGResult (*writeGlobalMemory55)(uint32_t dev,
                                       uint32_t sm,
                                       uint32_t wp,
                                       uint32_t ln,
                                       uint64_t addr,
                                       const void* buf,
                                       uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readPinnedMemory
     * \brief Read content at pinned address in system memory.
     *
     * Depending on the platform, this method may fail and a platform-specific CPU RAM way of
     * reading memory from the debuggee must be used (e.g. ptrace).
     *
     * Since CUDA 3.2.
     *
     * \ingroup READ
     *
     * \param[in] addr - system memory address
     * \param[out] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readPinnedMemory)(uint64_t addr, void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writePinnedMemory
     * \brief Write to a pinned memory address
     *
     * It's not possible to access an ambiguous access allocated on several devices that don't
     * support UVA.
     *
     * Since CUDA 3.2.
     *
     * \ingroup WRITE
     *
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readPinnedMemory
     */
    CUDBGResult (*writePinnedMemory)(uint64_t addr, const void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::setBreakpoint
     * \brief Set a breakpoint at the given instruction address for the given device.
     *
     * Before setting a breakpoint, getAdjustedCodeAddress() should be called to get the adjusted
     * breakpoint address.
     *
     * Since CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param[in] dev - device index
     * \param[in] addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa unsetBreakpoint
     */
    CUDBGResult (*setBreakpoint)(uint32_t dev, uint64_t addr);

    /**
     * \fn CUDBGAPI_st::unsetBreakpoint
     * \brief Unset a breakpoint at the given instruction address for the given device.
     *
     * Since CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param[in] dev - device index
     * \param[in] addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setBreakpoint
     */
    CUDBGResult (*unsetBreakpoint)(uint32_t dev, uint64_t addr);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback40
     * \brief Provides the API with the function to call to notify the debugger of a new application
     * or device event.
     *
     * Behaves like setNotifyNewEventCallback but doesn't allow passing in the user data pointer.
     *
     * Since CUDA 3.2.
     *
     * \note DEPRECATED in CUDA 4.1: Use \ref setNotifyNewEventCallback instead.
     *
     * \ingroup EVENT
     *
     * \param[in] callback - the callback function
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setNotifyNewEventCallback
     */
    CUDBGResult (*setNotifyNewEventCallback40)(CUDBGNotifyNewEventCallback40 callback);

    /* 4.0 Extensions */

    /**
     * \fn CUDBGAPI_st::getNextEvent42
     * \brief Copies the next available event in the event queue into 'event' and removes it from
     * the queue.
     *
     * Behaves like getNextEvent but only for SYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 4.0.
     *
     * \note DEPRECATED in CUDA 5.0: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextEvent42)(CUDBGEvent42* event);

    /**
     * \fn CUDBGAPI_st::readTextureMemory
     * \brief This method is no longer supported since CUDA 12.0.
     *
     * Will always return CUDBG_ERROR_NOT_SUPPORTED.
     *
     * Since CUDA 4.0.
     *
     * \note DEPRECATED in CUDA 12.0: Do not use.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] id - texture id (the value of DW_AT_location attribute in the relocated ELF image)
     * \param[in] dim - texture dimension (1 to 4)
     * \param[in] coords - array of coordinates of size dim
     * \param[out] buf - result buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*readTextureMemory)(uint32_t dev,
                                     uint32_t vsm,
                                     uint32_t wp,
                                     uint32_t id,
                                     uint32_t dim,
                                     uint32_t* coords,
                                     void* buf,
                                     uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readBlockIdx
     * \brief Read the CUDA block index running on a valid warp.
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] blockIdx - the returned CUDA block index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readBlockIdx)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3* blockIdx);

    /**
     * \fn CUDBGAPI_st::getGridDim
     * \brief Get the dimensions in blocks of the given grid.
     *
     * Since CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] gridDim - the dimensions of the grid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getBlockDim
     * \sa getClusterDim
     */
    CUDBGResult (*getGridDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3* gridDim);

    /**
     * \fn CUDBGAPI_st::readCallDepth
     * \brief Read the call depth (number of calls) for a given thread.
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readReturnAddress
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t* depth);

    /**
     * \fn CUDBGAPI_st::readReturnAddress
     * \brief Read the return address (offset) for a call level.
     *
     * The returned return address is an offset from the start of the current function. If a
     * function can't be found, the full virtual address is returned.
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] level - the specified call level
     * \param[out] ra - the returned return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCallDepth
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readReturnAddress)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint32_t ln,
                                     uint32_t level,
                                     uint64_t* ra);

    /**
     * \fn CUDBGAPI_st::readVirtualReturnAddress
     * \brief Read the virtual return address for a call level.
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] level - the specified call level
     * \param[out] ra - the returned virtual return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCallDepth
     * \sa readReturnAddress
     */
    CUDBGResult (*readVirtualReturnAddress)(uint32_t dev,
                                            uint32_t sm,
                                            uint32_t wp,
                                            uint32_t ln,
                                            uint32_t level,
                                            uint64_t* ra);

    /**
     * \fn CUDBGAPI_st::getElfImage
     * \brief Get the relocated or non-relocated ELF image and size for the grid on the given
     * device.
     *
     * Since CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] relocated - set to true to specify the relocated ELF image, false otherwise
     * \param[out] elfImage - pointer to the ELF image
     * \param[out] size - size of the ELF image (64 bits)
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getElfImageByHandle
     * \sa getLoadedFunctionInfo
     */
    CUDBGResult (*getElfImage)(uint32_t dev,
                               uint32_t sm,
                               uint32_t wp,
                               bool relocated,
                               void** elfImage,
                               uint64_t* size);

    /* 4.1 Extensions */

    /**
     * \fn CUDBGAPI_st::getHostAddrFromDeviceAddr
     * \brief Given a device virtual address, return a corresponding system memory virtual address.
     *
     * Since CUDA 4.1.
     *
     * \ingroup DWARF
     *
     * \param[in] dev - device index
     * \param[in] device_addr - device memory address
     * \param[out] host_addr - returned system memory address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_SEGMENT,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readGenericMemory
     * \sa writeGenericMemory
     */
    CUDBGResult (*getHostAddrFromDeviceAddr)(uint32_t dev, uint64_t device_addr, uint64_t* host_addr);

    /**
     * \fn CUDBGAPI_st::singleStepWarp41
     * \brief Single step an individual warp on a suspended CUDA device.
     *
     * Behaves like singleStepWarp65 with nsteps set to 1.
     *
     * Since CUDA 4.1.
     *
     * \note DEPRECATED in CUDA 6.5: Use \ref singleStepWarp instead.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] warpMask - the warps that have been single-stepped
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa singleStepWarp
     */
    CUDBGResult (*singleStepWarp41)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t* warpMask);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback41
     * \brief Provides the API with the function to call to notify the debugger of a new application
     * or device event.
     *
     * Behaves like setNotifyNewEventCallback but doesn't allow passing in the user data pointer.
     * The timeout field is always 0.
     *
     * Since CUDA 4.1.
     *
     * \note DEPRECATED in CUDA 13.0: Use \ref setNotifyNewEventCallback instead.
     *
     * \ingroup EVENT
     *
     * \param[in] callback - the callback function
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setNotifyNewEventCallback
     */
    CUDBGResult (*setNotifyNewEventCallback41)(CUDBGNotifyNewEventCallback41 callback);

    /**
     * \fn CUDBGAPI_st::readSyscallCallDepth
     * \brief Read the call depth of syscalls for a given thread.
     *
     * Will always return 0.
     *
     * Since CUDA 4.1.
     *
     * \note DEPRECATED in CUDA 12.9: Do not use.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readSyscallCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t* depth);

    /* 4.2 Extensions */

    /**
     * \fn CUDBGAPI_st::readTextureMemoryBindless
     * \brief This method is no longer supported since CUDA 12.0.
     *
     * Will always return CUDBG_ERROR_NOT_SUPPORTED.
     *
     * Since CUDA 4.2.
     *
     * \note DEPRECATED in CUDA 12.0: Do not use.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] texSymtabIndex - global symbol table index of the texture symbol
     * \param[in] dim - texture dimension (1 to 4)
     * \param[in] coords - array of coordinates of size dim
     * \param[out] buf - result buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*readTextureMemoryBindless)(uint32_t dev,
                                             uint32_t vsm,
                                             uint32_t wp,
                                             uint32_t texSymtabIndex,
                                             uint32_t dim,
                                             uint32_t* coords,
                                             void* buf,
                                             uint32_t sz);

    /* 5.0 Extensions */

    /**
     * \fn CUDBGAPI_st::clearAttachState
     * \brief Clear attach-specific state prior to detach.
     *
     * This call prepares the API for detaching. See the "Attaching and Detaching" section for more
     * information.
     *
     * Since CUDA 5.0.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*clearAttachState)(void);

    /**
     * \fn CUDBGAPI_st::getNextSyncEvent50
     * \brief Copies the next available event in the synchronous event queue into 'event' and
     * removes it from the queue.
     *
     * Behaves like getNextEvent but only for SYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 5.0.
     *
     * \note DEPRECATED in CUDA 5.5: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextSyncEvent50)(CUDBGEvent50* event);

    /**
     * \fn CUDBGAPI_st::memcheckReadErrorAddress
     * \brief Get the address that memcheck detected an error on.
     *
     * Will always return CUDBG_ERROR_NOT_SUPPORTED.
     *
     * Since CUDA 5.0.
     *
     * \note DEPRECATED in CUDA 12.0: Do not use.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] address - returned address detected by memcheck
     * \param[out] storage - returned address class of address
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*memcheckReadErrorAddress)(uint32_t dev,
                                            uint32_t sm,
                                            uint32_t wp,
                                            uint32_t ln,
                                            uint64_t* address,
                                            ptxStorageKind* storage);

    /**
     * \fn CUDBGAPI_st::acknowledgeSyncEvents
     * \brief Inform the debugger API that synchronous events have been processed.
     *
     * This resumes any process that was interrupted by the synchronous event (e.g. a context
     * creation, a module load, etc.).
     * This method always acknowledges only those SYNC events that have been read with getNextEvent
     * (or its deprecated variants). SYNC events that haven't been read are not acknowledged and
     * will continue to prevent their corresponding processes from proceeding.
     * ASYNC events do not require acknowledgement.
     *
     * Since CUDA 5.0.
     *
     * \ingroup EVENT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE
     *
     * \sa getNextEvent
     * \sa setNotifyNewEventCallback
     */
    CUDBGResult (*acknowledgeSyncEvents)(void);

    /**
     * \fn CUDBGAPI_st::getNextAsyncEvent50
     * \brief Copies the next available event in the asynchronous event queue into 'event' and
     * removes it from the queue.
     *
     * Behaves like getNextEvent but only for ASYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 5.0.
     *
     * \note DEPRECATED in CUDA 5.5: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextAsyncEvent50)(CUDBGEvent50* event);

    /**
     * \fn CUDBGAPI_st::requestCleanupOnDetach55
     * \brief Request for cleanup of driver state when detaching.
     *
     * Needs to be conditionally called by the client depending on the state of the debugged
     * application. See the "Attaching and Detaching" section for more information.
     *
     * Since CUDA 5.0.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref requestCleanupOnDetach instead.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa requestCleanupOnDetach
     */
    CUDBGResult (*requestCleanupOnDetach55)(void);

    /**
     * \fn CUDBGAPI_st::initializeAttachStub
     * \brief Initialize the attach stub.
     *
     * This is no longer necessary starting with driver version r590.
     *
     * Since CUDA 5.0.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*initializeAttachStub)(void);

    /**
     * \fn CUDBGAPI_st::getGridStatus50
     * \brief Check whether the grid corresponding to the ID is still present on the device.
     *
     * Behaves like getGridStatus, but takes a 32-bit grid ID instead of a 64-bit one.
     *
     * Since CUDA 5.0.
     *
     * \note DEPRECATED in CUDA 5.5: Use \ref getGridStatus instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId - grid ID
     * \param[out] status - enum indicating the grid status
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridStatus
     */
    CUDBGResult (*getGridStatus50)(uint32_t dev, uint32_t gridId, CUDBGGridStatus* status);

    /* 5.5 Extensions */

    /**
     * \fn CUDBGAPI_st::getNextSyncEvent55
     * \brief Copies the next available event in the synchronous event queue into 'event' and
     * removes it from the queue.
     *
     * Behaves like getNextEvent but only for SYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 5.5.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextSyncEvent55)(CUDBGEvent55* event);

    /**
     * \fn CUDBGAPI_st::getNextAsyncEvent55
     * \brief Copies the next available event in the asynchronous event queue into 'event' and
     * removes it from the queue.
     *
     * Behaves like getNextEvent but only for ASYNC events and doesn't support the latest event
     * struct format so some fields won't be available.
     *
     * Since CUDA 5.5.
     *
     * \note DEPRECATED in CUDA 6.0: Use \ref getNextEvent instead.
     *
     * \ingroup EVENT
     *
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_CONTEXT
     *
     * \sa getNextEvent
     */
    CUDBGResult (*getNextAsyncEvent55)(CUDBGEvent55* event);

    /**
     * \fn CUDBGAPI_st::getGridInfo55
     * \brief Get information about the specified grid.
     *
     * Behaves like getGridInfo, but returns less information.
     * Returns CUDBG_ERROR_INVALID_GRID if the context of the grid has already been destroyed (even
     * if grid ID itself is correct).
     *
     * Since CUDA 5.5.
     *
     * \note DEPRECATED in CUDA 12.0: Use \ref getGridInfo instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId - grid ID for which information is to be collected
     * \param[out] gridInfo - pointer to a client allocated structure in which grid info will be
     * returned
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridInfo
     */
    CUDBGResult (*getGridInfo55)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo55* gridInfo);

    /**
     * \fn CUDBGAPI_st::readGridId
     * \brief Read the 64-bit CUDA grid index running on a valid warp.
     *
     * The grid ID is guaranteed to be unique within a device, but not globally.
     *
     * Since CUDA 5.5.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] gridId - the returned 64-bit CUDA grid index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readThreadIdx
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readGridId)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t* gridId64);

    /**
     * \fn CUDBGAPI_st::getGridStatus
     * \brief Check whether the grid corresponding to the ID is still present on the device.
     *
     * Since CUDA 5.5.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId64 - 64-bit grid ID
     * \param[out] status - enum indicating the grid status
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getGridStatus)(uint32_t dev, uint64_t gridId64, CUDBGGridStatus* status);

    /**
     * \fn CUDBGAPI_st::setKernelLaunchNotificationMode
     * \brief Set the launch notification policy.
     *
     * If mode is CUDBG_KNL_LAUNCH_NOTIFY_EVENT, enable synchronous launch notification reporting
     * (via events). This can noticeably slow down the execution of the application.
     * If mode is CUDBG_KNL_LAUNCH_NOTIFY_DEFER, the launch notifications are not reported at all.
     *
     * Since CUDA 5.5.
     *
     * \ingroup EXEC
     *
     * \param[in] mode - mode to deliver kernel launch notifications in
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*setKernelLaunchNotificationMode)(CUDBGKernelLaunchNotifyMode mode);

    /**
     * \fn CUDBGAPI_st::getDevicePCIBusInfo
     * \brief Get PCI bus and device ids associated with device index.
     *
     * Since CUDA 5.5.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[out] pciBusId - pointer where corresponding PCI BUS ID would be stored
     * \param[out] pciDevId - pointer where corresponding PCI DEVICE ID would be stored
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getDevicePCIBusInfo)(uint32_t dev, uint32_t* pciBusId, uint32_t* pciDevId);

    /**
     * \fn CUDBGAPI_st::readDeviceExceptionState80
     * \brief Get the exception state of the SMs on the device.
     *
     * Behaves like readDeviceExceptionState but only supports up to 64 SMs.
     *
     * Since CUDA 5.5.
     *
     * \note DEPRECATED in CUDA 9.0: Use \ref readDeviceExceptionState instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[out] exceptionSMMask - Bit field containing a 1 at (1 << i) if SM i hit an exception
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readDeviceExceptionState
     */
    CUDBGResult (*readDeviceExceptionState80)(uint32_t dev, uint64_t* exceptionSMMask);

    /* 6.0 Extensions */

    /**
     * \fn CUDBGAPI_st::getAdjustedCodeAddress
     * \brief Get the adjusted code address for a given code address for a given device.
     *
     * The client must call this function before inserting a breakpoint, or when the previous or
     * next code address is needed for breakpoint inserting purposes.
     *
     * Since CUDA 5.5.
     *
     * \ingroup BP
     *
     * \param[in] dev - device index
     * \param[in] addr - instruction address
     * \param[out] adjustedAddress - adjusted address
     * \param[in] adjAction - whether the adjusted next, previous or current address is needed
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setBreakpoint
     */
    CUDBGResult (*getAdjustedCodeAddress)(uint32_t dev,
                                          uint64_t address,
                                          uint64_t* adjustedAddress,
                                          CUDBGAdjAddrAction adjAction);

    /**
     * \fn CUDBGAPI_st::readErrorPC
     * \brief Get the hardware reported error PC if it exists.
     *
     * The error PC, if available, shows the PC where an error happened (the thread can progress
     * past that so its PC could be beyond that).
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] warp - warp index
     * \param[out] errorPC - PC ofthe exception
     * \param[out] errorPCValid - boolean to indicate that the returned error PC is valid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readErrorPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t* errorPC, bool* errorPCValid);

    /**
     * \fn CUDBGAPI_st::getNextEvent
     * \brief Copies the next available event into 'event' and removes it from the queue.
     *
     * CUDBG_ERROR_NO_EVENT_AVAILABLE is returned if the queue is empty.
     * ASYNC and SYNC queues are separate and each one is ordered separately, but it's impossible to
     * find out the relative order of ASYNC and SYNC events.
     *
     * Since CUDA 6.0.
     *
     * \ingroup EVENT
     *
     * \param[in] type - application event queue type
     * \param[out] event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE
     *
     * \sa acknowledgeSyncEvents
     * \sa setNotifyNewEventCallback
     */
    CUDBGResult (*getNextEvent)(CUDBGEventQueueType type, CUDBGEvent* event);

    /**
     * \fn CUDBGAPI_st::getElfImageByHandle
     * \brief Get the relocated or non-relocated ELF image for the given handle on the given device.
     *
     * The handle is provided in the ELF Image Loaded notification event.
     *
     * Since CUDA 6.0.
     *
     * \ingroup DWARF
     *
     * \param[in] dev - device index
     * \param[in] handle - elf image handle
     * \param[in] type - type of the requested ELF image
     * \param[out] elfImage - pointer to the ELF image
     * \param[in] elfImage_size - size of the ELF image
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getElfImage
     * \sa getLoadedFunctionInfo
     */
    CUDBGResult (*getElfImageByHandle)(uint32_t dev,
                                       uint64_t handle,
                                       CUDBGElfImageType type,
                                       void* elfImage,
                                       uint64_t size);

    /**
     * \fn CUDBGAPI_st::resumeWarpsUntilPC60
     * \brief Insert a temporary breakpoint at the specified virtual PC and resume all warps in the
     * specified bitmask on a given SM.
     *
     * Compared to resumeDevice(), this method provides finer-grain control by resuming a selected
     * set of warps on the same SM.
     * The main intended usage is to accelerate the single-stepping process when the target PC is
     * known in advance. Instead of single-stepping each warp individually until the target PC is
     * hit, the client can use this method.
     * If an unsteppable barrier is hit by the resumed warps, this method returns early (before
     * reaching the target PC).
     * When this method is used, errors within CUDA kernels will no longer be reported precisely.
     * In the situation where resuming warps is not possible, this method will return
     * CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE. The client should then fall back to using
     * singleStepWarp() or resumeDevice().
     *
     * Since CUDA 6.0.
     *
     * \note DEPRECATED in CUDA 13.2: Use \ref resumeWarpsUntilPC instead.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] warpMask - the bitmask of warps to resume (1 = resume, 0 = do not resume)
     * \param[in] virtPC - the virtual PC where the temporary breakpoint will be inserted
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa resumeWarpsUntilPC
     */
    CUDBGResult (*resumeWarpsUntilPC60)(uint32_t dev, uint32_t sm, uint64_t warpMask, uint64_t virtPC);

    /**
     * \fn CUDBGAPI_st::readWarpState60
     * \brief Read the state of a given warp.
     *
     * Behaves like readWarpState but returns fewer fields.
     *
     * Since CUDA 6.0.
     *
     * \note DEPRECATED in CUDA 12.0: Use \ref readWarpState instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] state - pointer to structure that contains warp state
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readWarpState
     */
    CUDBGResult (*readWarpState60)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState60* state);

    /**
     * \fn CUDBGAPI_st::readRegisterRange60
     * \brief Read content of a range of hardware registers.
     *
     * Since CUDA 6.0.
     *
     * \note DEPRECATED in CUDA 13.2: Use \ref readRegisterRange instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] index - index of the first register to read
     * \param[in] registers_size - number of registers to read
     * \param[out] registers - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readRegisterRange
     */
    CUDBGResult (*readRegisterRange60)(uint32_t dev,
                                       uint32_t sm,
                                       uint32_t wp,
                                       uint32_t ln,
                                       uint32_t index,
                                       uint32_t registers_size,
                                       uint32_t* registers);

    /**
     * \fn CUDBGAPI_st::readGenericMemory
     * \brief Read content at an address in any memory segment.
     *
     * The address will be used to determine whether the read is to local, shared or global memory.
     * The target address range should entirely reside within a single memory segment. Coordinate
     * arguments are only used when relevant. They should be provided for the following segments:
     * - Shared memory: SM and Warp
     * - Local memory: SM, Warp and Lane
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] buf_size - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readCodeMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readGenericMemory)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint32_t ln,
                                     uint64_t addr,
                                     void* buf,
                                     uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeGenericMemory
     * \brief Write to an address in any memory segment.
     *
     * The address will be used to determine whether the write is to local, shared or global memory.
     * The target address range should entirely reside within a single memory segment. Coordinate
     * arguments are only used when relevant. They should be provided for the following segments:
     * - Shared memory: SM and Warp
     * - Local memory: SM, Warp and Lane
     *
     * Since CUDA 6.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa writeGlobalMemory
     * \sa writeLocalMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     */
    CUDBGResult (*writeGenericMemory)(uint32_t dev,
                                      uint32_t sm,
                                      uint32_t wp,
                                      uint32_t ln,
                                      uint64_t addr,
                                      const void* buf,
                                      uint32_t sz);

    /**
     * \fn CUDBGAPI_st::readGlobalMemory
     * \brief Read content at an address in the global address space.
     *
     * If the address is valid on more than one device and one of those devices does not support
     * UVA, an error is returned.
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param[in] addr - memory address
     * \param[out] buf - buffer
     * \param[in] buf_size - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readCodeMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readGlobalMemory)(uint64_t addr, void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::writeGlobalMemory
     * \brief Write to an address in global memory
     *
     * It's not possible to access a shared memory page or an ambiguous address allocated on several
     * devices that don't support UVA.
     *
     * Since CUDA 6.0.
     *
     * \ingroup WRITE
     *
     * \param[in] addr - address
     * \param[in] buf - buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM,
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa writeGenericMemory
     * \sa writeLocalMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     */
    CUDBGResult (*writeGlobalMemory)(uint64_t addr, const void* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::getManagedMemoryRegionInfo
     * \brief Get a sorted list of managed memory regions.
     *
     * The sorted list of memory regions starts from a region containing the specified starting
     * address. If the starting address is set to 0, a sorted list of managed memory regions is
     * returned which starts from the managed memory region with the lowest start address.
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param[in] startAddress - the address that the first region in the list must contain
     * \param[out] memoryInfo - client-allocated array of memory region records of type
     * CUDBGMemoryInfo
     * \param[in] memoryInfo_size - number of records of type CUDBGMemoryInfo that
     * memoryInfo can hold
     * \param[out] numEntries - pointer to a client-allocated variable holding
     * the number of valid entries returned in memoryInfo
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getManagedMemoryRegionInfo)(uint64_t startAddress,
                                              CUDBGMemoryInfo* memoryInfo,
                                              uint32_t memoryInfo_size,
                                              uint32_t* numEntries);

    /**
     * \fn CUDBGAPI_st::isDeviceCodeAddress
     * \brief Determine whether a virtual address resides within device code.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param[in] addr - virtual address
     * \param[out] isDeviceAddress - true if address resides within device code
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*isDeviceCodeAddress)(uintptr_t addr, bool* isDeviceAddress);

    /**
     * \fn CUDBGAPI_st::requestCleanupOnDetach
     * \brief Request for cleanup of driver state when detaching.
     *
     * Needs to be conditionally called by the client depending on the state of the debugged
     * application. See the "Attaching and Detaching" section for more information.
     *
     * Since CUDA 6.0.
     *
     * \ingroup INIT
     *
     * \param[in] appResumeFlag - value of CUDBG_RESUME_FOR_ATTACH_DETACH as read from the
     * application's process space.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*requestCleanupOnDetach)(uint32_t appResumeFlag);

    /* 6.5 Extensions */

    /**
     * \fn CUDBGAPI_st::readPredicates
     * \brief Read content of hardware predicate registers.
     *
     * Since CUDA 6.5.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] predicates_size - number of predicate registers to read
     * \param[out] predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readGlobalMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readPredicates)(uint32_t dev,
                                  uint32_t sm,
                                  uint32_t wp,
                                  uint32_t ln,
                                  uint32_t predicates_size,
                                  uint32_t* predicates);

    /**
     * \fn CUDBGAPI_st::writePredicates
     * \brief Write to hardware predicates
     *
     * This method writes to predicates_size predicates, starting from P0.
     * Each predicate value must be either 0 or 1.
     *
     * Since CUDA 6.5.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] predicates_size - predicates count
     * \param[in] predicates - predicate values
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writeRegister
     * \sa writeUniformPredicates
     * \sa writeUniformRegister
     */
    CUDBGResult (*writePredicates)(uint32_t dev,
                                   uint32_t sm,
                                   uint32_t wp,
                                   uint32_t ln,
                                   uint32_t predicates_size,
                                   const uint32_t* predicates);

    /**
     * \fn CUDBGAPI_st::getNumPredicates
     * \brief Get the number of predicate registers per lane on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 6.5.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numPredicates - the returned number of predicate registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     */
    CUDBGResult (*getNumPredicates)(uint32_t dev, uint32_t* numPredicates);

    /**
     * \fn CUDBGAPI_st::readCCRegister
     * \brief Read the hardware CC register.
     *
     * The CC register is no longer available in the supported hardware.
     *
     * Since CUDA 6.5.
     *
     * \note DEPRECATED in CUDA 13.1: Do not use.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] val - the returned value of the CC register
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t* val);

    /**
     * \fn CUDBGAPI_st::writeCCRegister
     * \brief Write to the hardware CC register.
     *
     * The CC register is no longer available in the supported hardware.
     *
     * Since CUDA 6.5.
     *
     * \note DEPRECATED in CUDA 13.1: Do not use.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] val - the new value of the CC register
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*writeCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);

    /**
     * \fn CUDBGAPI_st::getDeviceName
     * \brief Get the device name string.
     *
     * Returns CUDBG_ERROR_BUFFER_TOO_SMALL if the provided buffer is not large enough.
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 6.5.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] buf - the destination buffer
     * \param[in] sz - buffer size in bytes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getDeviceType
     * \sa getSMType
     */
    CUDBGResult (*getDeviceName)(uint32_t dev, char* buf, uint32_t sz);

    /**
     * \fn CUDBGAPI_st::singleStepWarp65
     * \brief Single step an individual warp nsteps times on a suspended CUDA device.
     *
     * Behaves like singleStepWarp with no lane hint and the
     * CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS flag set.
     *
     * Since CUDA 6.5.
     *
     * \note DEPRECATED in CUDA 12.4: Use \ref singleStepWarp instead.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] nsteps - number of single steps
     * \param[out] warpMask - the warps that have been single-stepped
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa singleStepWarp
     */
    CUDBGResult (*singleStepWarp65)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    uint32_t nsteps,
                                    uint64_t* warpMask);

    /* 9.0 Extensions */

    /**
     * \fn CUDBGAPI_st::readDeviceExceptionState
     * \brief Get the exception state of the SMs on the device.
     *
     * If the CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS capability is enabled,
     * exceptions in exited warps will be reported.
     *
     * Since CUDA 9.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[out] mask - Arbitrarily sized bit field containing a 1 at (1 << i) if SM i hit an
     * exception
     * \param[in] numWords - Number of uint64_t elements in \p mask (must be large enough
     * to hold a bit for each sm on the device)
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumSMs
     */
    CUDBGResult (*readDeviceExceptionState)(uint32_t dev, uint64_t* mask, uint32_t numWords);

    /* 10.0 Extensions */

    /**
     * \fn CUDBGAPI_st::getNumUniformRegisters
     * \brief Get the number of uniform registers per warp on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 10.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numRegs - the returned number of uniform registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformPredicates
     * \sa getNumWarps
     */
    CUDBGResult (*getNumUniformRegisters)(uint32_t dev, uint32_t* numRegs);

    /**
     * \fn CUDBGAPI_st::readUniformRegisterRange
     * \brief Read a range of uniform registers.
     *
     * Since CUDA 10.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] regno - starting index into uniform register file
     * \param[in] registers_size - number of bytes to read
     * \param[out] registers - pointer to buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readRegister
     */
    CUDBGResult (*readUniformRegisterRange)(uint32_t dev,
                                            uint32_t sm,
                                            uint32_t wp,
                                            uint32_t regno,
                                            uint32_t registers_size,
                                            uint32_t* registers);

    /**
     * \fn CUDBGAPI_st::writeUniformRegister
     * \brief Write to a hardware uniform register
     *
     * Since CUDA 10.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] regno - register number
     * \param[in] val - value
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writePredicates
     * \sa writeRegister
     * \sa writeUniformPredicates
     */
    CUDBGResult (*writeUniformRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val);

    /**
     * \fn CUDBGAPI_st::getNumUniformPredicates
     * \brief Get the number of uniform predicate registers per warp on the device.
     *
     * This value is constant within a single session for a given device.
     *
     * Since CUDA 10.0.
     *
     * \ingroup DEV
     *
     * \param[in] dev - device index
     * \param[out] numPredicates - the returned number of uniform predicate registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getNumDevices
     * \sa getNumLanes
     * \sa getNumPredicates
     * \sa getNumRegisters
     * \sa getNumSMs
     * \sa getNumUniformRegisters
     * \sa getNumWarps
     */
    CUDBGResult (*getNumUniformPredicates)(uint32_t dev, uint32_t* numPredicates);

    /**
     * \fn CUDBGAPI_st::readUniformPredicates
     * \brief Read contents of uniform predicate registers.
     *
     * Since CUDA 10.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] predicates_size - number of predicate registers to read
     * \param[out] predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readPredicates
     */
    CUDBGResult (*readUniformPredicates)(uint32_t dev,
                                         uint32_t sm,
                                         uint32_t wp,
                                         uint32_t predicates_size,
                                         uint32_t* predicates);

    /**
     * \fn CUDBGAPI_st::writeUniformPredicates
     * \brief Write to hardware uniform predicates
     *
     * This method writes to predicates_size uniform predicates, starting from UP0.
     * Each predicate value must be either 0 or 1.
     *
     * Since CUDA 10.0.
     *
     * \ingroup WRITE
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] predicates_size - predicates count
     * \param[in] predicates - predicate values
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writePredicates
     * \sa writeRegister
     * \sa writeUniformRegister
     */
    CUDBGResult (*writeUniformPredicates)(uint32_t dev,
                                          uint32_t sm,
                                          uint32_t wp,
                                          uint32_t predicates_size,
                                          const uint32_t* predicates);

    /* 11.8 Extensions */

    /**
     * \fn CUDBGAPI_st::getLoadedFunctionInfo118
     * \brief Get the section number and address of loaded functions for a given module.
     *
     * Behaves like getLoadedFunctionInfo but doesn't allow querying a subset of all lazily loaded
     * functions.
     *
     * Since CUDA 11.8.
     *
     * \note DEPRECATED in CUDA 12.3: Use \ref getLoadedFunctionInfo instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] handle - ELF/cubin image handle
     * \param[out] info - information about loaded functions
     * \param[in] numEntries - number of function load entries to read
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getLoadedFunctionInfo
     */
    CUDBGResult (*getLoadedFunctionInfo118)(uint32_t dev,
                                            uint64_t handle,
                                            CUDBGLoadedFunctionInfo* info,
                                            uint32_t numEntries);

    /* 12.0 Extensions */

    /**
     * \fn CUDBGAPI_st::getGridInfo120
     * \brief Get information about the specified grid.
     *
     * Behaves like getGridInfo, but returns less information.
     * Returns CUDBG_ERROR_INVALID_GRID if the context of the grid has already been destroyed (even
     * if grid ID itself is correct).
     *
     * Since CUDA 12.0.
     *
     * \note DEPRECATED in CUDA 12.7: Use \ref getGridInfo instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId - grid ID for which information is to be collected
     * \param[out] gridInfo - pointer to a client allocated structure in which grid info will be
     * returned
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getGridInfo
     */
    CUDBGResult (*getGridInfo120)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo120* gridInfo);

    /**
     * \fn CUDBGAPI_st::getClusterDim120
     * \brief Get the number of blocks in the given cluster.
     *
     * Behaves like getClusterDim, but takes a grid ID instead of warp coordinates. In newer GPU
     * architectures, it's possible to have different warps belong to blocks of clusters of
     * different size.
     *
     * Since CUDA 12.0.
     *
     * \note DEPRECATED in CUDA 12.7: Use \ref getClusterDim instead.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId64 - grid ID
     * \param[out] clusterDim - the returned number of blocks in the cluster
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getClusterDim
     */
    CUDBGResult (*getClusterDim120)(uint32_t dev, uint64_t gridId64, CuDim3* clusterDim);

    /**
     * \fn CUDBGAPI_st::readWarpState120
     * \brief Read the state of a given warp.
     *
     * Behaves like readWarpState but returns fewer fields.
     *
     * Since CUDA 12.0.
     *
     * \note DEPRECATED in CUDA 12.7: Use \ref readWarpState instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] state - pointer to structure that contains warp state
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readWarpState
     */
    CUDBGResult (*readWarpState120)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState120* state);

    /**
     * \fn CUDBGAPI_st::readClusterIdx
     * \brief Read the CUDA cluster index running on a valid warp.
     *
     * Since CUDA 12.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] clusterIdx - the returned CUDA cluster index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readActiveLanes
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readValidLanes
     * \sa readValidWarps
     */
    CUDBGResult (*readClusterIdx)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3* clusterIdx);

    /* 12.2 Extensions */

    /**
     * \fn CUDBGAPI_st::getErrorStringEx
     * \brief Fills a user-provided buffer with an error message encoded as a null-terminated ASCII
     * string.
     *
     * The error message is specific to the last failed API call and is invalidated after every API
     * method call except this one.
     * It's possible to query the size of the error message without reading it by passing 0 as `buf`
     * and `bufSz` parameters.
     * The `msgSz` parameter is optional unless 0 as passed in as `buf` and `bufSz`.
     * CUDBG_ERROR_BUFFER_TOO_SMALL is returned when the passed in buffer is too small to contain
     * the message.
     *
     * Since CUDA 12.2.
     *
     * \ingroup EVENT
     *
     * \param[out] buf - the destination buffer
     * \param[in] bufSz - the size of the destination buffer in bytes
     * \param[out] msgSz - the size of the written error message including the terminating null
     * character.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getErrorStringEx)(char* buf, uint32_t bufSz, uint32_t* msgSz);

    /* 12.3 Extensions */

    /**
     * \fn CUDBGAPI_st::getLoadedFunctionInfo
     * \brief Get the section number and address of loaded functions for a given module.
     *
     * If the CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING capability is enabled, CUDA loads
     * functions lazily after the module has been reported. This method could be used to get the
     * lazily loaded functions.
     *
     * Since CUDA 12.3.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] handle - ELF/cubin image handle
     * \param[out] info - information about loaded functions
     * \param[in] startIndex - start index of the entries to get
     * \param[in] numEntries - number of function load entries to read
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getLoadedFunctionInfo)(uint32_t dev,
                                         uint64_t handle,
                                         CUDBGLoadedFunctionInfo* info,
                                         uint32_t startIndex,
                                         uint32_t numEntries);

    /**
     * \fn CUDBGAPI_st::generateCoredump
     * \brief Generate a coredump for the current GPU state.
     *
     * Since CUDA 12.3.
     *
     * \ingroup READ
     *
     * \param[in] filename - target coredump file name
     * \param[in] flags - coredump generation flags/options
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*generateCoredump)(const char* filename, CUDBGCoredumpGenerationFlags flags);

    /**
     * \fn CUDBGAPI_st::getConstBankAddress123
     * \brief Convert constant bank number and offset into GPU VA.
     *
     * It's more convenient to get the constbank address and then calculate the VA for any const
     * address using that.
     *
     * Since CUDA 12.3.
     *
     * \note DEPRECATED in CUDA 12.4: Use \ref getConstBankAddress instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] bank - constant bank number
     * \param[in] offset - offset within the bank
     * \param[out] address - GPU VA
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_MISSING_DATA
     *
     * \sa getConstBankAddress
     */
    CUDBGResult (*getConstBankAddress123)(uint32_t dev,
                                          uint32_t sm,
                                          uint32_t wp,
                                          uint32_t bank,
                                          uint32_t offset,
                                          uint64_t* address);

    /* 12.4 Extensions */

    /**
     * \fn CUDBGAPI_st::getDeviceInfoSizes
     * \brief Return sizes for device info structs and defined attributes.
     *
     * Since CUDA 12.4.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[out] sizes - device info sizes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getDeviceInfo
     */
    CUDBGResult (*getDeviceInfoSizes)(uint32_t dev, CUDBGDeviceInfoSizes* sizes);

    /**
     * \fn CUDBGAPI_st::getDeviceInfo
     * \brief Read full device info for the device.
     *
     * Information returned by this method is cheap to calculate, so it can be used after every
     * suspend to quickly get the updated device state.
     * For convenience, the caller can always request partial updates, the API will return a full
     * response when returning a partial one is not possible.
     * If the CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS capability is enabled,
     * exceptions in exited warps will be reported.
     *
     * Since CUDA 12.4.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] type - query type (full or delta)
     * \param[out] buffer - output buffer
     * \param[in] length - output buffer length
     * \param[out] dataLength - number of bytes written to the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getDeviceInfoSizes
     */
    CUDBGResult (*getDeviceInfo)(uint32_t dev,
                                 CUDBGDeviceInfoQueryType_t type,
                                 void* buffer,
                                 uint32_t length,
                                 uint32_t* dataLength);

    /**
     * \fn CUDBGAPI_st::getConstBankAddress
     * \brief Get constant bank GPU VA and size.
     *
     * Since CUDA 12.4.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] gridId64 - grid ID of the grid containing the constant bank
     * \param[in] bank - constant bank number
     * \param[out] address - GPU VA of the bank memory
     * \param[out] size - bank size
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_MISSING_DATA
     *
     * \sa getBindlessConstAddress
     */
    CUDBGResult (*getConstBankAddress)(uint32_t dev,
                                       uint64_t gridId64,
                                       uint32_t bank,
                                       uint64_t* address,
                                       uint32_t* size);

    /**
     * \fn CUDBGAPI_st::singleStepWarp
     * \brief Single step an individual warp nsteps times on a suspended CUDA device.
     *
     * By default, if the warp is on a convergence barrier, resumeWarpsUntilPC is called internally
     * to quickly advance the warp past that barrier. If the
     * CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS flag is passed in, this optimization is
     * not performed (which would likely lead to diverged threads becoming focused and starting to
     * advance towards the convergence barrier).
     * If a warp is on a block-wide barrier (or wider), other warps required to advance past the
     * barrier are automatically resumed. The output parameter warpMask will have the warps resumed
     * in the current SM. Warps can also be resumed in other SMs, but are not reported via the API.
     * This method is synchronous and will not return until the step is complete.
     *
     * Since CUDA 12.4.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] laneHint - focused lane (~0 to let the API decide)
     * \param[in] nsteps - number of single steps
     * \param[in] flags - flags of type CUDBGSingleStepFlags to change the stepping behavior
     * \param[out] warpMask - the warps that have been single-stepped
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa resumeAllDevices
     * \sa resumeWarpsUntilPC
     * \sa suspendAllDevices
     */
    CUDBGResult (*singleStepWarp)(uint32_t dev,
                                  uint32_t sm,
                                  uint32_t wp,
                                  uint32_t laneHint,
                                  uint32_t nsteps,
                                  uint32_t flags,
                                  uint64_t* warpMask);

    /* 12.5 Extensions */

    /**
     * \fn CUDBGAPI_st::readAllVirtualReturnAddresses
     * \brief Read all the virtual return addresses for a thread (the full backtrace).
     *
     * Note that syscallCallDepth is always set to 0.
     *
     * Since CUDA 12.5.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] addrs - the returned addresses array
     * \param[in] numAddrs - number of elements in addrs array
     * \param[out] callDepth - the returned call depth
     * \param[out] syscallCallDepth - the returned syscall call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCallDepth
     * \sa readReturnAddress
     */
    CUDBGResult (*readAllVirtualReturnAddresses)(uint32_t dev,
                                                 uint32_t sm,
                                                 uint32_t wp,
                                                 uint32_t ln,
                                                 uint64_t* addrs,
                                                 uint32_t numAddrs,
                                                 uint32_t* callDepth,
                                                 uint32_t* syscallCallDepth);

    /**
     * \fn CUDBGAPI_st::getSupportedDebuggerCapabilities
     * \brief Returns debug agent capabilities that are supported by this version of the API.
     *
     * This API method can be called without initializing the API.
     *
     * Since CUDA 12.5.
     *
     * \ingroup INIT
     *
     * \param[out] capabilities - returned debug engine capabilities
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getSupportedDebuggerCapabilities)(CUDBGCapabilityFlags* capabilities);

    /**
     * \fn CUDBGAPI_st::readSmException
     * \brief Get the SM exception status if it exists.
     *
     * If the CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS capability is enabled,
     * exceptions in exited warps will be reported.
     *
     * Since CUDA 12.5.
     *
     * \ingroup READ
     *
     * \param[in] dev - the device index
     * \param[in] sm - the SM index
     * \param[out] exception - returned exception
     * \param[out] errorPC - returned PC of the exception
     * \param[out] errorPCValid - boolean to indicate that the returned error PC is valid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readSmException)(uint32_t dev,
                                   uint32_t sm,
                                   CUDBGException_t* exception,
                                   uint64_t* errorPC,
                                   bool* errorPCValid);

    /* 12.6 Extensions */

    /**
     * \fn CUDBGAPI_st::executeInternalCommand
     * \brief Execute an internal command (not available in public driver builds)
     *
     * Always returns CUDBG_ERROR_NOT_SUPPORTED.
     *
     * Since CUDA 12.6.
     *
     * \ingroup EXEC
     *
     * \param[in] command - the command name and arguments
     * \param[out] resultBuffer - the destination buffer
     * \param[in] sizeInBytes - buffer size in bytes
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*executeInternalCommand)(const char* command, char* resultBuffer, uint32_t sizeInBytes);

    /* 12.7 Extensions */

    /**
     * \fn CUDBGAPI_st::getGridInfo
     * \brief Get information about the specified grid.
     *
     * Returns CUDBG_ERROR_INVALID_GRID if the context of the grid has already been destroyed (even
     * if grid ID itself is correct).
     *
     * Since CUDA 12.7.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] gridId - grid ID for which information is to be collected
     * \param[out] gridInfo - pointer to a client allocated structure in which grid info will be
     * returned
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getGridInfo)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo* gridInfo);

    /**
     * \fn CUDBGAPI_st::getClusterDim
     * \brief Get the number of blocks in the given cluster.
     *
     * Since CUDA 12.7.
     *
     * \ingroup GRID
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] clusterDim - the returned number of blocks in the cluster
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getBlockDim
     * \sa getGridDim
     */
    CUDBGResult (*getClusterDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3* clusterDim);

    /**
     * \fn CUDBGAPI_st::readWarpState127
     * \brief Read the state of a given warp.
     *
     * Behaves like readWarpState but returns fewer fields.
     *
     * Since CUDA 12.7.
     *
     * \note DEPRECATED in CUDA 12.9: Use \ref readWarpState instead.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] state - pointer to structure that contains warp state
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readWarpState
     */
    CUDBGResult (*readWarpState127)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState127* state);

    /**
     * \fn CUDBGAPI_st::getClusterExceptionTargetBlock
     * \brief Get the target block index and validity status for cluster exceptions.
     *
     * Since CUDA 12.7.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] blockIdx - pointer to a `CuDim3` structure that will be populated with the target
     * block index
     * \param[out] blockIdxValid - pointer to a boolean variable that will be set to `true` if the target
     * block index is valid, and `false` otherwise. Value will be set to false if the warp is not stopped on a
     * cluster exception
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getClusterExceptionTargetBlock)(uint32_t dev,
                                                  uint32_t sm,
                                                  uint32_t wp,
                                                  CuDim3* blockIdx,
                                                  bool* blockIdxValid);

    /* 12.8 Extensions */

    /**
     * \fn CUDBGAPI_st::readWarpResources
     * \brief Get the resources assigned to a given warp.
     *
     * Note that these resources can change between suspends, which makes this method useful for
     * avoiding warp data access errors.
     *
     * Since CUDA 12.8.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] resources - pointer to structure that contains warp resources
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readWarpResources)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpResources* resources);

    /* 12.9 Extensions */

    /**
     * \fn CUDBGAPI_st::getCbuWarpState
     * \brief Get the CBU state of a given warp.
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] warpMask - bitmask of the warps which states should be returned in warpStates
     * \param[out] warpStates - pointer to the array of CUDBGCbuWarpState structures
     * \param[in] numWarpStates - number of elements in warpStates array
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getCbuWarpState)(uint32_t dev,
                                   uint32_t sm,
                                   uint64_t warpMask,
                                   CUDBGCbuWarpState* warpStates,
                                   uint32_t numWarpStates);

    /**
     * \fn CUDBGAPI_st::readWarpState
     * \brief Read the state of a given warp.
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] state - pointer to structure that contains warp state
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readWarpState)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState* state);

    /**
     * \fn CUDBGAPI_st::consumeCudaLogs129
     * \brief Get CUDA error log entries.
     *
     * This consumes the log entries, so they will not be available in subsequent calls.
     * This functionality is only available if the CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS
     * capability is enabled.
     *
     * \note Since CUDA 13.4, this function can be called from the notification callback function.
     *
     * Since CUDA 12.9.
     *
     * \note DEPRECATED in CUDA 13.4: Use \ref consumeCudaLogs instead.
     *
     * \ingroup READ
     *
     * \param[out] logMessages - client-allocated array to store log entries
     * \param[in] numMessages - capacity of the logMessages array, in number of elements
     * \param[out] numConsumed - returned number of entries written to logMessages
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa consumeCudaLogs
     */
    CUDBGResult (*consumeCudaLogs129)(CUDBGCudaLogMessage129* logMessages,
                                      uint32_t numMessages,
                                      uint32_t* numConsumed);

    /**
     * \fn CUDBGAPI_st::readCPUCallStack
     * \brief Read the CPU call stack captured at the time of kernel launch.
     *
     * This method only works if the
     * CUDBG_DEBUGGER_CAPABILITY_COLLECT_CPU_CALL_STACK_FOR_KERNEL_LAUNCHES capability is enabled.
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] gridId64 - 64-bit grid ID
     * \param[out] addrs - the returned addresses array, can be NULL
     * \param[in] numAddrs - capacity of addrs (possibly 0)
     * \param[out] totalNumAddrs - the actual size of the stack (number of frames) is written here;
     * the value written can be greater than numAddrs
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*readCPUCallStack)(uint32_t dev,
                                    uint64_t gridId64,
                                    uint64_t* addrs,
                                    uint32_t numAddrs,
                                    uint32_t* totalNumAddrs);

    /* 13.0 Extensions */

    /**
     * \fn CUDBGAPI_st::getCudaExceptionString
     * \brief Get the error string for CUDA Exceptions.
     *
     * Since CUDA 13.0.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] buf - buffer for the error string
     * \param[in] bufSz - buffer size
     * \param[out] msgSz - error message size with null character, can be null
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getCudaExceptionString)(uint32_t dev,
                                          uint32_t sm,
                                          uint32_t wp,
                                          uint32_t ln,
                                          char* buf,
                                          uint32_t bufSz,
                                          uint32_t* msgSz);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback
     * \brief Provides the API with the function to call to notify the debugger of a new application
     * or device event.
     *
     * The callback function is called for every ASYNC and SYNC event.
     * The callback function is always called on the same thread. No API methods can be called from
     * that thread except getNextEvent(), acknowledgeSyncEvents() (and their deprecated variants),
     * and consumeCudaLogs() (since CUDA 13.4), otherwise CUDBG_ERROR_RECURSIVE_API_CALL will be
     * returned.
     *
     * Since CUDA 13.0.
     *
     * \ingroup EVENT
     *
     * \param[in] callback - the callback function
     * \param[in] data - a pointer to be passed to the callback when called
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa acknowledgeSyncEvents
     * \sa getNextEvent
     * \sa consumeCudaLogs
     */
    CUDBGResult (*setNotifyNewEventCallback)(CUDBGNotifyNewEventCallback callback, void* userData);

    /* 13.1 Extensions */

    /**
     * \fn CUDBGAPI_st::getHardwareBarrierInfo
     * \brief Get hardware barrier information.
     *
     * Since CUDA 13.1.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[out] scope - barrier scope
     * \param[out] buf - buffer for the barrier information
     * \param[in] bufSz - buffer size
     * \param[out] msgSz - error message size with null character, can be null
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     */
    CUDBGResult (*getHardwareBarrierInfo)(uint32_t dev,
                                          uint32_t sm,
                                          uint32_t wp,
                                          uint32_t ln,
                                          CUDBGBarrierScope* scope,
                                          char* buf,
                                          uint32_t bufSz,
                                          uint32_t* msgSz);

    /* 13.2 Extensions */

    /**
     * \fn CUDBGAPI_st::readRegisterRange
     * \brief Read content of a hardware range of hardware registers.
     *
     * Since CUDA 13.2.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[in] ln - lane index
     * \param[in] index - index of the first register to read
     * \param[in] numRegisters - number of registers to read
     * \param[out] registers - buffer
     * \param[out] numRegistersRead - number of registers actually read, ignored if null
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readParamMemory
     * \sa readRegister
     * \sa readSharedMemory
     * \sa readTextureMemory
     */
    CUDBGResult (*readRegisterRange)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint32_t ln,
                                     uint32_t index,
                                     uint32_t numRegisters,
                                     uint32_t* registers,
                                     uint32_t* numRegistersRead);

    /**
     * \fn CUDBGAPI_st::insertBreakpoint
     * \brief Set a breakpoint at the given instruction address for the given device.
     *
     * Before setting a breakpoint, getAdjustedCodeAddress() should be called to get the adjusted
     * breakpoint address.
     * The returned handle can be used to enable/disable/remove the breakpoint.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] dev - the device index
     * \param[in] addr - instruction address
     * \param[out] handle - the returned breakpoint handle
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_BREAKPOINT_STATE_CONFLICT
     *
     * \sa disableBreakpoint
     * \sa enableBreakpoint
     * \sa getWarpHitBreakpoint
     * \sa isBreakpointEnabled
     * \sa removeBreakpoint
     */
    CUDBGResult (*insertBreakpoint)(uint32_t dev, uint64_t addr, CUDBGBreakpointHandle* handle);

    /**
     * \fn CUDBGAPI_st::removeBreakpoint
     * \brief Remove a breakpoint specified by its handle.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] handle - the breakpoint handle.  If it's \ref CUDBG_BREAKPOINT_HANDLE_ALL_USER_BREAKPOINTS,
     * remove all breakpoints inserted by the client (except the break-on-launch breakpoint). It's fine to use
     * ALL_USER_BREAKPOINTS when there are no breakpoints added.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa disableBreakpoint
     * \sa enableBreakpoint
     * \sa getWarpHitBreakpoint
     * \sa insertBreakpoint
     * \sa isBreakpointEnabled
     */
    CUDBGResult (*removeBreakpoint)(CUDBGBreakpointHandle handle);

    /**
     * \fn CUDBGAPI_st::enableBreakpoint
     * \brief Enable a breakpoint specified by its handle.
     *
     * Disabling/enabling a breakpoint might be faster than removing and inserting it again.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] handle - the breakpoint handle.  If it's \ref CUDBG_BREAKPOINT_HANDLE_ALL_USER_BREAKPOINTS,
     * enable all breakpoints inserted by the client (except the break-on-launch breakpoint). It's fine to use
     * ALL_USER_BREAKPOINTS when there are no breakpoints added.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_BREAKPOINT_STATE_CONFLICT
     *
     * \sa disableBreakpoint
     * \sa getWarpHitBreakpoint
     * \sa insertBreakpoint
     * \sa isBreakpointEnabled
     * \sa removeBreakpoint
     */
    CUDBGResult (*enableBreakpoint)(CUDBGBreakpointHandle handle);

    /**
     * \fn CUDBGAPI_st::disableBreakpoint
     * \brief Disable a breakpoint specified by its handle.
     *
     * Disabling/enabling a breakpoint might be faster than removing and inserting it again.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] handle - the breakpoint handle.  If it's \ref CUDBG_BREAKPOINT_HANDLE_ALL_USER_BREAKPOINTS,
     * disable all breakpoints inserted by the client (except the break-on-launch breakpoint). It's fine to
     * use ALL_USER_BREAKPOINTS when there are no breakpoints added.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL,
     * \return CUDBG_ERROR_BREAKPOINT_STATE_CONFLICT
     *
     * \sa enableBreakpoint
     * \sa getWarpHitBreakpoint
     * \sa insertBreakpoint
     * \sa isBreakpointEnabled
     * \sa removeBreakpoint
     */
    CUDBGResult (*disableBreakpoint)(CUDBGBreakpointHandle handle);

    /**
     * \fn CUDBGAPI_st::isBreakpointEnabled
     * \brief Check if a breakpoint specified by its handle is enabled.
     *
     * The breakpoint enablement state is never implicitly modified by the API implementation;
     * the API user must always enable or disable them manually.
     * Prior to CUDA 13.4, an error is returned if the CUDBG_BREAKPOINT_HANDLE_BREAK_ON_LAUNCH
     * handle is provided.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] handle - the breakpoint handle.
     * \param[out] enabled - whether the breakpoint is enabled
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa disableBreakpoint
     * \sa enableBreakpoint
     * \sa getWarpHitBreakpoint
     * \sa insertBreakpoint
     * \sa removeBreakpoint
     */
    CUDBGResult (*isBreakpointEnabled)(CUDBGBreakpointHandle handle, uint32_t* enabled);

    /**
     * \fn CUDBGAPI_st::getWarpHitBreakpoint
     * \brief Get the handle of the breakpoint that the given warp hit.
     *
     * An error is returned if the warp did not hit a breakpoint.
     * Use readBrokenWarps() to check if the warp is broken before calling this method.
     * Some breakpoint handles are special, see the documentation of CUDBGBreakpointHandle for more
     * details.
     *
     * Since CUDA 13.2.
     *
     * \ingroup BP
     *
     * \param[in] dev - device index
     * \param[in] sm - SM index
     * \param[in] wp - warp index
     * \param[out] handle - the returned breakpoint handle
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa disableBreakpoint
     * \sa enableBreakpoint
     * \sa insertBreakpoint
     * \sa isBreakpointEnabled
     * \sa readBrokenWarps
     * \sa removeBreakpoint
     */
    CUDBGResult (*getWarpHitBreakpoint)(uint32_t dev,
                                        uint32_t sm,
                                        uint32_t wp,
                                        CUDBGBreakpointHandle* handle);

    /**
     * \fn CUDBGAPI_st::resumeWarpsUntilPC
     * \brief Insert a temporary breakpoint at the specified virtual PC and resume all warps in the
     * specified bitmask on a given SM.
     *
     * Compared to resumeDevice(), this method provides finer-grain control by resuming a selected
     * set of warps on the same SM.
     * The main intended usage is to accelerate the single-stepping process when the target PC is
     * known in advance. Instead of single-stepping each warp individually until the target PC is
     * hit, the client can use this method.
     * If an unsteppable barrier is hit by the resumed warps, this method returns early (before
     * reaching the target PC).
     * When this method is used, errors within CUDA kernels will no longer be reported precisely.
     * In the situation where resuming warps is not possible, this method will return
     * CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE. The client should then fall back to using
     * singleStepWarp() or resumeDevice().
     *
     * Since CUDA 13.2.
     *
     * \ingroup EXEC
     *
     * \param[in] dev - device index
     * \param[in] sm - the SM index
     * \param[in] warpMask - the bitmask of warps to resume (1 = resume, 0 = do not resume)
     * \param[in] virtPC - the virtual PC where the temporary breakpoint will be inserted
     * \param[in] flags - flags of type CUDBGSingleStepFlags to change the stepping behavior
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_INVALID_CONTEXT,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE,
     * \return CUDBG_ERROR_INVALID_WARP_MASK,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa resumeAllDevices
     * \sa singleStepWarp
     */
    CUDBGResult (*resumeWarpsUntilPC)(uint32_t dev,
                                      uint32_t sm,
                                      uint64_t warpMask,
                                      uint64_t pc,
                                      uint32_t flags);

    /**
     * \fn CUDBGAPI_st::suspendAllDevices
     * \brief Suspend all running CUDA devices.
     *
     * If the nonBlocking flag is non-zero, the function returns immediately and sends
     * CUDBG_EVENT_ALL_DEVICES_SUSPENDED when the operation finishes in the background.
     * Otherwise, if the function returns with CUDBG_SUCCESS, that guarantees that all devices have
     * been suspended.
     *
     * Since CUDA 13.2.
     *
     * \ingroup EXEC
     *
     * \param[in] nonBlocking - whether or not asynchronous operation is desired
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_SUSPENDED_DEVICE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa resumeAllDevices
     */
    CUDBGResult (*suspendAllDevices)(uint32_t nonBlocking);

    /**
     * \fn CUDBGAPI_st::resumeAllDevices
     * \brief Resume all running CUDA devices.
     *
     * Since CUDA 13.2.
     *
     * \ingroup EXEC
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa suspendAllDevices
     */
    CUDBGResult (*resumeAllDevices)();

    /* 13.4 Extensions */

    /**
     * \fn CUDBGAPI_st::getBindlessConstAddress
     * \brief Get bindless constant GPU VA and size from its header.
     *
     * Since CUDA 13.4.
     *
     * This function does not validate the provided header. If the header is invalid,
     * the function would still return success, but the returned address and size would be invalid.
     *
     * \ingroup READ
     *
     * \param[in] dev - device index
     * \param[in] header - bindless constant header
     * \param[out] address - GPU VA of the start of the mapped bindless constant memory region
     * \param[out] size - size of the mapped bindless constant memory region
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa getConstBankAddress
     */
    CUDBGResult (*getBindlessConstAddress)(uint32_t dev, uint64_t header, uint64_t* address, uint32_t* size);

    /**
     * \fn CUDBGAPI_st::setCudaLogRules
     * \brief Set CUDA log filtering rules.
     *
     * All rules are applied in order, the first matching rule wins. If no rules match, the log
     * message is excluded. The log level filter for each rule is applied by an equality test with
     * the log level of the message. The API user can implement log level filtering in a flag enum
     * style or nested levels style on its side.
     *
     * The default ruleset is:
     * \code
     *     CUDBGCudaLogRule{CUDBG_CUDA_LOG_RULE_ACTION_SEND_ASYNC, CUDBG_CUDA_LOG_LEVEL_FILTER_ANY, 0, NULL}
     * \endcode
     * meaning send all CUDA log messages to the API client asynchronously. The default ruleset index is 0.
     *
     * \note This function returns immediately and the new rules are applied asynchronously in the
     * debuggee process. It's possible that the log messages sent immediately after calling this
     * function are not affected by the new rules.
     *
     * Since CUDA 13.4.
     *
     * \ingroup READ
     *
     * \param[in] rules - array of log filtering rules
     * \param[in] numRules - number of log filtering rules in the array
     * \param[out] newRulesetIndex - the index of the new ruleset, guaranteed to be unique and increasing,
     *                               can be passed as NULL if the caller is not interested in the index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa consumeCudaLogs
     */
    CUDBGResult (*setCudaLogRules)(const CUDBGCudaLogRule* rules,
                                   uint32_t numRules,
                                   uint32_t* newRulesetIndex);

    /**
     * \fn CUDBGAPI_st::consumeCudaLogs
     * \brief Get CUDA error log entries.
     *
     * This consumes the log entries, so they will not be available in subsequent calls.
     * This functionality is only available if the CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS
     * capability is enabled.
     *
     * This function can be called from the notification callback function.
     *
     * Since CUDA 13.4.
     *
     * \ingroup READ
     *
     * \param[out] logMessages - client-allocated array to store log entries
     * \param[in] numMessages - capacity of the logMessages array, in number of elements
     * \param[out] numConsumed - returned number of entries written to logMessages
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa setCudaLogRules
     */
    CUDBGResult (*consumeCudaLogs)(CUDBGCudaLogMessage* logMessages,
                                   uint32_t numMessages,
                                   uint32_t* numConsumed);

    /**
     * \fn CUDBGAPI_st::readRpcRegisters
     * \brief Read the RPC.LO and/or RPC.HI hardware registers for a specific thread.
     *
     * These registers are normally used by hardware to store the program counter of diverged
     * threads. In some cases the compiler may also spill data into them for active threads.
     * These registers can be read for both active and diverged threads.
     *
     * Passing NULL for \p rpcLo or \p rpcHi causes that register to be skipped.
     * Passing NULL for both is an error.
     *
     * Since CUDA 13.4.
     *
     * \ingroup READ
     *
     * \param[in]  dev    - device index
     * \param[in]  sm     - SM index
     * \param[in]  wp     - warp index
     * \param[in]  ln     - lane index
     * \param[out] rpcLo  - receives RPC.LO (lower 32 bits), or NULL to skip
     * \param[out] rpcHi  - receives RPC.HI (upper 32 bits), or NULL to skip
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa writeRpcRegisters
     */
    CUDBGResult (*readRpcRegisters)(uint32_t dev,
                                    uint32_t sm,
                                    uint32_t wp,
                                    uint32_t ln,
                                    uint32_t* rpcLo,
                                    uint32_t* rpcHi);

    /**
     * \fn CUDBGAPI_st::writeRpcRegisters
     * \brief Write the RPC.LO and/or RPC.HI hardware registers for a specific active thread.
     *
     * Only active (non-diverged) threads may be written. This means that it's impossible to use this method
     * to change the PC of a diverged thread, but also that it's impossible to change the PC of an active
     * thread with it since the active thread does not restore its PC from the RPC registers.
     *
     * Passing NULL for \p rpcLo or \p rpcHi causes that register to be skipped.
     * Passing NULL for both is an error.
     *
     * Since CUDA 13.4.
     *
     * \ingroup WRITE
     *
     * \param[in] dev    - device index
     * \param[in] sm     - SM index
     * \param[in] wp     - warp index
     * \param[in] ln     - lane index
     * \param[in] rpcLo  - new value for RPC.LO (lower 32 bits), or NULL to skip
     * \param[in] rpcHi  - new value for RPC.HI (upper 32 bits), or NULL to skip
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INITIALIZATION_FAILURE,
     * \return CUDBG_ERROR_RECURSIVE_API_CALL
     *
     * \sa readRpcRegisters
     */
    CUDBGResult (*writeRpcRegisters)(uint32_t dev,
                                     uint32_t sm,
                                     uint32_t wp,
                                     uint32_t ln,
                                     const uint32_t* rpcLo,
                                     const uint32_t* rpcHi);
};

#ifdef __cplusplus
}
#endif

#endif
