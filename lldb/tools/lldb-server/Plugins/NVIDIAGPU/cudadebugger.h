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

/************************** Files Information *****************************/

/**
 * \file cudadebugger.h
 * \brief Header file for the CUDA debugger API.
 */


/******************************* CUDBGAPI_st ******************************/

/**
 * \struct CUDBGAPI_st
 * \brief The CUDA debugger API routines.
 */

/****************************** CUDBGResult *******************************/

/** \enum CUDBGResult
 * \brief Result values of all the API routines.
 * \ingroup GENERAL */
/** \var CUDBGResult CUDBG_SUCCESS
 *  \brief The API call executed successfully. */
/** \var CUDBGResult CUDBG_ERROR_UNKNOWN
 *  \brief Error type not listed below. */
/** \var CUDBGResult CUDBG_ERROR_BUFFER_TOO_SMALL
 *  \brief Cannot copy all the queried data into the buffer argument. */
/** \var CUDBGResult CUDBG_ERROR_UNKNOWN_FUNCTION
 *  \brief Function cannot be found in the CUDA kernel. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_ARGS
 *  \brief Wrong use of arguments (NULL pointer, illegal value,....). */
/** \var CUDBGResult CUDBG_ERROR_UNINITIALIZED
 *  \brief Debugger API has not yet been properly initialized. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_COORDINATES
 *  \brief Invalid block or thread coordinates were provided. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_MEMORY_SEGMENT
 *  \brief Invalid memory segment requested. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_MEMORY_ACCESS
 *  \brief Requested address (+size) is not within proper segment boundaries. */
/** \var CUDBGResult CUDBG_ERROR_MEMORY_MAPPING_FAILED
 *  \brief Memory is not mapped and cannot be mapped. */
/** \var CUDBGResult CUDBG_ERROR_INTERNAL
 *  \brief A debugger internal error occurred. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_DEVICE
 *  \brief Specified device cannot be found. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_SM
 *  \brief Specified sm cannot be found. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_WARP
 *  \brief Specified warp cannot be found. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_LANE
 *  \brief Specified lane cannot be found. */
/** \var CUDBGResult CUDBG_ERROR_SUSPENDED_DEVICE
 *  \brief The requested operation is not allowed when the device is suspended. */
/** \var CUDBGResult CUDBG_ERROR_RUNNING_DEVICE
 *  \brief Device is running and not suspended. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_ADDRESS
 *  \brief Address is out-of-range. */
/** \var CUDBGResult CUDBG_ERROR_INCOMPATIBLE_API
 *  \brief The requested API is not available. */
/** \var CUDBGResult CUDBG_ERROR_INITIALIZATION_FAILURE
 *  \brief The API could not be initialized. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_GRID
 *  \brief The specified grid is not valid. */
/** \var CUDBGResult CUDBG_ERROR_NO_EVENT_AVAILABLE
 *  \brief The event queue is empty and there is no event left to be processed. */
/** \var CUDBGResult CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED
 *  \brief Some devices were excluded because they have a watchdog associated with them. */
/** \var CUDBGResult CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED
 *  \brief All devices were exclude because they have a watchdog associated with them. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_ATTRIBUTE
 *  \brief Specified attribute does not exist or is incorrect */
/** \var CUDBGResult CUDBG_ERROR_ZERO_CALL_DEPTH
 *  \brief No function calls have been made on the device */
/** \var CUDBGResult CUDBG_ERROR_INVALID_CALL_LEVEL
 *  \brief Specified call level is invalid */
/** \var CUDBGResult CUDBG_ERROR_COMMUNICATION_FAILURE
 *  \brief Communication error between the debugger and the application. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_CONTEXT
 *  \brief Specified context cannot be found. */
/** \var CUDBGResult CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
 *  \brief Requested address was not originally allocated from device memory (most likely visible in system memory). */
/** \var CUDBGResult CUDBG_ERROR_MEMORY_UNMAPPING_FAILED
 *  \brief Requested address is not mapped and cannot be unmapped. */
/** \var CUDBGResult CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER
 *  \brief The display driver is incompatible with the API. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_MODULE
 *  \brief The specified module is not valid */
/** \var CUDBGResult CUDBG_ERROR_LANE_NOT_IN_SYSCALL
 *  \brief The specified lane is not inside a device syscall. */
/** \var CUDBGResult CUDBG_ERROR_MEMCHECK_NOT_ENABLED
 *  \brief Memcheck has not been enabled. */
/** \var CUDBGResult CUDBG_ERROR_INVALID_ENVVAR_ARGS
 *  \brief Some environment variable's value is invalid. */
/** \var CUDBGResult CUDBG_ERROR_OS_RESOURCES
 *  \brief Error while allocating resources from the OS */
/** \var CUDBGResult CUDBG_ERROR_FORK_FAILED
 *  \brief Error while forking the debugger process */
/** \var CUDBGResult CUDBG_ERROR_NO_DEVICE_AVAILABLE
 *  \brief No CUDA capable device was found */
/** \var CUDBGResult CUDBG_ERROR_ATTACH_NOT_POSSIBLE
 *  \brief Attaching to the CUDA program is not possible */
/** \var CUDBGResult CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS
 *  \brief Specified device pointer cannot be resolved to a GPU unambiguously because it is valid on more than one GPU */

/****************************** CUDBGException****************************/

/** \enum CUDBGException_t
 *  \brief Harwdare Exception Types. */
/** \var CUDBGException_t CUDBG_EXCEPTION_UNKNOWN
 * \brief Reported if we do not know what exception the chip has hit (global error) */
/** \var CUDBGException_t CUDBG_EXCEPTION_NONE
 * \brief Reported when there is no exception on the chip (no error) */
/** \var CUDBGException_t CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS
 * \brief Reported when memcheck(enabled within cuda-gdb) finds access violations (lane error: precise software generated exception) */
/** \var CUDBGException_t CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS
 * \brief This error is deprecated as of CUDA 9.0 */
/** \var CUDBGException_t CUDBG_EXCEPTION_LANE_INVALID_ATOMSYS
 * \brief Reported when memcheck(enabled within cuda-gdb) finds system-scoped atomic to a location where it's not allowed (lane error: precise software generated exception) */
/** \var CUDBGException_t CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW
 * \brief Reported from user (data) stack overflow checks in each function's prologue (lane error: precise software generated exception, ABI-only) */
/** \var CUDBGException_t CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW
 * \brief Reported if CRS overflows (global error: the warp that caused this will terminate) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION
 * \brief Reported when any lane in a warp executes an illegal instruction (warp error: invalid branch target, invalid opcode, misaligned/oor reg, invalid immediates, etc.) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS
 * \brief Reported when any lane in a warp accesses memory that is out of range (warp error: lmem_lo/hi, shared, and 40-bit va accesses) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS
 * \brief Reported when any lane in a warp accesses memory that is misaligned (warp error: lmem_lo/hi, shared, and 40-bit va accesses) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE
 * \brief Reported when any lane in a warp executes an instruction that accesses a memory space that is not permitted for that instruction (warp error) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_INVALID_PC
 * \brief Reported when any lane in a warp advances its PC beyond the valid address space (warp error) */
/** \var CUDBGException_t CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW
 * \brief Reported when any lane in a warp hits (uncommon) stack issues (warp error: stack error or api stack overflow) */
/** \var CUDBGException_t CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS
 * \brief Reported when MMU detects an error (global error: L1 error status field is set in the global esr -- for the most part this catches errors SM couldn't catch with oor address detection) */
/** \var CUDBGException_t CUDBG_EXCEPTION_LANE_MISALIGNED_ADDRESS
 * \brief Reported when memcheck(enabled within cuda-gdb) finds access violations (lane error: precise software generated exception) */

/***************************** CUDBGAttribute ****************************/

/** \enum CUDBGAttribute
 *  \brief Query attribute. */
/** \var CUDBGAttribute CUDBG_ATTR_GRID_LAUNCH_BLOCKING
 * \brief whether the launch is synchronous or not. */
/** \var CUDBGAttribute CUDBG_ATTR_GRID_TID
 * \brief The id of the host thread that launched the grid. */

/**************************** CUDBGKernelType ****************************/

/** \enum CUDBGKernelType
 *  \brief Kernel types. */
/** \var CUDBGKernelType CUDBG_KNL_TYPE_UNKNOWN
 * \brief Unknown kernel type. Fall-back value. */
/** \var CUDBGKernelType CUDBG_KNL_TYPE_SYSTEM
 * \brief System kernel, launched by the CUDA driver (cudaMemset, ...). */
/** \var CUDBGKernelType CUDBG_KNL_TYPE_APPLICATION
 * \brief Application kernel, launched by the application. */

/************************* Elf Image Properties **************************/

/** \enum CUDBGElfImageProperties
 *  \brief ELF Image Properties. */
/** \var CUDBGElfImageProperties CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM
 * \brief ELF image contains system kernels, launched by the CUDA driver. */

/**************************** CUDBGKernelOrigin **************************/

/** \enum CUDBGKernelOrigin
 *  \brief Kernel origin. */
/** \var CUDBGKernelOrigin CUDBG_KNL_ORIGIN_CPU
 * \brief The kernel was launched from the CPU. */
/** \var CUDBGKernelOrigin CUDBG_KNL_ORIGIN_GPU
 * \brief The kernel was launched from the GPU. */

/**************** CUDBGKernelLaunchNotifyMode  **************************/

/** \enum CUDBGKernelLaunchNotifyMode
 *  \brief Kernel launch notification mode */
/** \var CUDBGKernelLaunchNotifyMode CUDBG_KNL_LAUNCH_NOTIFY_EVENT
 * \brief Kernel launches generate launch notification events */
/** \var CUDBGKernelLaunchNotifyMode CUDBG_KNL_ORIGIN_NOTIFY_DEFER
 * \brief Kernel launches do not generate any notification  */

/**************************** CUDBGAdjAddrAction *************************/

/** \enum CUDBGAdjAddrAction
 *  \brief Describes which adjusted code address is to be returned. */
/** \var CUDBGAdjAddrAction CUDBG_ADJ_PREVIOUS_ADDRESS
 * \brief The adjusted previous code address is to be returned. */
/** \var CUDBGAdjAddrAction CUDBG_ADJ_NEXT_ADDRESS
 * \brief The adjusted next code address is to be returned. */
/** \var CUDBGAdjAddrAction CUDBG_ADJ_CURRENT_ADDRESS
 * \brief The adjusted current code address is to be returned. */

/****************************** CUDBGRegClass ****************************/

/** \enum CUDBGRegClass
 *  \brief Physical register types. */
/** \var CUDBGRegClass REG_CLASS_INVALID
 * \brief The physical register is invalid. */
/** \var CUDBGRegClass REG_CLASS_REG_CC
 * \brief The physical register is a condition code register. Unused. */
/** \var CUDBGRegClass REG_CLASS_REG_PRED
 * \brief The physical register is a predicate register. Unused. */
/** \var CUDBGRegClass REG_CLASS_REG_ADDR
 * \brief The physical register is an address register. Unused. */
/** \var CUDBGRegClass REG_CLASS_REG_HALF
 * \brief The physical register is a 16-bit register. Unused. */
/** \var CUDBGRegClass REG_CLASS_REG_FULL
 * \brief The physical register is a 32-bit register. */
/** \var CUDBGRegClass REG_CLASS_MEM_LOCAL
 * \brief The content of the physical register has been spilled to memory. */
/** \var CUDBGRegClass REG_CLASS_LMEM_REG_OFFSET
 * \brief The content of the physical register has been spilled to the local stack (ABI only). */

/***************************** CUDBGEventKind ****************************/

/** \enum CUDBGEventKind
    \brief CUDA Kernel Events.
    \ingroup EVENT */
/** \var CUDBGEventKind CUDBG_EVENT_INVALID
    \brief Invalid event. */
/** \var CUDBGEventKind CUDBG_EVENT_ELF_IMAGE_LOADED
    \brief The ELF image for a CUDA source module is available. */
/** \var CUDBGEventKind CUDBG_EVENT_KERNEL_READY
    \brief A CUDA kernel is about to be launched. */
/** \var CUDBGEventKind CUDBG_EVENT_INTERNAL_ERROR
    \brief An internal error occur. The debugging framework may be unstable. */
/** \var CUDBGEventKind CUDBG_EVENT_KERNEL_FINISHED
    \brief A CUDA kernel has terminated. */
/** \var CUDBGEventKind CUDBG_EVENT_CTX_PUSH
    \brief A CUDA context was pushed. */
/** \var CUDBGEventKind CUDBG_EVENT_CTX_POP
    \brief A CUDA CTX was popped. */
/** \var CUDBGEventKind CUDBG_EVENT_CTX_CREATE
    \brief A CUDA CTX was created. */
/** \var CUDBGEventKind CUDBG_EVENT_CTX_DESTROY
    \brief A CUDA context was destroyed. */
/** \var CUDBGEventKind CUDBG_EVENT_TIMEOUT
    \brief An timeout event is sent at regular interval. This event can safely ge ignored. */
/** \var CUDBGEventKind CUDBG_EVENT_ATTACH_COMPLETE
    \brief The attach process has completed and debugging of device code may start. */

/******************************* CUDBGEvent ******************************/

/** \struct CUDBGEvent
    \brief Event information container.
    \ingroup EVENT */
/** \var CUDBGEvent::kind
    \brief Event type. */
/** \var CUDBGEvent::cases
    \brief Information for each type of event. */

/** \union CUDBGEvent::cases_st */
/** \var CUDBGEvent::cases_st::elfImageLoaded
    \brief Information about the loaded ELF image. */
/** \var CUDBGEvent::cases_st::kernelReady
    \brief Information about the kernel ready to be launched. */
/** \var CUDBGEvent::cases_st::kernelFinished
    \brief Information about the kernel that just terminated. */
/** \var CUDBGEvent::cases_st::contextPush
    \brief Information about the context being pushed. */
/** \var CUDBGEvent::cases_st::contextPop
    \brief Information about the context being popped. */
/** \var CUDBGEvent::cases_st::contextCreate
    \brief Information about the context being created. */
/** \var CUDBGEvent::cases_st::contextDestroy
    \brief Information about the context being destroyed. */
/** \var CUDBGEvent::cases_st::internalError
    \brief Information about internal erros. */

/** \struct CUDBGEvent::cases_st::elfImageLoaded_st */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::relocatedElfImage
    \brief pointer to the relocated ELF image for a CUDA source module. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::nonRelocatedElfImage
    \brief pointer to the non-relocated ELF image for a CUDA source module. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::size32
    \brief size of the ELF image (32-bit).
    \deprecated in CUDA 4.0. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::dev
    \brief device index of the kernel. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::context
    \brief context of the kernel. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::module
    \brief module of the kernel. */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::size
    \brief size of the ELF image (64-bit). */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::handle
    \brief ELF image handle */
/** \var CUDBGEvent::cases_st::elfImageLoaded_st::properties
    \brief ELF image properties */

/** \struct CUDBGEvent::cases_st::kernelReady_st */
/** \var CUDBGEvent::cases_st::kernelReady_st::dev
    \brief device index of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::gridId
    \brief grid index of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::tid
    \brief host thread id (or LWP id) of the thread hosting the kernel (Linux only). */
/** \var CUDBGEvent::cases_st::kernelReady_st::context
    \brief context of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::module
    \brief module of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::function
    \brief function of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::functionEntry
    \brief entry PC of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::gridDim
    \brief grid dimensions of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::blockDim
    \brief block dimensions of the kernel. */
/** \var CUDBGEvent::cases_st::kernelReady_st::type
    \brief the type of the kernel: system or application. */
/** \var CUDBGEvent::cases_st::kernelReady_st::parentGridId
    \brief 64-bit grid index of the parent grid. */
/** \var CUDBGEvent::cases_st::kernelReady_st::gridId64
    \brief 64-bit grid index of the kernel. */

/** \struct CUDBGEvent::cases_st::kernelFinished_st */
/** \var CUDBGEvent::cases_st::kernelFinished_st::dev
    \brief device index of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::gridId
    \brief grid index of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::tid
    \brief host thread id (or LWP id) of the thread hosting the kernel (Linux only). */
/** \var CUDBGEvent::cases_st::kernelFinished_st::context
    \brief context of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::module
    \brief module of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::function
    \brief function of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::functionEntry
    \brief entry PC of the kernel. */
/** \var CUDBGEvent::cases_st::kernelFinished_st::gridId64
    \brief 64-bit grid index of the kernel. */

/** \struct CUDBGEvent::cases_st::contextPush_st */
/** \var CUDBGEvent::cases_st::contextPush_st::dev
    \brief device index of the context. */
/** \var CUDBGEvent::cases_st::contextPush_st::tid
    \brief host thread id (or LWP id) of the thread hosting the context (Linux only). */
/** \var CUDBGEvent::cases_st::contextPush_st::context
    \brief the context being pushed. */

/** \struct CUDBGEvent::cases_st::contextPop_st */
/** \var CUDBGEvent::cases_st::contextPop_st::dev
    \brief device index of the context. */
/** \var CUDBGEvent::cases_st::contextPop_st::tid
    \brief host thread id (or LWP id) of the thread hosting the context (Linux only). */
/** \var CUDBGEvent::cases_st::contextPop_st::context
    \brief the context being popped. */

/** \struct CUDBGEvent::cases_st::contextCreate_st */
/** \var CUDBGEvent::cases_st::contextCreate_st::dev
    \brief device index of the context. */
/** \var CUDBGEvent::cases_st::contextCreate_st::tid
    \brief host thread id (or LWP id) of the thread hosting the context (Linux only). */
/** \var CUDBGEvent::cases_st::contextCreate_st::context
    \brief the context being created. */

/** \struct CUDBGEvent::cases_st::contextDestroy_st */
/** \var CUDBGEvent::cases_st::contextDestroy_st::dev
    \brief device index of the context. */
/** \var CUDBGEvent::cases_st::contextDestroy_st::tid
    \brief host thread id (or LWP id) of the thread hosting the context (Linux only). */
/** \var CUDBGEvent::cases_st::contextDestroy_st::context
    \brief the context being destroyed. */

/** \struct CUDBGEvent::cases_st::internalError_st */
/** \var CUDBGEvent::cases_st::internalError_st::errorType
    \brief Type of the internal error. */

/****************************** Event Callback ***************************/

/** \typedef CUDBGNotifyNewEventCallback31
    \brief function type of the function called to notify debugger of the presence of a new event in the event queue.
    \deprecated in CUDA 3.2.
    \ingroup EVENT */

/** \typedef CUDBGNotifyNewEventCallback
    \brief function type of the function called to notify debugger of the presence of a new event in the event queue.
    \ingroup EVENT */

/** \struct CUDBGEventCallbackData40
    \brief Event information passed to callback set with setNotifyNewEventCallback function.
    \deprecated in CUDA 4.1.
    \ingroup EVENT */
/** \var CUDBGEventCallbackData40::tid
    \brief Host thread id of the context generating the event. Zero if not available. */

/** \struct CUDBGEventCallbackData
    \brief Event information passed to callback set with setNotifyNewEventCallback function.
    \ingroup EVENT */
/** \var CUDBGEventCallbackData::tid
    \brief Host thread id of the context generating the event. Zero if not available. */
/** \var CUDBGEventCallbackData::timeout
    \brief A boolean notifying the debugger that the debug API timed while waiting for a reponse from the debugger to a previous event.
           It is up to the debugger to decide what to do in response to a timeout. */

/******************************* Section Order ***************************/

/** \defgroup GENERAL General */
/** \defgroup INIT    Initialization */
/** \defgroup EXEC    Device Execution Control */
/** \defgroup BP      Breakpoints */
/** \defgroup READ    Device State Inspection*/
/** \defgroup WRITE   Device State Alteration */
/** \defgroup GRID    Grid Properties */
/** \defgroup DEV     Device Properties */
/** \defgroup DWARF   DWARF Utilities */

/********************************* Events ********************************/

/** \defgroup EVENT Events

One of those events will create a CUDBGEvent:
\arg the elf image of the current kernel has been loaded and the
     addresses within its DWARF sections have been relocated (and can
     now be used to set breakpoints),
\arg a device breakpoint has been hit,
\arg a CUDA kernel is ready to be launched,
\arg a CUDA kernel has terminated.

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
        case CUDBG_EVENT_KERNEL_READY:
            //...
            break;
        case CUDBG_EVENT_KERNEL_FINISHED:
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

/******************************* Grid Status ******************************/

/** \enum CUDBGGridStatus
 *  \brief Grid status.
 *  \ingroup GRID */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_INVALID
 * \brief  An invalid grid ID was passed, or an error occurred during status lookup. */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_PENDING
 * \brief  The grid was launched but is not running on the HW yet. */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_ACTIVE
 * \brief  The grid is currently running on the HW. */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_SLEEPING
 * \brief  The grid is on the device, doing a join. */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_TERMINATED
 * \brief  The grid has finished executing. */
/** \var CUDBGGridStatus CUDBG_GRID_STATUS_UNDETERMINED
 * \brief  The grid is either QUEUED or TERMINATED. */

/******************************* Grid Info *******************************/

/** \struct CUDBGGridInfo
 *  \brief Grid info.
 *  \ingroup GRID */
/** \var CUDBGGridInfo::dev
 * \brief The index of the device this grid is running on. */
/** \var CUDBGGridInfo::gridId64
 * \brief The 64-bit grid ID of this grid. */
/** \var CUDBGGridInfo::tid
 * \brief The host thread ID that launched this grid. */
/** \var CUDBGGridInfo::context
 * \brief The context this grid belongs to. */
/** \var CUDBGGridInfo::module
 * \brief The module this grid belongs to. */
/** \var CUDBGGridInfo::function
 * \brief The function corresponding to this grid. */
/** \var CUDBGGridInfo::functionEntry
 * \brief The entry address of the function corresponding to this grid. */
/** \var CUDBGGridInfo::gridDim
 * \brief The grid dimensions. */
/** \var CUDBGGridInfo::blockDim
 * \brief The block dimensions. */
/** \var CUDBGGridInfo::type
 * \brief The type of the grid. */
/** \var CUDBGGridInfo::parentGridId
 * \brief The 64-bit grid ID that launched this grid. */
/** \var CUDBGGridInfo::origin
 * \brief The origin of this grid, CPU or GPU. */

/******************************* CUDA Log Message *******************************/

/** \enum CUDBGCudaLogLevel
 * \brief CUDA Log severity level. 
 * \ingroup GENERAL */
/** \var CUDBGCudaLogLevel CUDBG_CUDA_LOG_LEVEL_INVALID
 * \brief Invalid log level. */
/** \var CUDBGCudaLogLevel CUDBG_CUDA_LOG_LEVEL_ERROR
 * \brief Error log level, matches CU_LOG_LEVEL_ERROR. */
/** \var CUDBGCudaLogLevel CUDBG_CUDA_LOG_LEVEL_WARNING
 * \brief Warning log level, matches CU_LOG_LEVEL_WARNING. */

/** \struct CUDBGCudaLogMessage
 *  \brief CUDA Log Message.
 *  \ingroup GENERAL */
/** \var CUDBGCudaLogMessage::unixTimestampNs
 * \brief The timestamp the log message was received at, in nanoseconds since the Unix epoch. */
/** \var CUDBGCudaLogMessage::osThreadId
 * \brief The OS thread ID of the thread that generated the log message. */
/** \var CUDBGCudaLogMessage::logLevel
 * \brief The log severity level of the message. */
/** \var CUDBGCudaLogMessage::message
 * \brief The log message string. */

/******************************* Main Page ********************************/

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

This shows variable 'c' has address class 2 (ptxRegStorage) and its
location can be found by decoding the ULEB128 value, DW_OP_regx:
160631632185.  See cuda-tdep.c in the cuda-gdb source drop for
information on decoding this value and how to obtain which physical
register holds this variable during a specific device PC range. Access
to physical registers liveness information requires a 0-based physical
PC. See the API call:
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


/*--------------------------------- Includes --------------------------------*/

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

#define CUDBG_API_VERSION_MAJOR      13 /* Major release version number */
#define CUDBG_API_VERSION_MINOR       0 /* Minor release version number */
#define CUDBG_API_VERSION_REVISION  163 /* Revision (build) number */

/*---------------------------------- Constants -------------------------------*/

#define CUDBG_MAX_DEVICES       64 /* Maximum number of supported devices */
#define CUDBG_MAX_SMS          256 /* Maximum number of SMs per device */
#define CUDBG_MAX_WARPS         64 /* Maximum number of warps per SM */
#define CUDBG_MAX_LANES         32 /* Maximum number of lanes per warp */
#define CUDBG_MAX_WARP_BARRIERS 16 /* Maximum number of convergence barriers per warp */
#define CUDBG_MAX_LOG_LEN      256 /* Maximum length of a single CUDA log message */

/*----------------------- Thread/Block Coordinates Types ---------------------*/

typedef struct { uint32_t x, y; }    CuDim2;   /* DEPRECATED */
typedef struct { uint32_t x, y, z; } CuDim3;   /* 3-dimensional coordinates for threads,... */

/*--------------------- Memory Segments (as used in DWARF) -------------------*/

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

#define CUDBG_IPC_FLAG_NAME                 cudbgIpcFlag
#define CUDBG_RPC_ENABLED                   cudbgRpcEnabled
#define CUDBG_APICLIENT_PID                 cudbgApiClientPid
#define CUDBG_DEBUGGER_INITIALIZED          cudbgDebuggerInitialized
#define CUDBG_APICLIENT_REVISION            cudbgApiClientRevision
#define CUDBG_SESSION_ID                    cudbgSessionId
#define CUDBG_ATTACH_HANDLER_AVAILABLE      cudbgAttachHandlerAvailable
#define CUDBG_DETACH_SUSPENDED_DEVICES_MASK cudbgDetachSuspendedDevicesMask
#define CUDBG_ENABLE_LAUNCH_BLOCKING        cudbgEnableLaunchBlocking
#define CUDBG_ENABLE_INTEGRATED_MEMCHECK    cudbgEnableIntegratedMemcheck
#define CUDBG_ENABLE_PREEMPTION_DEBUGGING   cudbgEnablePreemptionDebugging
#define CUDBG_RESUME_FOR_ATTACH_DETACH      cudbgResumeForAttachDetach

/*
 * Bitmask of the capabilities supported by the debugger front-end
 */
#define CUDBG_DEBUGGER_CAPABILITIES         cudbgDebuggerCapabilities

/*
 * Can be read to detect whether the external debugger implementation
 * (libcudadebugger.so) is used or not.
 */
#define CUDBG_USE_EXTERNAL_DEBUGGER          cudbgUseExternalDebugger

typedef enum {
    CUDBG_DEBUGGER_CAPABILITY_NONE                  = 0,
    CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING = (1 << 0), /* Static flag: cannot be changed after initialization */
    CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS        = (1 << 1), /* Static flag: cannot be changed after initialization */
    CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS = (1 << 2), /* Static flag: cannot be changed after initialization */
    CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS        = (1 << 3), /* Static flag: cannot be changed after initialization */
    CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS                  = (1 << 4), /* Dynamic flag: can be changed after initialization */
    CUDBG_DEBUGGER_CAPABILITY_COLLECT_CPU_CALL_STACK_FOR_KERNEL_LAUNCHES = (1 << 5), /* Dynamic flag: can be changed after initialization */
} CUDBGCapabilityFlags;

/*---------------- Internal Breakpoint Entries for Error Reporting ------------*/

#define CUDBG_REPORT_DRIVER_API_ERROR                   cudbgReportDriverApiError
#define CUDBG_REPORT_DRIVER_API_ERROR_FLAGS             cudbgReportDriverApiErrorFlags
#define CUDBG_REPORTED_DRIVER_API_ERROR_CODE            cudbgReportedDriverApiErrorCode
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE  cudbgReportedDriverApiErrorFuncNameSize
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR  cudbgReportedDriverApiErrorFuncNameAddr
#define CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE          cudbgReportedDriverApiErrorSource
#define CUDBG_REPORTED_DRIVER_API_ERROR_NAME_SIZE       cudbgReportedDriverApiErrorNameSize
#define CUDBG_REPORTED_DRIVER_API_ERROR_NAME_ADDR       cudbgReportedDriverApiErrorNameAddr
#define CUDBG_REPORTED_DRIVER_API_ERROR_STRING_SIZE     cudbgReportedDriverApiErrorStringSize
#define CUDBG_REPORTED_DRIVER_API_ERROR_STRING_ADDR     cudbgReportedDriverApiErrorStringAddr
#define CUDBG_REPORT_DRIVER_INTERNAL_ERROR              cudbgReportDriverInternalError
#define CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE       cudbgReportedDriverInternalErrorCode

/*----------------------------- API Return Types -----------------------------*/

typedef enum {
    CUDBG_SUCCESS                           = 0x0000,  /* Successful execution */
    CUDBG_ERROR_UNKNOWN                     = 0x0001,  /* Error type not listed below */
    CUDBG_ERROR_BUFFER_TOO_SMALL            = 0x0002,  /* Cannot copy all the queried data into the buffer argument */
    CUDBG_ERROR_UNKNOWN_FUNCTION            = 0x0003,  /* Function cannot be found in the CUDA kernel */
    CUDBG_ERROR_INVALID_ARGS                = 0x0004,  /* Wrong use of arguments (NULL pointer, illegal value,...) */
    CUDBG_ERROR_UNINITIALIZED               = 0x0005,  /* Debugger API has not yet been properly initialized */
    CUDBG_ERROR_INVALID_COORDINATES         = 0x0006,  /* Invalid block or thread coordinates were provided */
    CUDBG_ERROR_INVALID_MEMORY_SEGMENT      = 0x0007,  /* Invalid memory segment requested (read/write) */
    CUDBG_ERROR_INVALID_MEMORY_ACCESS       = 0x0008,  /* Requested address (+size) is not within proper segment boundaries */
    CUDBG_ERROR_MEMORY_MAPPING_FAILED       = 0x0009,  /* Memory is not mapped and cannot be mapped */
    CUDBG_ERROR_INTERNAL                    = 0x000a,  /* A debugger internal error occurred */
    CUDBG_ERROR_INVALID_DEVICE              = 0x000b,  /* Specified device cannot be found */
    CUDBG_ERROR_INVALID_SM                  = 0x000c,  /* Specified sm cannot be found */
    CUDBG_ERROR_INVALID_WARP                = 0x000d,  /* Specified warp cannot be found */
    CUDBG_ERROR_INVALID_LANE                = 0x000e,  /* Specified lane cannot be found */
    CUDBG_ERROR_SUSPENDED_DEVICE            = 0x000f,  /* device is suspended */
    CUDBG_ERROR_RUNNING_DEVICE              = 0x0010,  /* device is running and not suspended */
    CUDBG_ERROR_RESERVED_0                  = 0x0011,  /* Reserved error code */
    CUDBG_ERROR_INVALID_ADDRESS             = 0x0012,  /* address is out-of-range */
    CUDBG_ERROR_INCOMPATIBLE_API            = 0x0013,  /* API version does not match */
    CUDBG_ERROR_INITIALIZATION_FAILURE      = 0x0014,  /* The CUDA Driver failed to initialize */
    CUDBG_ERROR_INVALID_GRID                = 0x0015,  /* Specified grid cannot be found */
    CUDBG_ERROR_NO_EVENT_AVAILABLE          = 0x0016,  /* No event left to be processed */
    CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED    = 0x0017,  /* One or more devices have an associated watchdog (eg. X) */
    CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED     = 0x0018,  /* All devices have an associated watchdog (eg. X) */
    CUDBG_ERROR_INVALID_ATTRIBUTE           = 0x0019,  /* Specified attribute does not exist or is incorrect */
    CUDBG_ERROR_ZERO_CALL_DEPTH             = 0x001a,  /* No function calls have been made on the device */
    CUDBG_ERROR_INVALID_CALL_LEVEL          = 0x001b,  /* Specified call level is invalid */
    CUDBG_ERROR_COMMUNICATION_FAILURE       = 0x001c,  /* Communication error between the debugger and the application. */
    CUDBG_ERROR_INVALID_CONTEXT             = 0x001d,  /* Specified context cannot be found */
    CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM   = 0x001e,  /* Requested address was not originally allocated from device memory (most likely visible in system memory) */
    CUDBG_ERROR_MEMORY_UNMAPPING_FAILED     = 0x001f,  /* Memory is not unmapped and cannot be unmapped */
    CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER = 0x0020,  /* The display driver is incompatible with the API */
    CUDBG_ERROR_INVALID_MODULE              = 0x0021,  /* The specified module is not valid */
    CUDBG_ERROR_LANE_NOT_IN_SYSCALL         = 0x0022,  /* The specified lane is not inside a device syscall */
    CUDBG_ERROR_MEMCHECK_NOT_ENABLED        = 0x0023,  /* Memcheck has not been enabled */
    CUDBG_ERROR_INVALID_ENVVAR_ARGS         = 0x0024,  /* Some environment variable's value is invalid */
    CUDBG_ERROR_OS_RESOURCES                = 0x0025,  /* Error while allocating resources from the OS */
    CUDBG_ERROR_FORK_FAILED                 = 0x0026,  /* Error while forking the debugger process */
    CUDBG_ERROR_NO_DEVICE_AVAILABLE         = 0x0027,  /* No CUDA capable device was found */
    CUDBG_ERROR_ATTACH_NOT_POSSIBLE         = 0x0028,  /* Attaching to the CUDA program is not possible */
    CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE    = 0x0029,  /* The resumeWarpsUntilPC() API is not possible, use resumeDevice() or singleStepWarp() instead */
    CUDBG_ERROR_INVALID_WARP_MASK           = 0x002a,  /* Specified warp mask is zero, or contains invalid warps */
    CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS    = 0x002b,  /* Address cannot be resolved to a GPU unambiguously */
    CUDBG_ERROR_RECURSIVE_API_CALL          = 0x002c,  /* Debug API entry point called from within a debug API callback */
    CUDBG_ERROR_MISSING_DATA                = 0x002d,  /* The requested data is missing */
    CUDBG_ERROR_NOT_SUPPORTED               = 0x002e,  /* Attempted operation is not supported */
} CUDBGResult;

static const char *CUDBGResultNames[] = {
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
    "CUDBG_ERROR_MEMCHECK_NOT_ENABLED",
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
};

static _CUDBG_INLINE const char *cudbgGetErrorString (CUDBGResult error)
{
    if (((unsigned)error)*sizeof(char *) >= sizeof(CUDBGResultNames))
        return "*UNDEFINED*";
    return CUDBGResultNames[(unsigned)error];
}


/*------------------------- API Error Reporting Flags -------------------------*/
typedef enum {
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_NONE = 0x0000, /* Default is that there is no flag */
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_SUPPRESS_NOT_READY = ( 1U << 0 ), /* When set, cudaErrorNotReady/cuErrorNotReady will not be reported */
} CUDBGReportDriverApiErrorFlags;

typedef enum {
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_NONE     = 0x000,   /* Default is that there is no error and no source */
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_DRIVER   = 0x001,   /* The error originates from the CUDA Driver API */
    CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_RUNTIME  = 0x002,   /* The error originates from the CUDA Runtime API */
} CUDBGReportedDriverApiErrorSource;

/*------------------------------ Grid Attributes -----------------------------*/

typedef enum {
    CUDBG_ATTR_GRID_LAUNCH_BLOCKING    = 0x000,   /* Whether the grid launch is blocking or not. */
    CUDBG_ATTR_GRID_TID                = 0x001,   /* Id of the host thread that launched the grid. */
} CUDBGAttribute;

typedef struct {
    CUDBGAttribute attribute;
    uint64_t       value;
} CUDBGAttributeValuePair;

typedef enum {
    CUDBG_GRID_STATUS_INVALID,          /* An invalid grid ID was passed, or an error occurred during status lookup */
    CUDBG_GRID_STATUS_PENDING,          /* The grid was launched but is not running on the HW yet */
    CUDBG_GRID_STATUS_ACTIVE,           /* The grid is currently running on the HW */
    CUDBG_GRID_STATUS_SLEEPING,         /* The grid is on the device, doing a join */
    CUDBG_GRID_STATUS_TERMINATED,       /* The grid has finished executing */
    CUDBG_GRID_STATUS_UNDETERMINED,     /* The grid is either PENDING or TERMINATED */
} CUDBGGridStatus;

/*------------------------------- Kernel Types -------------------------------*/

typedef enum {
    CUDBG_KNL_TYPE_UNKNOWN             = 0x000,   /* Any type not listed below. */
    CUDBG_KNL_TYPE_SYSTEM              = 0x001,   /* System kernel, such as MemCpy. */
    CUDBG_KNL_TYPE_APPLICATION         = 0x002,   /* Application kernel, user-defined or libraries. */
} CUDBGKernelType;

/*--------------------------- Elf Image Properties ---------------------------*/

typedef enum {
    CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM  = 0x001,   /* ELF image contains system kernels. */
} CUDBGElfImageProperties;

/*-------------------------- Physical Register Types -------------------------*/

typedef enum {
    REG_CLASS_INVALID                  = 0x000,   /* invalid register */
    REG_CLASS_REG_CC                   = 0x001,   /* Condition register */
    REG_CLASS_REG_PRED                 = 0x002,   /* Predicate register */
    REG_CLASS_REG_ADDR                 = 0x003,   /* Address register */
    REG_CLASS_REG_HALF                 = 0x004,   /* 16-bit register (Currently unused) */
    REG_CLASS_REG_FULL                 = 0x005,   /* 32-bit register */
    REG_CLASS_MEM_LOCAL                = 0x006,   /* register spilled in memory */
    REG_CLASS_LMEM_REG_OFFSET          = 0x007,   /* register at stack offset (ABI only) */
    REG_CLASS_UREG_PRED                = 0x009,   /* uniform predicate register */
    REG_CLASS_UREG_HALF                = 0x00a,   /* 16-bit uniform register */
    REG_CLASS_UREG_FULL                = 0x00b,   /* 32-bit uniform register */
} CUDBGRegClass;

/*---------------------------- Application Events ----------------------------*/

typedef enum {
    CUDBG_EVENT_INVALID                     = 0x000,   /* Invalid event */
    CUDBG_EVENT_ELF_IMAGE_LOADED            = 0x001,   /* ELF image for CUDA kernel(s) is ready */
    CUDBG_EVENT_KERNEL_READY                = 0x002,   /* A CUDA kernel is ready to be launched */
    CUDBG_EVENT_KERNEL_FINISHED             = 0x003,   /* A CUDA kernel has terminated */
    CUDBG_EVENT_INTERNAL_ERROR              = 0x004,   /* Unexpected error. The API may be unstable. */
    CUDBG_EVENT_CTX_PUSH                    = 0x005,   /* A CUDA context has been pushed. */
    CUDBG_EVENT_CTX_POP                     = 0x006,   /* A CUDA context has been popped. */
    CUDBG_EVENT_CTX_CREATE                  = 0x007,   /* A CUDA context has been created and pushed. */
    CUDBG_EVENT_CTX_DESTROY                 = 0x008,   /* A CUDA context has been, popped if pushed, then destroyed. */
    CUDBG_EVENT_TIMEOUT                     = 0x009,   /* Nothing happened for a while. This is heartbeat event.
                                                            NOTE: Only sent by the classic backend. */
    CUDBG_EVENT_ATTACH_COMPLETE             = 0x00a,   /* Attach complete. */
    CUDBG_EVENT_DETACH_COMPLETE             = 0x00b,   /* Detach complete. */
    CUDBG_EVENT_ELF_IMAGE_UNLOADED          = 0x00c,   /* ELF image for CUDA kernels(s) no longer available */
    CUDBG_EVENT_FUNCTIONS_LOADED            = 0x00d,   /* A group of functions/kernels have been loaded
                                                        *   NOTE: Will only be sent if the debugger capability
                                                        *   CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING is set.
                                                        */
    CUDBG_EVENT_ALL_DEVICES_SUSPENDED       = 0x00e,   /* All CUDA devices have been suspended due to a breakpoint hit
                                                        *   or an exception. Does not get sent for GPU events that
                                                        *   result in synchronous API method calls, such as
                                                        *   singleStepWarp or resumeWarpsUntilPC.
                                                        *   NOTE: Will only be sent if the debugger capability
                                                        *   CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS is set.
                                                        */
    CUDBG_EVENT_CUDA_LOGS_AVAILABLE         = 0x00f,   /* (Async) CUDA Logs are available for the debugger to consume.
                                                        *   After receiving this asynchronous event, debuggers should
                                                        *   drain all available log entries by repeatedly calling
                                                        *   consumeCudaLogs until no more logs are available.
                                                        *   This event is only sent for the first log message
                                                        *   that's generated after the client has read all logs with
                                                        *   consumeCudaLogs.
                                                        *   It is not sent by default, and can be enabled via the
                                                        *   capability CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS.
                                                        */
    CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED = 0x010,   /* (Sync) CUDA Logs buffer has reached an implementation-defined threshold.
                                                        *   The client should call consumeCudaLogs to avoid
                                                        *   excessive log buildup. New logs will still be collected
                                                        *   even if consumeCudaLogs is not called.
                                                        */
} CUDBGEventKind;

/*------------------------------- Kernel Origin ------------------------------*/

typedef enum {
    CUDBG_KNL_ORIGIN_CPU               = 0x000,   /* The kernel was launched from the CPU. */
    CUDBG_KNL_ORIGIN_GPU               = 0x001,   /* The kernel was launched from the GPU. */
} CUDBGKernelOrigin;

/*------------------------ Kernel Launch Notify Mode --------------------------*/

typedef enum {
    CUDBG_KNL_LAUNCH_NOTIFY_EVENT      = 0x000,   /* The kernel notifications generate events */
    CUDBG_KNL_LAUNCH_NOTIFY_DEFER      = 0x001,   /* The kernel notifications are deferred */
} CUDBGKernelLaunchNotifyMode;

/*---------------------- Application Event Queue Type ------------------------*/

typedef enum {
    CUDBG_EVENT_QUEUE_TYPE_SYNC      = 0,   /* Synchronous event queue */
    CUDBG_EVENT_QUEUE_TYPE_ASYNC     = 1,   /* Asynchronous event queue */
} CUDBGEventQueueType;

/*------------------------------ Elf Image Type ------------------------------*/

typedef enum {
    CUDBG_ELF_IMAGE_TYPE_NONRELOCATED      = 0,   /* Non-relocated ELF image */
    CUDBG_ELF_IMAGE_TYPE_RELOCATED         = 1,   /* Relocated ELF image */
} CUDBGElfImageType;

/*------------------------------ Code Address --------------------------------*/

typedef enum {
    CUDBG_ADJ_PREVIOUS_ADDRESS         = 0x000,   /* Get the adjusted previous code address. */
    CUDBG_ADJ_CURRENT_ADDRESS          = 0x001,   /* Get the adjusted current code address. */
    CUDBG_ADJ_NEXT_ADDRESS             = 0x002,   /* Get the adjusted next code address. */
} CUDBGAdjAddrAction;

/*------------------------------ Single Step Flags --------------------------------*/

typedef enum {
    /* Default behavior */
    CUDBG_SINGLE_STEP_FLAGS_NONE                        = 0,
    /* Do not step over warp-wide barriers using a breakpoint and resume,
     * instead perform a single step and return. Passing this flag in means
     * that the API client plans to repeat the singleStepWarp() call until
     * the warp barrier is stepped over. This gives a more precise exception
     * information if an exception is encountered by the diverged threads
     * while stepping. */
    CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS  = (1U << 0),
} CUDBGSingleStepFlags;

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases30_st {
        struct elfImageLoaded30_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size;
        } elfImageLoaded;
        struct kernelReady30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelReady;
        struct kernelFinished30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelFinished;
    } cases;
} CUDBGEvent30;

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases32_st {
        struct elfImageLoaded32_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
        } elfImageLoaded;
        struct kernelReady32_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelReady;
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

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases42_st {
        struct elfImageLoaded42_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady42_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
        } kernelReady;
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

typedef struct {
    CUDBGEventKind kind;
    union cases50_st {
        struct elfImageLoaded50_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady50_st{
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
        } kernelReady;
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

typedef struct {
    CUDBGEventKind kind;
    union cases55_st {
        struct elfImageLoaded55_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady55_st{
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
            uint64_t parentGridId;
            uint64_t gridId64;
            CUDBGKernelOrigin origin;
        } kernelReady;
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

#pragma pack(push,1)
typedef struct {
    CUDBGEventKind kind;
    union cases_st {
        struct elfImageLoaded_st {
            uint32_t dev;
            uint64_t context;
            uint64_t module;
            uint64_t size;
            uint64_t handle;
            uint32_t properties;
        } elfImageLoaded;
        struct elfImageUnloaded_st {
            uint32_t dev;
            uint64_t context;
            uint64_t module;
            uint64_t size;
            uint64_t handle;
        } elfImageUnloaded;
        struct kernelReady_st{
            uint32_t dev;
            uint32_t tid;
            uint64_t gridId;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
            uint64_t parentGridId;
            CUDBGKernelOrigin origin;
        } kernelReady;
        struct kernelFinished_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            uint64_t gridId;
        } kernelFinished;
        struct contextPush_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError_st {
            CUDBGResult errorType;
        } internalError;
        struct functionsLoaded_st {
            uint32_t dev;
            uint32_t count;
            uint64_t context;
            uint64_t module;
        } functionsLoaded;
        struct allDevicesSuspended_st {
            /* This mask has bits set for devices with any warps that hit a breakpoint */
            uint64_t brokenDevicesMask;
            /* This mask has bits set for devices with any warps that hit an exception */
            uint64_t faultedDevicesMask;
        } allDevicesSuspended;
    } cases;
} CUDBGEvent;
#pragma pack(pop)

typedef struct {
    uint32_t tid;
} CUDBGEventCallbackData40;

typedef struct {
    uint32_t tid;
    uint32_t timeout;
} CUDBGEventCallbackData41;

typedef struct {
    void* userData;
    uint32_t tid;
} CUDBGEventCallbackData;

#pragma pack(push,1)
typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3   gridDim;
    CuDim3   blockDim;
    CUDBGKernelType type;
    uint64_t parentGridId;
    CUDBGKernelOrigin origin;
} CUDBGGridInfo55;

typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3   gridDim;
    CuDim3   blockDim;
    CUDBGKernelType type;
    uint64_t parentGridId;
    CUDBGKernelOrigin origin;
    CuDim3   clusterDim;
} CUDBGGridInfo120;

typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3   gridDim;
    CuDim3   blockDim;
    CUDBGKernelType type;
    uint64_t parentGridId;
    CUDBGKernelOrigin origin;
    CuDim3   clusterDim;
    CuDim3   preferredClusterDim;
} CUDBGGridInfo;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct {
    uint64_t sectionIndex;
    uint64_t address;
} CUDBGLoadedFunctionInfo;
#pragma pack(pop)

typedef void (*CUDBGNotifyNewEventCallback31)(void *data);
typedef void (*CUDBGNotifyNewEventCallback40)(CUDBGEventCallbackData40 *data);
typedef void (*CUDBGNotifyNewEventCallback41)(CUDBGEventCallbackData41 *data);
typedef void (*CUDBGNotifyNewEventCallback)(CUDBGEventCallbackData *data);

/*-------------------------------- Exceptions ------------------------------*/

typedef enum {
    CUDBG_EXCEPTION_UNKNOWN = 0xFFFFFFFFU, // Force sizeof(CUDBGException_t)==4
    CUDBG_EXCEPTION_NONE = 0,
    CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS = 1,
    CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW = 2,
    CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW = 3,
    CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION = 4,
    CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS = 5,
    CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS = 6,
    CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE = 7,
    CUDBG_EXCEPTION_WARP_INVALID_PC = 8,
    CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW = 9,
    CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS = 10,
    CUDBG_EXCEPTION_LANE_MISALIGNED_ADDRESS = 11,
    CUDBG_EXCEPTION_WARP_ASSERT = 12,
    CUDBG_EXCEPTION_LANE_SYSCALL_ERROR = 13,
    CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS = 14,
    CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS = 15,
    CUDBG_EXCEPTION_LANE_INVALID_ATOMSYS = 16,
    CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS = 17,
    CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT = 18,
    CUDBG_EXCEPTION_WARP_STACK_CANARY = 19,
    CUDBG_EXCEPTION_WARP_TMEM_ACCESS_CHECK = 20,
    CUDBG_EXCEPTION_WARP_TMEM_LEAK = 21,
    CUDBG_EXCEPTION_WARP_CALL_REQUIRES_NEWER_DRIVER = 22,
} CUDBGException_t;

typedef enum {
    CUDBG_UVM_MEMORY_ACCESS_TYPE_UNKNOWN  = 0xFFFFFFFFU,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_INVALID  = 0,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_READ     = 1,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_WRITE    = 2,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_ATOMIC   = 3,
    CUDBG_UVM_MEMORY_ACCESS_TYPE_PREFETCH = 4,
} CUDBGUvmMemoryAccessType_t;

typedef enum {
    CUDBG_UVM_FAULT_TYPE_UNKNOWN               =  0xFFFFFFFFU,
    CUDBG_UVM_FAULT_TYPE_INVALID               =  0,
    CUDBG_UVM_FAULT_TYPE_INVALID_PDE           =  1,
    CUDBG_UVM_FAULT_TYPE_INVALID_PTE           =  2,
    CUDBG_UVM_FAULT_TYPE_WRITE                 =  3,
    CUDBG_UVM_FAULT_TYPE_ATOMIC                =  4,
    CUDBG_UVM_FAULT_TYPE_INVALID_PDE_SIZE      =  5,
    CUDBG_UVM_FAULT_TYPE_LIMIT_VIOLATION       =  6,
    CUDBG_UVM_FAULT_TYPE_UNBOUND_INST_BLOCK    =  7,
    CUDBG_UVM_FAULT_TYPE_PRIV_VIOLATION        =  8,
    CUDBG_UVM_FAULT_TYPE_PITCH_MASK_VIOLATION  =  9,
    CUDBG_UVM_FAULT_TYPE_WORK_CREATION         = 10,
    CUDBG_UVM_FAULT_TYPE_UNSUPPORTED_APERTURE  = 11,
    CUDBG_UVM_FAULT_TYPE_COMPRESSION_FAILURE   = 12,
    CUDBG_UVM_FAULT_TYPE_UNSUPPORTED_KIND      = 13,
    CUDBG_UVM_FAULT_TYPE_REGION_VIOLATION      = 14,
    CUDBG_UVM_FAULT_TYPE_POISON                = 15,
} CUDBGUvmFaultType_t;

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

/*------------------------------ Warp State --------------------------------*/
#pragma pack(push,1)
typedef struct {
    uint64_t virtualPC;
    CuDim3 threadIdx;
    CUDBGException_t exception;
} CUDBGLaneState;

typedef struct {
    uint64_t gridId;
    uint64_t errorPC;
    CuDim3 blockIdx;
    uint32_t validLanes;
    uint32_t activeLanes;
    uint32_t errorPCValid;
    CUDBGLaneState lane[32];
} CUDBGWarpState60;

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
    uint32_t inSyscallLanes;
} CUDBGWarpState;

typedef struct {
    uint32_t sharedMemSize;
    uint32_t numRegisters;
} CUDBGWarpResources;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct {
    uint64_t startAddress;
    uint64_t size;
} CUDBGMemoryInfo;
#pragma pack(pop)

/*----------------------- Batched device info support ----------------------*/

/* uint32_t sized enum */
typedef enum {
    /* Request state information for all valid SMs/Warps/Lanes */
    CUDBG_RESPONSE_TYPE_FULL,

    /* Request state information for all changed SMs/Warps/Lanes since the last call */
    CUDBG_RESPONSE_TYPE_UPDATE,

    /* Force sizeof(CUDBGDeviceInfoQueryType_t)==4 */
    CUDBG_RESPONSE_TYPE_UNKNOWN = 0xFFFFFFFFU,
} CUDBGDeviceInfoQueryType_t;

/* uint32_t sized enum */
typedef enum {
    /* Mask of updated SMs reported by this response
       Optional: Yes, assume all 1's if absent
       Size: Number of SMs-sized bitmask, rounded up to be divisible by 8 */
    CUDBG_DEVICE_ATTRIBUTE_SM_UPDATE_MASK       = 0,
    /* Mask of SMs with any valid warp
       Optional: No, always returned by the API
       Size: Number of SMs-sized bitmask, rounded up to be divisible by 8 */
    CUDBG_DEVICE_ATTRIBUTE_SM_ACTIVE_MASK       = 1,
    /* Mask of SMs with any warps with exceptions
       Optional: Yes, assume all 0's if absent
       Size: Number of SMs-sized bitmask, rounded up to be divisible by 8 */
    CUDBG_DEVICE_ATTRIBUTE_SM_EXCEPTION_MASK    = 2,

    CUDBG_DEVICE_ATTRIBUTE_COUNT,
} CUDBGDeviceInfoAttribute_t;

/* uint32_t sized enum */
typedef enum {
    /* Mask of updated warps reported by this response
       Optional: Yes, assume all 1's if absent
       Size: uint64_t */
    CUDBG_SM_ATTRIBUTE_WARP_UPDATE_MASK  = 0,

    CUDBG_SM_ATTRIBUTE_COUNT,
} CUDBGSMInfoAttribute_t;

/* uint32_t sized enum */
typedef enum {
    /* Mask of updated lanes reported by this response
       Optional: Yes, assume all 1's if absent
       Size: uint32_t */
    CUDBG_WARP_ATTRIBUTE_LANE_UPDATE_MASK                   = 0,
    /* Signals whether the attribute flags field is present on the lane level for this warp
       Optional: Yes, assume no lane attributes for this warp if absent
       Size: 0 (doesn't have an associated warp-level field) */
    CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES                    = 1,
    /* CUDBGException_t for this warp
       Optional: Yes, assume CUDBG_EXCEPTION_NONE if absent
       Size: uint32_t */
    CUDBG_WARP_ATTRIBUTE_EXCEPTION                          = 2,
    /* Error PC for this warp
       Optional: Yes, assume no error PC is available if absent
       Size: uint64_t */
    CUDBG_WARP_ATTRIBUTE_ERRORPC                            = 3,
    /* Cluster index for this warp
       Optional: Yes if warp is not in a cluster
       Size: CuDim3 */
    CUDBG_WARP_ATTRIBUTE_CLUSTERIDX                         = 4,
    /* Cluster dimensions for this warp
       Optional: Yes if warp is not in a cluster
       Size: CuDim3 */
    CUDBG_WARP_ATTRIBUTE_CLUSTERDIM                         = 5,
    /* For cluster exceptions, this represents the target block index handling
       cluster requests.
       Optional: Yes, assume no block index is available if absent
       Size: CuDim3 */
    CUDBG_WARP_ATTRIBUTE_CLUSTER_EXCEPTION_TARGET_BLOCK_IDX = 6,
    /* Lane mask showing threads that are in a syscall
       Optional: Yes, use readSyscallCallDepth() if this attribute is not present
       Size: uint32_t */
    CUDBG_WARP_ATTRIBUTE_IN_SYSCALL_LANES                   = 7,

    CUDBG_WARP_ATTRIBUTE_COUNT,
} CUDBGWarpInfoAttribute_t;

/* uint32_t sized enum */
typedef enum {
    CUDBG_LANE_ATTRIBUTE_COUNT,
} CUDBGLaneInfoAttribute_t;

/* Sizes of the various structs returned by the batched device update APIs
   No explicit version field - implied by debugAPI major.minor.revision
*/
#pragma pack(push,1)
typedef struct {
    uint32_t requiredBufferSize;
    
    uint32_t deviceInfoSize;
    uint32_t deviceInfoAttributeSizes[32];

    uint32_t smInfoSize;
    uint32_t smInfoAttributeSizes[32];

    uint32_t warpInfoSize;
    uint32_t warpInfoAttributeSizes[32];
 
    uint32_t laneInfoSize;
    uint32_t laneInfoAttributeSizes[32];
} CUDBGDeviceInfoSizes;
#pragma pack(pop)

/* This is the first element in the deviceInfoBuffer, and is always present.
   getDeviceInfo() takes a deviceId as input, so no need to explicitly pass it back here
*/
#pragma pack(push,1)
typedef struct {
    CUDBGDeviceInfoQueryType_t responseType;

    /* Bitmask of CUDBGDeviceInfoAttribute_t enums for a Device */
    uint32_t deviceAttributeFlags;
} CUDBGDeviceInfo;
#pragma pack(pop)

/*
  Only "valid & updated" SMs/Warps/Lanes are included in the buffer, which allows us to determine
  indexes without having to encode an explicit ID field in the following buffer datastructures.
*/ 

/* Represents a SM */
#pragma pack(push,1)
typedef struct {
    uint64_t warpValidMask;
    uint64_t warpBrokenMask;

    /* Bitmask of CUDBGSmInfoAttribute_t enums for a SM */
    uint32_t smAttributeFlags;

    /* New elements are appended (but not added to the struct) */
} CUDBGSMInfo;
#pragma pack(pop)

/* Represents a Warp */
#pragma pack(push,1)
typedef struct {
    uint64_t gridId;

    CuDim3   blockIdx;
    CuDim3   baseThreadIdx;

    uint32_t validLanes;
    uint32_t activeLanes;

    /* Bitmask of CUDBGWarpInfoAttribute_t enums for warps and their lanes */
    uint32_t warpAttributeFlags;

    /* Optional fields based on the "warpAttributeFlags" bitmask */
} CUDBGWarpInfo;
#pragma pack(pop)
 
/* Represents a Lane */
#pragma pack(push,1)
typedef struct {
    uint64_t virtualPC;

    /* Optional: present only if CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES bit
       is set in CUDBGWarpInfo::warpAttributeFlags. Any additional data is
       appended here after this.

       uint32_t laneAttributeFlags;
     */
} CUDBGLaneInfo;
#pragma pack(pop)

/*----------------------- Coredump/snapshot support ------------------------*/

typedef enum {
    CUDBG_COREDUMP_DEFAULT_FLAGS                = 0,
    CUDBG_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = (1 << 0),
    CUDBG_COREDUMP_SKIP_GLOBAL_MEMORY           = (1 << 1),
    CUDBG_COREDUMP_SKIP_SHARED_MEMORY           = (1 << 2),
    CUDBG_COREDUMP_SKIP_LOCAL_MEMORY            = (1 << 3),

    /* The value used to be SKIP_ABORT, but it's impossible to change this behavior.  */
    /* DEPRECATED_VALUE_DO_NOT_USE              = (1 << 4), */

    CUDBG_COREDUMP_SKIP_CONSTBANK_MEMORY        = (1 << 5),

    CUDBG_COREDUMP_LIGHTWEIGHT_FLAGS = CUDBG_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES
                                     | CUDBG_COREDUMP_SKIP_GLOBAL_MEMORY
                                     | CUDBG_COREDUMP_SKIP_SHARED_MEMORY
                                     | CUDBG_COREDUMP_SKIP_LOCAL_MEMORY
                                     | CUDBG_COREDUMP_SKIP_CONSTBANK_MEMORY
} CUDBGCoredumpGenerationFlags;

/*-------------------------------- CBU state -------------------------------*/

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
 
#pragma pack(push,1)
typedef struct
{
    uint32_t activeMask;
    uint32_t exitedMask;
    uint32_t collectiveMask;
    uint32_t barrierMasks[CUDBG_MAX_WARP_BARRIERS];
    CUDBGCbuThreadState threadState[CUDBG_MAX_LANES];
} CUDBGCbuWarpState;
#pragma pack(pop)

typedef enum {
    CUDBG_CUDA_LOG_LEVEL_INVALID = 0xFFFFFFFFU,
    CUDBG_CUDA_LOG_LEVEL_ERROR   = 0,
    CUDBG_CUDA_LOG_LEVEL_WARNING = 1,
} CUDBGCudaLogLevel;

#pragma pack(push,1)
typedef struct
{
    uint64_t unixTimestampNs;
    uint32_t osThreadId;
    CUDBGCudaLogLevel logLevel;
    char message[CUDBG_MAX_LOG_LEN];
} CUDBGCudaLogMessage;
#pragma pack(pop)

/*--------------------------------- Exports --------------------------------*/

typedef const struct CUDBGAPI_st *CUDBGAPI;

/**
 * \brief Get the API associated with the major/minor/revision version numbers.
 *
 * \param major - the major version number
 * \param minor - the minor version number
 * \param rev   - the revision version number
 * \param api - the pointer to the API
 *
 * \return CUDBG_ERROR_INVALID_ARGS,
 * \return CUDBG_SUCCESS,
 * \return CUDBG_ERROR_INCOMPATIBLE_API
 *
 * \sa cudbgGetAPIVersion
 */
CUDBGResult cudbgGetAPI(uint32_t major, uint32_t minor, uint32_t rev, CUDBGAPI *api);

/**
 * \brief Get the API version supported by the CUDA driver.
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
CUDBGResult cudbgGetAPIVersion(uint32_t *major, uint32_t *minor, uint32_t *rev);
CUDBGResult cudbgMain(int apiClientPid, uint32_t apiClientRevision, int sessionId, int attachState,
                      int attachEventInitialized, int writeFd, int detachFd, int attachStubInUse,
                      int enablePreemptionDebugging);
void cudbgApiInit(uint32_t arg);
void cudbgApiAttach(void);
void cudbgApiDetach(void);
void CUDBG_REPORT_DRIVER_API_ERROR(void);
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
extern uint32_t CUDBG_ENABLE_PREEMPTION_DEBUGGING;
extern uint32_t CUDBG_RESUME_FOR_ATTACH_DETACH;
extern uint32_t CUDBG_REPORT_DRIVER_API_ERROR_FLAGS;
extern uint32_t CUDBG_DEBUGGER_CAPABILITIES;

/* Deprecated */
extern uint32_t CUDBG_DETACH_SUSPENDED_DEVICES_MASK;

/* Note this has no effect on virtual GPUs (such as NVIDIA GRID) */
extern uint32_t CUDBG_ENABLE_INTEGRATED_MEMCHECK;

struct CUDBGAPI_st {
    /* Initialization */
    /**
     * \fn CUDBGAPI_st::initialize
     * \brief Initialize the API.
     *
     * Since CUDA 3.0.
     *
     * \ingroup INIT
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNKNOWN
     *
     * \sa finalize
     */
    CUDBGResult (*initialize)(void);

    /**
     * \fn CUDBGAPI_st::finalize
     * \brief Finalize the API and free all memory.
     *
     * Since CUDA 3.0.
     *
     * \ingroup INIT
     *
     * \return ::CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_COMMUNICATION_FAILURE,
     * \return CUDBG_ERROR_UNKNOWN
     *
     * \sa initialize
     */
    CUDBGResult (*finalize)(void);

    /* Device Execution Control */

    /**
     * \fn CUDBGAPI_st::suspendDevice
     * \brief Suspends a running CUDA device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup EXEC
     *
     * \param dev - device index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa resumeDevice
     * \sa singleStepWarp
     */
    CUDBGResult (*suspendDevice)(uint32_t dev);

    /**
     * \fn CUDBGAPI_st::resumeDevice
     * \brief Resume a suspended CUDA device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup EXEC
     *
     * \param dev - device index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa suspendDevice
     * \sa singleStepWarp
     */
    CUDBGResult (*resumeDevice)(uint32_t dev);

    /**
     * \fn CUDBGAPI_st::singleStepWarp40
     * \brief Single step an individual warp on a suspended CUDA device.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 4.1.
     *
     * \ingroup EXEC
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_UNKNOWN,
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE
     *
     * \sa resumeDevice
     * \sa suspendDevice
     * \sa singleStepWarp
     */
    CUDBGResult (*singleStepWarp40)(uint32_t dev, uint32_t sm, uint32_t wp);

    /* Breakpoints */

    /**
     * \fn CUDBGAPI_st::setBreakpoint31
     * \brief Sets a breakpoint at the given instruction address.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_ADDRESS
     *
     * \sa unsetBreakpoint31
     */
    CUDBGResult (*setBreakpoint31)(uint64_t addr);

    /**
     * \fn CUDBGAPI_st::unsetBreakpoint31
     * \brief Unsets a breakpoint at the given instruction address.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa setBreakpoint31
     */
    CUDBGResult (*unsetBreakpoint31)(uint64_t addr);

    /* Device State Inspection */

    /**
     * \fn CUDBGAPI_st::readGridId50
     * \brief Reads the CUDA grid index running on a valid warp.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 5.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param gridId - the returned CUDA grid index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readGridId50)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *gridId);

    /**
     * \fn CUDBGAPI_st::readBlockIdx32
     * \brief Reads the two-dimensional CUDA block index running on a valid warp.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param blockIdx - the returned CUDA block index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readBlockIdx32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *blockIdx);

    /**
     * \fn CUDBGAPI_st::readThreadIdx
     * \brief Reads the CUDA thread index running on valid lane.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param threadIdx - the returned CUDA thread index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readThreadIdx)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);

    /**
     * \fn CUDBGAPI_st::readBrokenWarps
     * \brief Reads the bitmask of warps that are at a breakpoint on a given SM.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param brokenWarpsMask - the returned bitmask of broken warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readBrokenWarps)(uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask);

    /**
     * \fn CUDBGAPI_st::readValidWarps
     * \brief Reads the bitmask of valid warps on a given SM.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param validWarpsMask - the returned bitmask of valid warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readValidWarps)(uint32_t dev, uint32_t sm, uint64_t *validWarpsMask);

    /**
     * \fn CUDBGAPI_st::readValidLanes
     * \brief Reads the bitmask of valid lanes on a given warp.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param validLanesMask - the returned bitmask of valid lanes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readActiveLanes
     */
    CUDBGResult (*readValidLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *validLanesMask);
    /**
     * \fn CUDBGAPI_st::readActiveLanes
     * \brief Reads the bitmask of active lanes on a valid warp.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param activeLanesMask - the returned bitmask of active lanes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     */
    CUDBGResult (*readActiveLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *activeLanesMask);
    /**
     * \fn CUDBGAPI_st::readCodeMemory
     * \brief Reads content at address in the code memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readCodeMemory)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readConstMemory129
     * \brief Reads content at address in the constant memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readCodeMemory
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readConstMemory129)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readGlobalMemory31
     * \brief Reads content at address in the global memory segment.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.2.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readGlobalMemory31)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readParamMemory
     * \brief Reads content at address in the param memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readParamMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readSharedMemory
     * \brief Reads content at address in the shared memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readLocalMemory
     * \sa readTextureMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readSharedMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readLocalMemory
     * \brief Reads content at address in the local memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readLocalMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readRegister
     * \brief Reads content of a hardware register.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param regno - register index
     * \param val - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readPC
     */
    CUDBGResult (*readRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
    /**
     * \fn CUDBGAPI_st::readPC
     * \brief Reads the PC on the given active lane.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param pc - the returned PC
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readVirtualPC
     */
    CUDBGResult (*readPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
    /**
     * \fn CUDBGAPI_st::readVirtualPC
     * \brief Reads the virtual PC on the given active lane.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param pc - the returned PC
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION
     *
     * \sa readPC
     */
    CUDBGResult (*readVirtualPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
    /**
     * \fn CUDBGAPI_st::readLaneStatus
     * \brief Reads the status of the given lane.  For specific error values, use readLaneException.
     *
     * Since CUDA 3.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param error - true if there is an error
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     */
    CUDBGResult (*readLaneStatus)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, bool *error);

    /* Device State Alteration */
    /**
     * \fn CUDBGAPI_st::writeGlobalMemory31
     * \brief Writes content to address in the global memory segment.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.2.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeGlobalMemory31)(uint32_t dev, uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeParamMemory
     * \brief Writes content to address in the param memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa writeGenericMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeParamMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeSharedMemory
     * \brief Writes content to address in the shared memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa writeGenericMemory
     * \sa writeParamMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeSharedMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeLocalMemory
     * \brief Writes content to address in the local memory segment.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED
     *
     * \sa writeGenericMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeLocalMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeRegister
     * \brief Writes content to a hardware register.
     *
     * Since CUDA 3.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param regno - register index
     * \param val - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa writeGenericMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     */
    CUDBGResult (*writeRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);

    /* Grid Properties */
    /**
     * \fn CUDBGAPI_st::getGridDim32
     * \brief Get the number of blocks in the given grid.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param gridDim - the returned number of blocks in the grid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getBlockDim
     */
    CUDBGResult (*getGridDim32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *gridDim);
    /**
     * \fn CUDBGAPI_st::getBlockDim
     * \brief Get the number of threads in the given block.
     *
     * Since CUDA 3.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param blockDim - the returned number of threads in the block
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getGridDim
     */
    CUDBGResult (*getBlockDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockDim);
    /**
     * \fn CUDBGAPI_st::getTID
     * \brief Get the ID of the Linux thread hosting the context of the grid.
     *
     * Since CUDA 3.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param tid - the returned thread id
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getTID)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
    /**
     * \fn CUDBGAPI_st::getElfImage32
     * \brief Get the relocated or non-relocated ELF image and size for the grid on the given device.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param relocated - set to true to specify the relocated ELF image, false otherwise
     * \param *elfImage - pointer to the ELF image
     * \param size - size of the ELF image (32 bits)
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getElfImage32)(uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint32_t *size);

    /* Device Properties */
    /**
     * \fn CUDBGAPI_st::getDeviceType
     * \brief Get the string description of the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param buf - the destination buffer
     * \param sz - the size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getSMType
     */
    CUDBGResult (*getDeviceType)(uint32_t dev, char *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::getSmType
     * \brief Get the SM type of the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param buf - the destination buffer
     * \param sz - the size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getDeviceType
     */
    CUDBGResult (*getSmType)(uint32_t dev, char *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::getNumDevices
     * \brief Get the number of installed CUDA devices.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param numDev - the returned number of devices
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumSMs
     * \sa getNumWarps
     * \sa getNumLanes
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumDevices)(uint32_t *numDev);
    /**
     * \fn CUDBGAPI_st::getNumSMs
     * \brief Get the total number of SMs on the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numSMs - the returned number of SMs
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumDevices
     * \sa getNumWarps
     * \sa getNumLanes
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumSMs)(uint32_t dev, uint32_t *numSMs);
    /**
     * \fn CUDBGAPI_st::getNumWarps
     * \brief Get the number of warps per SM on the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numWarps - the returned number of warps
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumDevices
     * \sa getNumSMs
     * \sa getNumLanes
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumWarps)(uint32_t dev, uint32_t *numWarps);
    /**
     * \fn CUDBGAPI_st::getNumLanes
     * \brief Get the number of lanes per warp on the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numLanes - the returned number of lanes
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumDevices
     * \sa getNumSMs
     * \sa getNumWarps
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumLanes)(uint32_t dev, uint32_t *numLanes);
    /**
     * \fn CUDBGAPI_st::getNumRegisters
     * \brief Get the number of registers per lane on the device.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numRegs - the returned number of registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumDevices
     * \sa getNumSMs
     * \sa getNumWarps
     * \sa getNumLanes
     */
    CUDBGResult (*getNumRegisters)(uint32_t dev, uint32_t *numRegs);

    /* DWARF-related routines */
    /**
     * \fn CUDBGAPI_st::getPhysicalRegister30
     * \brief Get the physical register number(s) assigned to a virtual register name 'reg' at a given PC, if 'reg' is live at that PC.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.1.
     *
     * \ingroup DWARF
     *
     * \param pc - Program counter
     * \param reg - virtual register index
     * \param buf - physical register name(s)
     * \param sz - the physical register name buffer size
     * \param numPhysRegs - number of physical register names returned
     * \param regClass - the class of the physical registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNKNOWN
     */
    CUDBGResult (*getPhysicalRegister30)(uint64_t pc, char *reg, uint32_t *buf, uint32_t sz, uint32_t *numPhysRegs, CUDBGRegClass *regClass);
    /**
     * \fn CUDBGAPI_st::disassemble
     * \brief Disassemble instruction at instruction address.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param dev - device index
     * \param addr - instruction address
     * \param instSize - instruction size (32 or 64 bits)
     * \param buf - disassembled instruction buffer
     * \param sz - disassembled instruction buffer size
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNKNOWN
     */
    CUDBGResult (*disassemble)(uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::isDeviceCodeAddress55
     * \brief Determines whether a virtual address resides within device code. This API is strongly deprecated. Use CUDBGAPI_st::isDeviceCodeAddress instead.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 6.0
     *
     * \ingroup DWARF
     *
     * \param addr - virtual address
     * \param isDeviceAddress - true if address resides within device code
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*isDeviceCodeAddress55)(uintptr_t addr, bool *isDeviceAddress);
    /**
     * \fn CUDBGAPI_st::lookupDeviceCodeSymbol
     * \brief Determines whether a symbol represents a function in device code and returns its virtual address.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param symName - symbol name
     * \param symFound - set to true if the symbol is found
     * \param symAddr - the symbol virtual address if found
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*lookupDeviceCodeSymbol)(char *symName, bool *symFound, uintptr_t *symAddr);

    /* Events */
    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback31
     * \brief Provides the API with the function to call to notify the debugger of a new application or device event.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.2.
     *
     * \ingroup EVENT
     *
     * \param callback - the callback function
     * \param data - a pointer to be passed to the callback when called
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*setNotifyNewEventCallback31)(CUDBGNotifyNewEventCallback31 callback, void *data);

    /**
     * \fn CUDBGAPI_st::getNextEvent30
     * \brief Copies the next available event in the event queue into 'event' and removes it from the queue.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.1.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextEvent30)(CUDBGEvent30 *event);

    /**
     * \fn CUDBGAPI_st::acknowledgeEvent30
     * \brief Inform the debugger API that the event has been processed.
     *
     * Since CUDA 3.0.
     *
     * \deprecated in CUDA 3.1.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to the event that has been processed
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*acknowledgeEvent30)(CUDBGEvent30 *event);

    /* 3.1 Extensions */
    /**
     * \fn CUDBGAPI_st::getGridAttribute
     * \brief Get the value of a grid attribute
     *
     * Since CUDA 3.1.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param attr - the attribute
     * \param value - the returned value of the attribute
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_ATTRIBUTE,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getGridAttribute)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttribute attr, uint64_t *value);
    /**
     * \fn CUDBGAPI_st::getGridAttributes
     * \brief Get several grid attribute values in a single API call
     *
     * Since CUDA 3.1.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param pairs - array of attribute/value pairs
     * \param numPairs - the number of attribute/values pairs in the array
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_ATTRIBUTE,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getGridAttributes)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttributeValuePair *pairs, uint32_t numPairs);
    /**
     * \fn CUDBGAPI_st::getPhysicalRegister40
     *
     * \brief Get the physical register number(s) assigned to a virtual
     * register name 'reg' at a given PC, if 'reg' is live at that PC.
     *
     * Get the physical register number(s) assigned to a virtual register
     * name 'reg' at a given PC, if 'reg' is live at that PC. If a virtual
     * register name is mapped to more than one physical register, the
     * physical register with the lowest physical register index will
     * contain the highest bits of the virtual register, and the
     * physical register with the highest physical register index will
     * contain the lowest bits.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 4.1.
     *
     * \ingroup DWARF
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp indx
     * \param pc - Program counter
     * \param reg - virtual register index
     * \param buf - physical register name(s)
     * \param sz - the physical register name buffer size
     * \param numPhysRegs - number of physical register names returned
     * \param regClass - the class of the physical registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNKNOWN
     */
    CUDBGResult (*getPhysicalRegister40)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t pc, char *reg, uint32_t *buf, uint32_t sz, uint32_t *numPhysRegs, CUDBGRegClass *regClass);
    /**
     * \fn CUDBGAPI_st::readLaneException
     * \brief Reads the exception type for a given lane
     *
     * Since CUDA 3.1.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param exception - the returned exception type
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     */
    CUDBGResult (*readLaneException)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);

    /**
     * \fn CUDBGAPI_st::getNextEvent32
     * \brief Copies the next available event in the event queue into 'event' and removes it from the queue.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 4.0
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextEvent32)(CUDBGEvent32 *event);
    /**
     * \fn CUDBGAPI_st::acknowledgeEvents42
     * \brief Inform the debugger API that synchronous events have been processed.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 5.0.
     *
     * \ingroup EVENT
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*acknowledgeEvents42)(void);

    /* 3.1 - ABI */
    /**
     * \fn CUDBGAPI_st::readCallDepth32
     * \brief Reads the call depth (number of calls) for a given warp.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readReturnAddress32
     * \sa readVirtualReturnAddress32
     */
    CUDBGResult (*readCallDepth32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *depth);
    /**
     * \fn CUDBGAPI_st::readReturnAddress32
     * \brief Reads the physical return address for a call level.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param level - the specified call level
     * \param ra - the returned return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_ZERO_CALL_DEPTH,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCallDepth32
     * \sa readVirtualReturnAddress32
     */
    CUDBGResult (*readReturnAddress32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra);
    /**
     * \fn CUDBGAPI_st::readVirtualReturnAddress32
     * \brief Reads the virtual return address for a call level.
     *
     * Since CUDA 3.1.
     *
     * \deprecated in CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param level - the specified call level
     * \param ra - the returned virtual return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_ZERO_CALL_DEPTH,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL
     *
     * \sa readCallDepth32
     * \sa readReturnAddress32
     */
    CUDBGResult (*readVirtualReturnAddress32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra);

    /* 3.2 Extensions */
    /**
     * \fn CUDBGAPI_st::readGlobalMemory55
     * \brief Reads content at address in the global memory segment (entire 40-bit VA on Fermi+).
     *
     * Since CUDA 3.2.
     *
     * \deprecated in CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readGlobalMemory55)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeGlobalMemory55
     * \brief Writes content to address in the global memory segment (entire 40-bit VA on Fermi+).
     *
     * Since CUDA 3.2.
     *
     * \deprecated in CUDA 6.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     *
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeGlobalMemory55)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readPinnedMemory
     * \brief Reads content at pinned address in system memory.
     *
     * Since CUDA 3.2.
     *
     * \ingroup READ
     *
     * \param addr - system memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readPinnedMemory)(uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writePinnedMemory
     * \brief Writes content to pinned address in system memory.
     *
     * Since CUDA 3.2.
     *
     * \ingroup READ
     *
     * \param addr - system memory address
     * \param buf - buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*writePinnedMemory)(uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::setBreakpoint
     * \brief Sets a breakpoint at the given instruction address for the given device.
     *        Before setting a breakpoint, CUDBGAPI_st::getAdjustedCodeAddress should be
     *        called to get the adjusted breakpoint address.
     *
     * Since CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param dev - the device index
     * \param addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     * \sa unsetBreakpoint
     */
    CUDBGResult (*setBreakpoint)(uint32_t dev, uint64_t addr);
    /**
     * \fn CUDBGAPI_st::unsetBreakpoint
     * \brief Unsets a breakpoint at the given instruction address for the given device.
     *
     * Since CUDA 3.2.
     *
     * \ingroup BP
     *
     * \param dev - the device index
     * \param addr - instruction address
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     * \sa setBreakpoint
     */
    CUDBGResult (*unsetBreakpoint)(uint32_t dev, uint64_t addr);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback40
     * \brief Provides the API with the function to call to notify the debugger of
     * a new application or device event.
     *
     * Since CUDA 3.2.
     *
     * \deprecated in CUDA 4.1.
     *
     * \ingroup EVENT
     *
     * \param callback - the callback function
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*setNotifyNewEventCallback40)(CUDBGNotifyNewEventCallback40 callback);

    /* 4.0 Extensions */
    /**
     * \fn CUDBGAPI_st::getNextEvent42
     * \brief Copies the next available event in the event queue into 'event' and removes it from the queue.
     *
     * Since CUDA 4.0.
     *
     * \deprecated in CUDA 5.0
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextEvent42)(CUDBGEvent42 *event);
    /**
     * \fn CUDBGAPI_st::readTextureMemory
     * \brief This method is no longer supported since CUDA 12.0
     *
     * \ingroup READ
     *
     * \param devId - device index
     * \param vsm - SM index
     * \param wp - warp index
     * \param id - texture id (the value of DW_AT_location attribute in the relocated ELF image)
     * \param dim - texture dimension (1 to 4)
     * \param coords - array of coordinates of size dim
     * \param buf - result buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readTextureMemory)(uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t id, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::readBlockIdx
     * \brief Reads the CUDA block index running on a valid warp.
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param blockIdx - the returned CUDA block index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readBlockIdx)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
    /**
     * \fn CUDBGAPI_st::getGridDim
     * \brief Get the number of blocks in the given grid.
     *
     * Since CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param gridDim - the returned number of blocks in the grid
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getBlockDim
     */
    CUDBGResult (*getGridDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *gridDim);
    /**
     * \fn CUDBGAPI_st::readCallDepth
     * \brief Reads the call depth (number of calls) for a given lane
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readReturnAddress
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);
    /**
     * \fn CUDBGAPI_st::readReturnAddress
     * \brief Reads the physical return address for a call level
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param level - the specified call level
     * \param ra - the returned return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_ZERO_CALL_DEPTH,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCallDepth
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra);
    /**
     * \fn CUDBGAPI_st::readVirtualReturnAddress
     * \brief Reads the virtual return address for a call level
     *
     * Since CUDA 4.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param level - the specified call level
     * \param ra - the returned virtual return address for level
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INVALID_CALL_LEVEL,
     * \return CUDBG_ERROR_ZERO_CALL_DEPTH,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL
     *
     * \sa readCallDepth
     * \sa readReturnAddress
     */
    CUDBGResult (*readVirtualReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra);
    /**
     * \fn CUDBGAPI_st::getElfImage
     * \brief Get the relocated or non-relocated ELF image and size for the grid on the given device
     *
     * Since CUDA 4.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param relocated - set to true to specify the relocated ELF image, false otherwise
     * \param *elfImage - pointer to the ELF image
     * \param size - size of the ELF image (64 bits)
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getElfImage)(uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint64_t *size);

    /* 4.1 Extensions */
    /**
    * \fn CUDBGAPI_st::getHostAddrFromDeviceAddr
    * \brief given a device virtual address, return a corresponding system memory virtual address.
     *
     * Since CUDA 4.1.
     *
    * \ingroup DWARF
    *
    * \param dev - device index
    * \param device_addr - device memory address
    * \param host_addr - returned system memory address
    *
    * \return CUDBG_SUCCESS,
    * \return CUDBG_ERROR_INVALID_ARGS,
    * \return CUDBG_ERROR_INVALID_DEVICE,
    * \return CUDBG_ERROR_INVALID_CONTEXT,
    * \return CUDBG_ERROR_INVALID_MEMORY_SEGMENT
    *
    * \sa readGenericMemory
    * \sa writeGenericMemory
    * \sa
    */
    CUDBGResult (*getHostAddrFromDeviceAddr)(uint32_t dev, uint64_t device_addr, uint64_t *host_addr);

    /**
     * \fn CUDBGAPI_st::singleStepWarp41
     * \brief Single step an individual warp on a suspended CUDA device.
     *
     * Since CUDA 4.1.
     *
     * \deprecated in CUDA 7.5.
     *
     * \ingroup EXEC
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param warpMask - the warps that have been single-stepped
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_UNKNOWN
     *
     * \sa resumeDevice
     * \sa suspendDevice
     */
    CUDBGResult (*singleStepWarp41)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *warpMask);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback41
     * \brief Provides the API with the function to call to notify the debugger of a new application or device event.
     *
     * Since CUDA 4.1.
     *
     * \ingroup EVENT
     *
     * \param callback - the callback function
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*setNotifyNewEventCallback41)(CUDBGNotifyNewEventCallback41 callback);
    /**
     * \fn CUDBGAPI_st::readSyscallCallDepth
     * \brief Reads the call depth of syscalls for a given lane
     *
     * Since CUDA 4.1.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param depth - the returned call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readReturnAddress
     * \sa readVirtualReturnAddress
     */
    CUDBGResult (*readSyscallCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);

    /* 4.2 Extensions */
    /**
     * \fn CUDBGAPI_st::readTextureMemoryBindless
     * \brief This method is no longer supported since CUDA 12.0
     *
     * \ingroup READ
     *
     * \param devId - device index
     * \param vsm - SM index
     * \param wp - warp index
     * \param texSymtabIndex - global symbol table index of the texture symbol
     * \param dim - texture dimension (1 to 4)
     * \param coords - array of coordinates of size dim
     * \param buf - result buffer
     * \param sz - size of the buffer
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readTextureMemoryBindless)(uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t texSymtabIndex, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);

    /* 5.0 Extensions */
    /**
     * \fn CUDBGAPI_st::clearAttachState
     * \brief Clear attach-specific state prior to detach.
     *
     * Since CUDA 5.0.
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*clearAttachState)(void);
    /**
     * \fn CUDBGAPI_st::getNextSyncEvent50
     *
     * Since CUDA 5.0.
     *
     * \deprecated in CUDA 5.5.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextSyncEvent50)(CUDBGEvent50 *event);
    /**
     * \fn CUDBGAPI_st::memcheckReadErrorAddress
     * \brief Get the address that memcheck detected an error on.
     *
     * Since CUDA 5.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param address - returned address detected by memcheck
     * \param storage - returned address class of address
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     */
    CUDBGResult (*memcheckReadErrorAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *address, ptxStorageKind *storage);
    /**
     * \fn CUDBGAPI_st::acknowledgeSyncEvents
     * \brief Inform the debugger API that synchronous events have been processed.
     *
     * Since CUDA 5.0.
     *
     * \ingroup EVENT
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*acknowledgeSyncEvents)(void);
    /**
     * \fn CUDBGAPI_st::getNextAsyncEvent50
     * \brief Copies the next available event in the asynchronous event queue into 'event' and removes it from the queue.  The asynchronous event queue is held separate from the normal event queue, and does not require acknowledgement from the debug client.
     *
     * Since CUDA 5.0.
     *
     * \deprecated in CUDA 5.5.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextAsyncEvent50)(CUDBGEvent50 *event);
    /**
     * \fn CUDBGAPI_st::requestCleanupOnDetach55
     * \brief Request for cleanup of driver state when detaching.
     *
     * Since CUDA 5.0.
     *
     * \deprecated in CUDA 6.0
     *
     * \return CUDBG_SUCCESS
     * \return CUDBG_ERROR_COMMUNICATION_FAILURE
     * \return CUDBG_ERROR_INVALID_ARGS
     * \return CUDBG_ERROR_INTERNAL
     */
    CUDBGResult (*requestCleanupOnDetach55)(void);

    /**
     * \fn CUDBGAPI_st::initializeAttachStub
     * \brief Initialize the attach stub.
     *
     * Since CUDA 5.0.
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*initializeAttachStub)(void);
    /**
     * \fn CUDBGAPI_st::getGridStatus50
     * \brief Check whether the grid corresponding to the given
     *        gridId is still present on the device.
     *
     * Since CUDA 5.0.
     *
     * \deprecated in CUDA 5.5.
     *
     * \ingroup GRID
     *
     * \param devId  - device index
     * \param gridId - grid ID
     * \param status - enum indicating whether the grid status is INVALID, PENDING, ACTIVE,
     *                 SLEEPING, TERMINATED or UNDETERMINED
     *
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INTERNAL
     */
    CUDBGResult (*getGridStatus50)(uint32_t dev, uint32_t gridId, CUDBGGridStatus *status);

    /* 5.5 Extensions */
    /**
     * \fn CUDBGAPI_st::getNextSyncEvent55
     * \brief Copies the next available event in the synchronous event queue into 'event' and removes it from the queue.
     *
     * Since CUDA 5.5.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextSyncEvent55)(CUDBGEvent55 *event);
    /**
     * \fn CUDBGAPI_st::getNextAsyncEvent55
     * \brief Copies the next available event in the asynchronous event queue into 'event' and removes it from the queue.  The asynchronous event queue is held separate from the normal event queue, and does not require acknowledgement from the debug client.
     *
     * Since CUDA 5.5.
     *
     * \ingroup EVENT
     *
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextAsyncEvent55)(CUDBGEvent55 *event);
    /**
     * \fn CUDBGAPI_st::getGridInfo55
     * \brief Get information about the specified grid.
     *        If the context of the grid has already been destroyed,
     *        the function will return CUDBG_ERROR_INVALID_GRID,
     *        although the grid id is correct.
     *
     * Since CUDA 5.5.
     *
     * \ingroup GRID
     *
     * \param devId    - device index
     * \param gridId   - grid ID for which information is to be collected
     * \param gridInfo - pointer to a client allocated structure in which grid
     *                   info will be returned.
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_SUCCESS
     *
     */
    CUDBGResult (*getGridInfo55)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo55 *gridInfo);
    /**
     * \fn CUDBGAPI_st::readGridId
     * \brief Reads the 64-bit CUDA grid index running on a valid warp.
     *
     * Since CUDA 5.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param gridId - the returned 64-bit CUDA grid index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readBlockIdx
     * \sa readThreadIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readGridId)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId64);
    /**
     * \fn CUDBGAPI_st::getGridStatus
     * \brief Check whether the grid corresponding to the given
     *        gridId is still present on the device.
     *
     * Since CUDA 5.5.
     *
     * \ingroup GRID
     *
     * \param devId    - device index
     * \param gridId64 - 64-bit grid ID
     * \param status   - enum indicating whether the grid status is INVALID, PENDING, ACTIVE,
     *                   SLEEPING, TERMINATED or UNDETERMINED
     *
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_INTERNAL
     */
    CUDBGResult (*getGridStatus)(uint32_t dev, uint64_t gridId64, CUDBGGridStatus *status);
    /**
     * \fn CUDBGAPI_st::setKernelLaunchNotificationMode
     * \brief Set the launch notification policy.
     *
     * Since CUDA 5.5.
     *
     * \param mode - mode to deliver kernel launch notifications in
     *
     * \return CUDBG_SUCCESS
     *
     */
    CUDBGResult (*setKernelLaunchNotificationMode)(CUDBGKernelLaunchNotifyMode mode);
    /**
     * \fn CUDBGAPI_st::getDevicePCIBusInfo
     * \brief Get PCI bus and device ids associated with device devId.
     *
     * \param devId    - the cuda device id
     * \param pciBusId - pointer where corresponding PCI BUS ID would be stored
     * \param pciDevId - pointer where corresponding PCI DEVICE ID would be stored
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     */
    CUDBGResult (*getDevicePCIBusInfo)(uint32_t devId, uint32_t *pciBusId, uint32_t *pciDevId);
    /**
     * \fn CUDBGAPI_st::readDeviceExceptionState80
     * \brief Get the exception state of the SMs on the device
     *
     * Since CUDA 5.5
     *
     * \param devId           - the cuda device id
     * \param exceptionSMMask - Bit field containing a 1 at (1 << i) if SM i hit an exception
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     */
    CUDBGResult (*readDeviceExceptionState80)(uint32_t devId, uint64_t *exceptionSMMask);

    /* 6.0 Extensions */
    /**
     * \fn CUDBGAPI_st::getAdjustedCodeAddress
     * \brief The client must call this function before inserting a breakpoint, or when the previous
     *        or next code address is needed.
     *        Returns the adjusted code address for a given code address for a given device.
     *
     * Since CUDA 5.5.
     *
     * \ingroup BP
     *
     * \param devId - the device index
     * \param addr - instruction address
     * \param adjustedAddress - adjusted address
     * \param adjAction - whether the adjusted next, previous or current address is needed
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     * \sa unsetBreakpoint
     */
    CUDBGResult (*getAdjustedCodeAddress)(uint32_t devId, uint64_t address, uint64_t *adjustedAddress, CUDBGAdjAddrAction adjAction);
    /**
     * \fn CUDBGAPI_st::readErrorPC
     * \brief Get the hardware reported error PC if it exists
     *
     * Since CUDA 6.0
     *
     * \ingroup READ
     *
     * \param devId - the device index
     * \param sm - the SM index
     * \param warp - the warp index
     * \param errorPC - PC ofthe exception
     * \param errorPCValid - boolean to indicate that the returned error PC is valid
     *
     * \return CUDBG_SUCCESS
     * \return CUDBG_ERROR_UNINITIALIZED
     * \return CUDBG_ERROR_INVALID_DEVICE
     * \return CUDBG_ERROR_INVALID_SM
     * \return CUDBG_ERROR_INVALID_WARP
     * \return CUDBG_ERROR_INVALID_ARGS
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION
     */
    CUDBGResult (*readErrorPC)(uint32_t devId, uint32_t sm, uint32_t wp, uint64_t *errorPC, bool *errorPCValid);
    /**
     * \fn CUDBGAPI_st::getNextEvent
     * \brief Copies the next available event into 'event' and removes it from the queue.
     *
     * Since CUDA 6.0.
     *
     * \ingroup EVENT
     *
     * \param type  - application event queue type
     * \param event - pointer to an event container where to copy the event parameters
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_INVALID_ARGS
     */
    CUDBGResult (*getNextEvent)(CUDBGEventQueueType type, CUDBGEvent  *event);
    /**
     * \fn CUDBGAPI_st::getElfImageByHandle
     * \brief Get the relocated or non-relocated ELF image for the given handle on the given device
     *
     * The handle is provided in the ELF Image Loaded notification event.
     *
     * Since CUDA 6.0.
     *
     * \ingroup DWARF
     *
     * \param devId - device index
     * \param handle - elf image handle
     * \param type - type of the requested ELF image
     * \param elfImage - pointer to the ELF image
     * \param elfImage_size - size of the ELF image
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     */
    CUDBGResult (*getElfImageByHandle)(uint32_t devId, uint64_t handle, CUDBGElfImageType type, void *elfImage, uint64_t size);

    /**
     * \fn CUDBGAPI_st::resumeWarpsUntilPC
     * \brief Inserts a temporary breakpoint at the specified virtual PC,
     *        and resumes all warps in the specified bitmask on a given SM.
     *        As compared to CUDBGAPI_st::resumeDevice, CUDBGAPI_st::resumeWarpsUntilPC
     *        provides finer-grain control by resuming a selected set of warps on the
     *        same SM.  The main intended usage is to accelerate the single-stepping
     *        process when the target PC is known in advance.  Instead of single-stepping
     *        each warp individually until the target PC is hit, the client can issue this
     *        API.  When this API is used, errors within CUDA kernels will no longer be
     *        reported precisely.  In the situation where resuming warps is not possible,
     *        this API will return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE.  The client
     *        should then fall back to using CUDBGAPI_st::singleStepWarp or
     *        CUDBGAPI_st::resumeDevice.
     *
     * Since CUDA 6.0.
     *
     * \ingroup EXEC
     *
     * \param devId - device index
     * \param sm - the SM index
     * \param warpMask - the bitmask of warps to resume (1 = resume, 0 = do not resume)
     * \param virtPC - the virtual PC where the temporary breakpoint will be inserted
     *
     * \return CUDBG_SUCCESS
     * \return CUDBG_ERROR_INVALID_ARGS
     * \return CUDBG_ERROR_INVALID_DEVICE
     * \return CUDBG_ERROR_INVALID_SM
     * \return CUDBG_ERROR_INVALID_WARP_MASK
     * \return CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa resumeDevice
     */
    CUDBGResult (*resumeWarpsUntilPC)(uint32_t devId, uint32_t sm, uint64_t warpMask, uint64_t virtPC);
    /**
     * \fn CUDBGAPI_st::readWarpState60
     * \brief Get state of a given warp
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param state - pointer to structure that contains warp status
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     */
    CUDBGResult (*readWarpState60)(uint32_t devId, uint32_t sm, uint32_t wp, CUDBGWarpState60 *state);
    /**
     * \fn CUDBGAPI_st::readRegisterRange
     * \brief Reads content of a hardware range of hardware registers
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param index - index of the first register to read
     * \param registers_size - number of registers to read
     * \param registers - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readPC
     * \sa readRegister
     */
    CUDBGResult (*readRegisterRange)(uint32_t devId, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t index, uint32_t registers_size, uint32_t *registers);
    /**
     * \fn CUDBGAPI_st::readGenericMemory
     * \brief Reads content at an address in the generic address space.
     *        This function determines if the given address falls into the local, shared, or global memory window.
     *        It then accesses memory taking into account the hardware co-ordinates provided as inputs.
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param buf - buffer
     * \param buf_size - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readGenericMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeGenericMemory
     * \brief Writes content to an address in the generic address space.
     *        This function determines if the given address falls into the local, shared, or global memory window.
     *        It then accesses memory taking into account the hardware co-ordinates provided as inputs.
     *
     * Since CUDA 6.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param[in] buf - buffer
     * \param buf_size - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     *
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeGenericMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
   /**
     * \fn CUDBGAPI_st::readGlobalMemory
     * \brief Reads content at an address in the global address space.
     *        If the address is valid on more than one device and one
     *        of those devices does not support UVA, an error is returned.
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param addr - memory address
     * \param[in] buf - buffer
     * \param buf_size - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS_
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readGlobalMemory)(uint64_t addr, void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::writeGlobalMemory
     * \brief Writes content to an address in the global address space.
     *        If the address is valid on more than one device and one
     *        of those devices does not support UVA, an error is returned.
     *
     * Since CUDA 6.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addr - memory address
     * \param[in] buf - buffer
     * \param buf_size - size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_MEMORY_MAPPING_FAILED,
     * \return CUDBG_ERROR_INVALID_MEMORY_ACCESS,
     * \return CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM
     * \return CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS_
     *
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writeGlobalMemory)(uint64_t addr, const void *buf, uint32_t sz);
    /**
     * \fn CUDBGAPI_st::getManagedMemoryRegionInfo
     * \brief Returns a sorted list of managed memory regions
     *        The sorted list of memory regions starts from a region containing the
     *        specified starting address. If the starting address is set to 0, a
     *        sorted list of managed memory regions is returned which starts from
     *        the managed memory region with the lowest start address.
     *
     * Since CUDA 6.0.
     *
     * \ingroup READ
     *
     * \param startAddress - The address that the first region in the list must contain.
     *                       If the starting address is set to 0, the list of managed memory regions
     *                       returned starts from the managed memory region with the lowest start address.
     * \param memoryInfo - Client-allocated array of memory region records of type CUDBGMemoryInfo.
     * \param memoryInfo_size - Number of records of type CUDBGMemoryInfo that memoryInfo can hold.
     * \param numEntries - Pointer to a client-allocated variable holding the number of valid
     *                     entries retured in memoryInfo. Valid entries are continguous and start
     *                     from memoryInfo[0].
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_ADDRESS,
     * \return CUDBG_ERROR_INTERNAL
     */
    CUDBGResult (*getManagedMemoryRegionInfo)(uint64_t startAddress, CUDBGMemoryInfo *memoryInfo, uint32_t memoryInfo_size, uint32_t *numEntries);
    /**
     * \fn CUDBGAPI_st::isDeviceCodeAddress
     * \brief Determines whether a virtual address resides within device code.
     *
     * Since CUDA 3.0.
     *
     * \ingroup DWARF
     *
     * \param addr - virtual address
     * \param isDeviceAddress - true if address resides within device code
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*isDeviceCodeAddress)(uintptr_t addr, bool *isDeviceAddress);
    /**
     * \fn CUDBGAPI_st::requestCleanupOnDetach
     * \brief Request for cleanup of driver state when detaching.
     *
     * Since CUDA 6.0.
     *
     * \param appResumeFlag - value of CUDBG_RESUME_FOR_ATTACH_DETACH as read
     *                        from the application's process space.
     *
     * \return CUDBG_SUCCESS
     * \return CUDBG_ERROR_COMMUNICATION_FAILURE
     * \return CUDBG_ERROR_INVALID_ARGS
     * \return CUDBG_ERROR_INTERNAL
     */
    CUDBGResult (*requestCleanupOnDetach)(uint32_t appResumeFlag);

   /* 6.5 Extensions */
    /**
     * \fn CUDBGAPI_st::readPredicates
     * \brief Reads content of hardware predicate registers.
     *
     * Since CUDA 6.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param predicates_size - number of predicate registers to read
     * \param predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readGlobalMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     */
    CUDBGResult (*readPredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
    /**
     * \fn CUDBGAPI_st::writePredicates
     * \brief Writes content to hardware predicate registers.
     *
     * Since CUDA 6.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param predicates_size - number of predicate registers to write
     * \param[in] predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa writeConstMemory
     * \sa writeGenericMemory
     * \sa writeGlobalMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeTextureMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     */
    CUDBGResult (*writePredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
    /**
     * \fn CUDBGAPI_st::getNumPredicates
     * \brief Get the number of predicate registers per lane on the device.
     *
     * Since CUDA 6.5.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numPredicates - the returned number of predicate registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumDevices
     * \sa getNumSMs
     * \sa getNumWarps
     * \sa getNumLanes
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumPredicates)(uint32_t dev, uint32_t *numPredicates);
    /**
     * \fn CUDBGAPI_st::readCCRegister
     * \brief Reads the hardware CC register.
     *
     * Since CUDA 6.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param val - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readCodeMemory
     * \sa readConstMemory129
     * \sa readGenericMemory
     * \sa readGlobalMemory
     * \sa readParamMemory
     * \sa readSharedMemory
     * \sa readTextureMemory
     * \sa readLocalMemory
     * \sa readRegister
     * \sa readPC
     * \sa readPredicates
     */
    CUDBGResult (*readCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
    /**
     * \fn CUDBGAPI_st::writeCCRegister
     * \brief Writes the hardware CC register.
     *
     * Since CUDA 6.5.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param val - value to write to the CC register
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa writeConstMemory
     * \sa writeGenericMemory
     * \sa writeGlobalMemory
     * \sa writeParamMemory
     * \sa writeSharedMemory
     * \sa writeTextureMemory
     * \sa writeLocalMemory
     * \sa writeRegister
     * \sa writePredicates
     */
    CUDBGResult (*writeCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);

    /**
     * \fn CUDBGAPI_st::getDeviceName
     * \brief Get the device name string.
     *
     * Since CUDA 6.5.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param buf - the destination buffer
     * \param sz - the size of the buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getSMType
     * \sa getDeviceType
     */
    CUDBGResult (*getDeviceName)(uint32_t dev, char *buf, uint32_t sz);
    CUDBGResult (*singleStepWarp65)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, uint64_t *warpMask);

    /* 9.0 Extensions */
    /**
     * \fn CUDBGAPI_st::readDeviceExceptionState
     * \brief Get the exception state of the SMs on the device
     * 
     * Since CUDA 9.0
     *
     * \param devId - the cuda device id
     * \param mask  - Arbitrarily sized bit field containing a 1 at (1 << i) if SM i hit an exception
     * \param sz    - Size (in bytes) of \p mask (must be large enough to hold a bit for each sm on the device)
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE
     *
     * \sa getNumSMs
     *
     */
    CUDBGResult (*readDeviceExceptionState)(uint32_t devId, uint64_t *mask, uint32_t numWords);

    /* 10.0 Extensions */
    /**
     * \fn CUDBGAPI_st::getNumUniformRegisters
     * \brief Get the number of uniform registers per warp on the device.
     *
     * Since CUDA 10.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numRegs - the returned number of uniform registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumRegisters
     */
    CUDBGResult (*getNumUniformRegisters)(uint32_t dev, uint32_t *numRegs);
    /**
     * \fn CUDBGAPI_st::readUniformRegisterRange
     * \brief Reads a range of uniform registers.
     *
     * Since CUDA 10.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param regno - starting index into uniform register file
     * \param registers_size - number of bytes to read
     * \param registers - pointer to buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readRegister
     */
    CUDBGResult (*readUniformRegisterRange)(uint32_t devId, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t registers_size, uint32_t *registers);
    /**
     * \fn CUDBGAPI_st::writeUniformRegister
     * \brief Writes content to a uniform register.
     *
     * Since CUDA 10.0.
     *
     * \ingroup WRITE
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param regno - register index
     * \param val - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa writeRegister
     * \sa readUniformRegisterRange
     */
    CUDBGResult (*writeUniformRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val);
    /**
     * \fn CUDBGAPI_st::getNumUniformPredicates
     * \brief Get the number of uniform predicate registers per warp on the device.
     *
     * Since CUDA 10.0.
     *
     * \ingroup DEV
     *
     * \param dev - device index
     * \param numPredicates - the returned number of uniform predicate registers
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getNumUniformPredicates
     */
    CUDBGResult (*getNumUniformPredicates)(uint32_t dev, uint32_t *numPredicates);
    /**
     * \fn CUDBGAPI_st::readUniformPredicates
     * \brief Reads contents of uniform predicate registers.
     *
     * Since CUDA 10.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param predicates_size - number of predicate registers to read
     * \param[in] predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readPredicates
     */
    CUDBGResult (*readUniformPredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, uint32_t *predicates);
    /**
     * \fn CUDBGAPI_st::writeUniformPredicates
     * \brief Writes to uniform predicate registers.
     *
     * Since CUDA 10.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param predicates_size - number of predicate registers to write
     * \param[in] predicates - buffer
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readUniformPredicate
     * \sa writeRegister
     */
    CUDBGResult (*writeUniformPredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates);

    /* 11.8 Extensions */
    /**
     * \fn CUDBGAPI_st::getLoadedFunctionInfo118
     * \brief Get the section number and address of loaded functions for a given module.
     * 
     * Since CUDA 11.8
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param handle - ELF/cubin image handle
     * \param functions - information about loaded functions
     * \param numEntries - number of function load entries to read
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getLoadedFunctionInfo118)(uint32_t devId, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries);

    /* 12.0 Extensions */
    /**
     * \fn CUDBGAPI_st::getGridInfo120
     * \brief Get information about the specified grid.
     *        If the context of the grid has already been destroyed,
     *        the function will return CUDBG_ERROR_INVALID_GRID,
     *        although the grid id is correct.
     *
     * Since CUDA 12.0.
     *
     * \ingroup GRID
     *
     * \param devId    - device index
     * \param gridId   - grid ID for which information is to be collected
     * \param gridInfo - pointer to a client allocated structure in which grid
     *                   info will be returned.
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_SUCCESS
     *
     */
    CUDBGResult (*getGridInfo120)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo120 *gridInfo);
    /**
     * \fn CUDBGAPI_st::getClusterDim120
     * \brief Get the number of blocks in the given cluster.
     *
     * Since CUDA 12.0.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param gridId64 - grid ID
     * \param clusterDim - the returned number of blocks in the cluster
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getBlockDim
     * \sa getGridDim
     */
    CUDBGResult (*getClusterDim120)(uint32_t dev, uint64_t gridId64, CuDim3 *clusterDim);
    /**
     * \fn CUDBGAPI_st::readWarpState120
     * \brief Get state of a given warp
     *
     * Since CUDA 12.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param state - pointer to structure that contains warp status
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     */
    CUDBGResult (*readWarpState120)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState120 *state);
    /**
     * \fn CUDBGAPI_st::readClusterIdx
     * \brief Reads the CUDA cluster index running on a valid warp.
     *
     * Since CUDA 12.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param clusterIdx - the returned CUDA cluster index
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa readGridId
     * \sa readThreadIdx
     * \sa readBlockIdx
     * \sa readBrokenWarps
     * \sa readValidWarps
     * \sa readValidLanes
     * \sa readActiveLanes
     */
    CUDBGResult (*readClusterIdx)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx);

    /* 12.2 Extensions */
    /**
     * \fn CUDBGAPI_st::getErrorStringEx
     * \brief Fills a user-provided buffer with an error message encoded as a null-terminated ASCII string.
     *        The error message is specific to the last failed API call and is invalidated after every API call.
     *
     * Since CUDA 12.2.
     *
     * \ingroup EVENT
     *
     * \param buf - the destination buffer
     * \param bufSz - the size of the destination buffer in bytes
     * \param msgSz - the size of an error message including the terminating null character.
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getErrorString
     */
    CUDBGResult (*getErrorStringEx)(char *buf, uint32_t bufSz, uint32_t *msgSz);

    /* 12.3 Extensions */
    /**
     * \fn CUDBGAPI_st::getLoadedFunctionInfo
     * \brief Get the section number and address of loaded functions for a given module.
     * 
     * Since CUDA 11.8
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param handle - ELF/cubin image handle
     * \param functions - information about loaded functions
     * \param numEntries - number of function load entries to read
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getLoadedFunctionInfo)(uint32_t devId, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t startIndex, uint32_t numEntries);
    /**
     * \fn CUDBGAPI_st::generateCoredump
     * \brief Generates a coredump for the current GPU state
     * 
     * Since CUDA 12.3
     *
     * \ingroup READ
     *
     * \param filename - target coredump file name
     * \param flags - coredump generation flags/options
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*generateCoredump)(const char* filename, CUDBGCoredumpGenerationFlags flags);
    /**
     * \fn CUDBGAPI_st::getConstBankAddress123
     * \brief Convert constant bank number and offset into GPU VA.
     *
     * Since CUDA 12.3
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param bank - constant bank number
     * \param offset - offset within the bank
     * \param address - (output) GPU VA
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getConstBankAddress123)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t bank, uint32_t offset, uint64_t* address);

    /* 12.4 Extensions */
    /**
     * \fn CUDBGAPI_st::gpudbgGetDeviceInfoSizes
     * \brief Returns sizes for device info structs and defined attributes.
     *
     * Since CUDA 12.4
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sizes - (output) device info sizes
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getDeviceInfoSizes)(uint32_t dev, CUDBGDeviceInfoSizes* sizes);
    /**
     * \fn CUDBGAPI_st::gpudbgGetDeviceInfo
     * \brief Returns full device info for the device.
     *
     * Since CUDA 12.4
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param type - query type (full or delta)
     * \param buffer - output buffer
     * \param length - output buffer length
     * \param dataLength - (output) number of bytes written to the buffer
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getDeviceInfo)(uint32_t dev, CUDBGDeviceInfoQueryType_t type, void *buffer, uint32_t length, uint32_t *dataLength);
    /**
     * \fn CUDBGAPI_st::getConstBankAddress
     * \brief Get constant bank GPU VA and size.
     *
     * Since CUDA 12.4
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param gridId64 - grid ID of the grid containing the constant bank
     * \param bank - constant bank number
     * \param address - (output) GPU VA of the bank memory
     * \param size - (output) bank size
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getConstBankAddress)(uint32_t dev, uint64_t gridId64, uint32_t bank, uint64_t* address, uint32_t* size);

    /**
     * \fn CUDBGAPI_st::singleStepWarp
     * \brief Single step an individual warp nsteps times on a suspended CUDA device.
     *        Only the last instruction in a range should be a control flow instruction.
     *
     * Since CUDA 7.5.
     *
     * \ingroup EXEC
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param nsteps   - number of single steps
     * \param warpMask - the warps that have been single-stepped
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_RUNNING_DEVICE,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_UNKNOWN
     *
     * \sa resumeDevice
     * \sa suspendDevice
     */
    CUDBGResult (*singleStepWarp)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t laneHint, uint32_t nsteps, uint32_t flags, uint64_t *warpMask);

    /* 12.5 Extensions */
    /**
     * \fn CUDBGAPI_st::readAllVirtualReturnAddresses
     * \brief Reads all the virtual return addresses
     *
     * Since CUDA 12.5.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param addrs - the returned addresses array
     * \param numAddrs - number of elements in addrs array
     * \param callDepth - the returned call depth
     * \param syscallCallDepth - the returned syscall call depth
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL
     *
     * \sa readCallDepth
     * \sa readReturnAddress
     */
    CUDBGResult (*readAllVirtualReturnAddresses)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *addrs, uint32_t numAddrs, uint32_t* callDepth, uint32_t* syscallCallDepth);
    /**
     * \fn CUDBGAPI_st::getSupportedDebuggerCapabilities
     * \brief Returns debugger capabilities that are supported by this version of the API
     *
     * Since CUDA 12.5.
     *
     * This API method can be called without initializing the API.
     *
     * \ingroup INIT
     *
     * \param capabilities - returned debugger capabilities
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     */
    CUDBGResult (*getSupportedDebuggerCapabilities)(CUDBGCapabilityFlags* capabilities);
    /**
     * \fn CUDBGAPI_st::readSmException
     * \brief Get the SM exception status if it exists
     *
     * Since CUDA 12.5.
     *
     * \ingroup READ
     *
     * \param dev - the device index
     * \param sm - the SM index
     * \param exception - returned exception
     * \param errorPC - returned PC of the exception
     * \param errorPCValid - boolean to indicate that the returned error PC is valid
     *
     * \return CUDBG_SUCCESS
     * \return CUDBG_ERROR_UNINITIALIZED
     * \return CUDBG_ERROR_INVALID_DEVICE
     * \return CUDBG_ERROR_INVALID_SM
     * \return CUDBG_ERROR_INVALID_ARGS
     * \return CUDBG_ERROR_UNKNOWN_FUNCTION
     */
    CUDBGResult (*readSmException)(uint32_t dev, uint32_t sm, CUDBGException_t *exception, uint64_t *errorPC, bool *errorPCValid);

    /* 12.6 Extensions */
    /**
     * \fn CUDBGAPI_st:executeInternalCommand
     * \brief Execute an internal command (not available in public driver builds)
     * 
     * Since CUDA 12.6.
     * 
     * \ingroup EXEC
     * 
     * \param command - the command name and arguments
     * \param resultBuffer - the destination buffer
     * \param sizeInBytes - size of the buffer
     * 
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*executeInternalCommand)(const char* command, char* resultBuffer, uint32_t sizeInBytes);

    /* 12.7 Extensions */
    /**
     * \fn CUDBGAPI_st::getGridInfo
     * \brief Get information about the specified grid.
     *        If the context of the grid has already been destroyed,
     *        the function will return CUDBG_ERROR_INVALID_GRID,
     *        although the grid id is correct.
     *
     * Since CUDA 12.7.
     *
     * \ingroup GRID
     *
     * \param devId    - device index
     * \param gridId   - grid ID for which information is to be collected
     * \param gridInfo - pointer to a client allocated structure in which grid
     *                   info will be returned.
     *
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_SUCCESS
     *
     */
    CUDBGResult (*getGridInfo)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo *gridInfo);
    /**
     * \fn CUDBGAPI_st::getClusterDim
     * \brief Get the number of blocks in the given cluster.
     *
     * Since CUDA 12.7.
     *
     * \ingroup GRID
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param clusterDim - the returned number of blocks in the cluster
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED
     *
     * \sa getBlockDim
     * \sa getGridDim
     */
    CUDBGResult (*getClusterDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterDim);
    /**
     * \fn CUDBGAPI_st::readWarpState127
     * \brief Get state of a given warp
     *
     * Since CUDA 12.7.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param state - pointer to structure that contains warp status
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     */
    CUDBGResult (*readWarpState127)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState127 *state);
    /**
     * \fn CUDBGAPI_st::getClusterExceptionTargetBlock
     * \brief Retrieves the target block index and validity status for a given
     * device, streaming multiprocessor, and warp for cluster exceptions.
     *
     * Since CUDA 12.7.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param blockIdx - pointer to a `CuDim3` structure that will be populated with the target block index
     * \param blockIdxValid - pointer to a boolean variable that will be set to `true` if the target block index is valid, and `false` otherwise. Value will be set to false if the warp is not stopped on a cluster exception
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_ARGS,
     */
    CUDBGResult (*getClusterExceptionTargetBlock)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx, bool *blockIdxValid);

    /* 12.8 Extensions */
    /**
     * \fn CUDBGAPI_st::readWarpResources
     * \brief Get the resources assigned to a given warp
     *
     * Since CUDA 12.8.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param resources - pointer to structure that contains warp resources
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     */
    CUDBGResult (*readWarpResources)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpResources *resources);

    /* 12.9 Extensions */
    /**
     * \fn CUDBGAPI_st::getCbuWarpState
     * \brief Gets CBU state of a given warp
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param warpMask   - bitmask of the warps which states should be returned in warpStates
     * \param warpStates - pointer to the array of CUDBGCbuWarpState structures
     * \param numWarpStates - number of elements in warpStates array
     *
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     */
    CUDBGResult (*getCbuWarpState)(uint32_t dev, uint32_t sm, uint64_t warpMask, CUDBGCbuWarpState* warpStates, uint32_t numWarpStates);
    /**
     * \fn CUDBGAPI_st::readWarpState
     * \brief Get state of a given warp
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm  - SM index
     * \param wp  - warp index
     * \param state - pointer to structure that contains warp state
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_UNINITIALIZED,
     */
    CUDBGResult (*readWarpState)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state);
    /**
     * \fn CUDBGAPI_st::consumeCudaLogs
     * \brief Get CUDA error log entries.
     *   This consumes the log entries, so they will not be available in subsequent calls.
     *
     * Since CUDA 12.9.
     *
     * \ingroup READ
     *
     * \param logMessages - client-allocated array to store log entries
     * \param numMessages - capacity of the logMessages array, in number of elements
     * \param numConsumed - returned number of entries written to logMessages
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_NO_EVENT_AVAILABLE,
     * \return CUDBG_ERROR_NOT_SUPPORTED,
     */
    CUDBGResult (*consumeCudaLogs)(CUDBGCudaLogMessage* logMessages, uint32_t numMessages, uint32_t* numConsumed);
    /**
     * \fn CUDBGAPI_st::readCPUCallStack
     * \brief Read CPU call stack captured at the time of kernel launch.
     *
     * Since CUDA 12.9.
     *
     * This method is only supported when the CUDBG_DEBUGGER_CAPABILITY_COLLECT_CPU_CALL_STACK_FOR_KERNEL_LAUNCHES capability has been enabled.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param gridId64 - 64-bit grid ID
     * \param addrs - the returned addresses array, can be NULL
     * \param numAddrs - capacity of addrs (possibly 0)
     * \param totalNumAddrs - the actual size of the stack (number of frames) is written here; the value written can be greater than numAddrs
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_GRID,
     * \return CUDBG_ERROR_UNINITIALIZED,
     * \return CUDBG_ERROR_INTERNAL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*readCPUCallStack)(uint32_t dev, uint64_t gridId64, uint64_t *addrs, uint32_t numAddrs, uint32_t* totalNumAddrs);

    /* 13.0 Extensions */
    /**
     * \fn CUDBGAPI_st::getCudaExceptionString
     * \brief Get error string for CUDA Exceptions.
     *
     * Since CUDA 13.0.
     *
     * \ingroup READ
     *
     * \param dev - device index
     * \param sm - SM index
     * \param wp - warp index
     * \param ln - lane index
     * \param buf - buffer for the error string
     * \param bufSz - buffer size
     * \param msgSz - error message size with null character, can be null
     *
     * \return CUDBG_SUCCESS,
     * \return CUDBG_ERROR_INVALID_DEVICE,
     * \return CUDBG_ERROR_INVALID_SM,
     * \return CUDBG_ERROR_INVALID_WARP,
     * \return CUDBG_ERROR_INVALID_LANE,
     * \return CUDBG_ERROR_INVALID_ARGS,
     * \return CUDBG_ERROR_BUFFER_TOO_SMALL,
     * \return CUDBG_ERROR_NOT_SUPPORTED
     */
    CUDBGResult (*getCudaExceptionString)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, char *buf, uint32_t bufSz, uint32_t *msgSz);

    /**
     * \fn CUDBGAPI_st::setNotifyNewEventCallback
     * \brief Provides the API with the function to call to notify the debugger of a new application or device event.
     *
     * Since CUDA 13.0.
     *
     * \ingroup EVENT
     *
     * \param callback - the callback function
     * \param userData - user data to be passed to the callback
     *
     * \return CUDBG_SUCCESS
     */
    CUDBGResult (*setNotifyNewEventCallback)(CUDBGNotifyNewEventCallback callback, void* userData);
};

#ifdef __cplusplus
}
#endif

#endif
