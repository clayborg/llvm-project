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

#ifndef __CUDACOREDUMP_H__
#define __CUDACOREDUMP_H__

#include <stdint.h>

/*
 * cudacoredump.h - Public format description of the CUDA coredump
 */

/* ==================== */
/* ===== Overview ===== */
/* ==================== */

/*
 * CUDA coredumps are ELF files with the following identifying header fields:
 *   - abi: ELFOSABI_CUDA (0x33)
 *   - machine: EM_CUDA (0xbe)
 *   - type: ET_CORE (4)
 *
 * Coredump data is stored in separate sections, each of which is described
 * below. Note that new fields can be added to each of these sections in new
 * driver versions, so any coredump readers need to be careful and check each
 * section's element size before accessing the fields that were not present
 * in the baseline version of the section. Section descriptions below make it
 * clear which fields might not be present.
 *
 * Coredump sections are named hierarchically, the hierarchy looks like this
 * (but not necessarily in this order):
 *   - Metadata
 *   - Global memory
 *   - CUDA device information
 *     - CUDA context information
 *       - Loaded modules information
 *         - Module's relocated ELF image (cubin)
 *         - Module's non-relocated ELF image (cubin)
 *     - CUDA grid information
 *       - Grid parameter memory
 *       - Grid constbank information
 *     - SM information
 *       - Block (CTA) information
 *         - Block shared memory
 *         - Warp information
 *           - Warp uniform registers
 *           - Warp uniform predicates
 *           - Warp convergence barrier masks
 *           - Thread information
 *             - Thread local memory
 *             - Thread registers
 *             - Thread predicates
 *             - Thread call stack
 *
 * More information about each particular section is given below.
 *
 * ===== Brief format history =====
 *
 * CUDA Driver r346:
 *   - Initial CUDA coredump functionality release
 * CUDA Driver r400:
 *   - Added uniform registers and uniform predicates
 * CUDA Driver r525:
 *   - Added cluster index and cluster dimensions
 *   - Added number of registers per warp
 * CUDA Driver r550:
 *   - Added constbank information
 * CUDA Driver r555:
 *   - Added exception information per SM
 * CUDA Driver r565:
 *   - Added metadata section
 *   - Added preferred cluster dimensions
 *   - Added per-CTA cluster dimensions
 *   - Added cluster exception target block index
 * CUDA Driver r570:
 *   - Added per-warp shared memory usage
 * CUDA Driver r575:
 *   - Added per-warp in syscall lanes mask
 *   - Added per-warp CBU active lanes mask
 *   - Added per-warp CBU exited lanes mask
 *   - Added per-warp CBU collective lanes mask
 *   - Added per-warp convergence barrier masks
 *   - Added per-lane thread state
 * CUDA Driver r580:
 *   - Added CUDA exception string
 * CUDA Driver r590:
 *   - Added per-warp barrier scope and additional barrier information
 * CUDA Driver r615:
 *   - Added RPC.LO and RPC.HI per-thread register values
 */

/* ======================================= */
/* ===== Section-related definitions ===== */
/* ======================================= */

#ifndef SHT_LOUSER
#define SHT_LOUSER 0x80000000
#endif

/* CUDA coredump section types.
 * See corresponding sections below for usage.
 */
typedef enum {
    CUDBG_SHT_MANAGED_MEM = SHT_LOUSER + 1,
    CUDBG_SHT_GLOBAL_MEM  = SHT_LOUSER + 2,
    CUDBG_SHT_LOCAL_MEM   = SHT_LOUSER + 3,
    CUDBG_SHT_SHARED_MEM  = SHT_LOUSER + 4,
    CUDBG_SHT_DEV_REGS    = SHT_LOUSER + 5,
    CUDBG_SHT_ELF_IMG     = SHT_LOUSER + 6,
    CUDBG_SHT_RELF_IMG    = SHT_LOUSER + 7,
    CUDBG_SHT_BT          = SHT_LOUSER + 8,
    CUDBG_SHT_DEV_TABLE   = SHT_LOUSER + 9,
    CUDBG_SHT_CTX_TABLE   = SHT_LOUSER + 10,
    CUDBG_SHT_SM_TABLE    = SHT_LOUSER + 11,
    CUDBG_SHT_GRID_TABLE  = SHT_LOUSER + 12,
    CUDBG_SHT_CTA_TABLE   = SHT_LOUSER + 13,
    CUDBG_SHT_WP_TABLE    = SHT_LOUSER + 14,
    CUDBG_SHT_LN_TABLE    = SHT_LOUSER + 15,
    CUDBG_SHT_MOD_TABLE   = SHT_LOUSER + 16,
    CUDBG_SHT_DEV_PRED    = SHT_LOUSER + 17,
    CUDBG_SHT_PARAM_MEM   = SHT_LOUSER + 18,
    /* Since CUDA Driver r400 */
    CUDBG_SHT_DEV_UREGS   = SHT_LOUSER + 19,
    CUDBG_SHT_DEV_UPRED   = SHT_LOUSER + 20,
    /* Since CUDA Driver r550 */
    CUDBG_SHT_CB_TABLE    = SHT_LOUSER + 21,
    /* Since CUDA Driver r565 */
    CUDBG_SHT_META_DATA   = SHT_LOUSER + 22,
    /* Since CUDA Driver r575 */
    CUDBG_SHT_CBU_BAR     = SHT_LOUSER + 23,
} CudbgSectionHeaderTypes;

/* CUDA section name prefixes.
 * See corresponding sections below for usage.
 */
#define CUDBG_SHNAME_GLOBAL    ".cudbg.global"
#define CUDBG_SHNAME_LOCAL     ".cudbg.local"
#define CUDBG_SHNAME_SHARED    ".cudbg.shared"
#define CUDBG_SHNAME_REGS      ".cudbg.regs"
#define CUDBG_SHNAME_PARAM     ".cudbg.param"
#define CUDBG_SHNAME_PRED      ".cudbg.pred"
#define CUDBG_SHNAME_DEVTABLE  ".cudbg.devtbl"
#define CUDBG_SHNAME_CTXTABLE  ".cudbg.ctxtbl"
#define CUDBG_SHNAME_SMTABLE   ".cudbg.smtbl"
#define CUDBG_SHNAME_GRIDTABLE ".cudbg.gridtbl"
#define CUDBG_SHNAME_CTATABLE  ".cudbg.ctatbl"
#define CUDBG_SHNAME_WPTABLE   ".cudbg.wptbl"
#define CUDBG_SHNAME_LNTABLE   ".cudbg.lntbl"
#define CUDBG_SHNAME_BT        ".cudbg.bt"
#define CUDBG_SHNAME_MODTABLE  ".cudbg.modtbl"
#define CUDBG_SHNAME_ELFIMG    ".cudbg.elfimg"
#define CUDBG_SHNAME_RELFIMG   ".cudbg.relfimg"
/* Since CUDA Driver r400 */
#define CUDBG_SHNAME_UREGS     ".cudbg.uregs"
#define CUDBG_SHNAME_UPRED     ".cudbg.upred"
/* Since CUDA Driver r550 */
#define CUDBG_SHNAME_CBTABLE   ".cudbg.cbankstbl"
/* Since CUDA Driver r565 */
#define CUDBG_SHNAME_META_DATA ".cudbg.meta"
/* Since CUDA Driver r575 */
#define CUDBG_SHNAME_CBU_BAR   ".cudbg.cbu_bar"

/* ========================================= */
/* ===== Detailed section descriptions ===== */
/* ========================================= */

/* Global memory
 *
 * Contains raw global memory (does not have a separate struct in this file).
 * These sections are dumped in order and their names contain the monotonically
 * increasing index, will be referred to as <memIdx> below.
 * Not present if dumping memory is disabled.
 *
 * Section name format: sprintf("%s.%d", CUDBG_SHNAME_GLOBAL, <memIdx>)
 * Section header type: CUDBG_SHT_GLOBAL_MEM (or, if managed memory, CUDBG_SHT_MANAGED_MEM)
 * Section element type: byte data
 * Section addr: global address of the start of this memory block
 * Section link: 0
 * Section info: 0
 */

/*
 * CUDA device table
 *
 * Contains descriptions of all CUDA devices visible to the application
 * at the moment of coredump generation. Device index in this table is used
 * in other section names, will be referred to as <devIdx> below.
 * Only one such section per coredump file.
 *
 * Section name format: CUDBG_SHNAME_DEVTABLE (no suffix)
 * Section header type: CUDBG_SHT_DEV_TABLE
 * Section element type: CudbgDeviceTableEntry
 * Section link: 0
 * Section info: 0
 *
 * Related sections:
 *
 * 1. CUDA context table, per device - see below
 *
 * 2. CUDA grid table, per device - see below
 *
 * 3. SM information, per device - see below
 */
typedef struct {
    /* Display name of the device
     * This field is an index into the string table.
     */
    uint64_t devName;
    /* Internal name of the device
     * This field is an index into the string table.
     */
    uint64_t devType;
    /* ISA version of the device
     * This field is an index into the string table.
     */
    uint64_t smType;
    /* CUDA device ID */
    uint32_t devId;
    /* PCI bus ID of the device */
    uint32_t pciBusId;
    /* PCI device ID of the device */
    uint32_t pciDevId;
    /* Number of SMs this device has */
    uint32_t numSMs;
    /* Number of warps in each SM */
    uint32_t numWarpsPerSM;
    /* Number of lanes in each warp */
    uint32_t numLanesPerWarp;
    /* Maximum number of registers per lane
     * Use CudbgGridTableEntry::numRegs and CudbgWarpTableEntry::numRegs
     * for the actual number of registers per grid and warp.
     */
    uint32_t numRegsPerLane;
    /* Number of predicates per lane */
    uint32_t numPredicatesPrLane;
    /* Major version of the SM */
    uint32_t smMajor;
    /* Minor version of the SM */
    uint32_t smMinor;
    /* GPU instruction size in bytes */
    uint32_t instructionSize;
    /* Device status
     * This field is of type CUDBGResult, see cudadebugger.h.
     */
    uint32_t status;

    /* ================================== */
    /* ===== Since CUDA Driver r400 ===== */
    /* ================================== */

    /* Number of uniform registers per warp */
    uint32_t numUniformRegsPrWarp;
    /* Number of uniform predicates per warp */
    uint32_t numUniformPredicatesPrWarp;

    /* ================================== */
    /* ===== Since CUDA Driver r575 ===== */
    /* ================================== */

    /* Maximum number of convergence barriers per warp */
    uint32_t numConvergenceBarriersPrWarp;
    /* Padding, ignore */
    uint32_t padding0;
} CudbgDeviceTableEntry;

/*
 * CUDA context table, per device
 *
 * Contains descriptions of all CUDA contexts for a particular device.
 * Context index in this table is used in other section names,
 * will be referred to as <ctxIdx> below.
 *
 * Section name format: sprintf("%s.dev%d", CUDBG_SHNAME_CTXTABLE, devIdx)
 * Section header type: CUDBG_SHT_CTX_TABLE
 * Section element type: CudbgContextTableEntry
 * Section link: section header index of the (unique) CUDA device table
 * Section info: devIdx
 *
 * Related sections:
 *
 * 1. Loaded modules table, per context - see below
 */
typedef struct {
    /* Handle of this context */
    uint64_t contextId;
    /* Global address of the start of the shared memory window */
    uint64_t sharedWindowBase;
    /* Global address of the start of the local memory window */
    uint64_t localWindowBase;
    /* Global address of the start of the global memory window */
    uint64_t globalWindowBase;
    /* CUDA device ID of the containing device */
    uint32_t deviceIdx;
    /* Thread ID of the host thread that owns this context */
    uint32_t tid;
} CudbgContextTableEntry;

/*
 * Loaded modules table, per context
 *
 * Contains information about all modules loaded in a particular context.
 * Module index in this table is used in other section names,
 * will be referred to as <modIdx> below.
 *
 * Section name format: sprintf("%s.dev%d.ctx%d", CUDBG_SHNAME_MODTABLE, devIdx, ctxIdx)
 * Section header type: CUDBG_SHT_MOD_TABLE
 * Section element type: CudbgModuleTableEntry
 * Section link: section header index of the corresponding CUDA context table
 * Section info: ctxIdx
 *
 * Related sections:
 *
 * 1. Relocated ELF image (cubin) of the module
 *
 * Contains raw cubin data (does not have a separate struct in this file).
 * There can be several sections with the same name since the module index
 * is not a part of the section name. They can be distinguished by the
 * section header's link or info fields.
 *
 * Section name format: sprintf("%s.dev%d.ctx%d", CUDBG_SHNAME_RELFIMG, devIdx, ctxIdx)
 * Section header type: CUDBG_SHT_RELF_IMG
 * Section element type: byte data
 * Section link: section header index of the corresponding module table
 * Section info: modIdx
 *
 * 2. Non-relocated ELF image (cubin) of the module
 *
 * Contains raw cubin data (does not have a separate struct in this file).
 * There can be several sections with the same name since the module index
 * is not a part of the section name. They can be distinguished by the
 * section header's link or info fields.
 * Not present if dumping non-relocated cubins is disabled.
 *
 * Section name format: sprintf("%s.dev%d.ctx%d", CUDBG_SHNAME_ELFIMG, devIdx, ctxIdx)
 * Section header type: CUDBG_SHT_ELF_IMG
 * Section element type: byte data
 * Section link: section header index of the corresponding module table
 * Section info: modIdx
 */
typedef struct {
    /* Handle of the loaded module */
    uint64_t moduleHandle;
} CudbgModuleTableEntry;

/*
 * CUDA grid table, per device
 *
 * Contains descriptions of all grids running on the device at the moment of
 * coredump generation. Grid index in this table is used in other section names,
 * will be referred to as <gridIdx> below.
 *
 * Section name format: sprintf("%s.dev%d", CUDBG_SHNAME_GRIDTABLE, devIdx)
 * Section header type: CUDBG_SHT_GRID_TABLE
 * Section element type: CudbgGridTableEntry
 * Section link: section header index of the (unique) CUDA device table
 * Section info: devIdx
 *
 * Related sections:
 *
 * 1. CUDA grid param memory, per grid
 *
 * Contains raw grid parameter memory (does not have a separate struct
 * in this file). Not present if dumping memory is disabled.
 *
 * Section name format: sprintf("%s.dev%d.grid%d", CUDBG_SHNAME_PARAM, devIdx, gridIdx)
 * Section header type: CUDBG_SHT_PARAM_MEM
 * Section element type: byte data
 * Section link: section header index of the corresponding CUDA grid table
 * Section info: gridIdx
 *
 * 2. CUDA constbank information, per grid - see below
 */
typedef struct {
    /* Grid ID, an opaque 64bit number */
    uint64_t gridId64;
    /* Handle of the context containing this grid */
    uint64_t contextId;
    /* Handle of the kernel that this grid is executing */
    uint64_t function;
    /* Entry address of this grid's kernel */
    uint64_t functionEntry;
    /* Handler of the module containing this grid's kernel */
    uint64_t moduleHandle;
    /* Reserved deprecated field kept for ABI compatibility.
     * Used to be ID of the parent grid (in case of a device-launched CDP grid).
     * Deprecated since CUDA 13.4 */
    uint64_t reserved0;
    /* Offset in the constbank 0 where the parameters start */
    uint64_t paramsOffset;
    /* Type of this grid's kernel
     * This field is of type CUDBGKernelType, see cudadebugger.h.
     */
    uint32_t kernelType;
    /* Where this grid was launched from
     * This field is of type CUDBGKernelOrigin, see cudadebugger.h.
     */
    uint32_t origin;
    /* Status of this grid at the moment of coredump generation
     * This field is of type CUDBGGridStatus, see cudadebugger.h.
     */
    uint32_t gridStatus;
    /* Number of registers this grid uses */
    uint32_t numRegs;
    /* Grid dimension X */
    uint32_t gridDimX;
    /* Grid dimension Y */
    uint32_t gridDimY;
    /* Grid dimension Z */
    uint32_t gridDimZ;
    /* Block dimension X */
    uint32_t blockDimX;
    /* Block dimension Y */
    uint32_t blockDimY;
    /* Block dimension Z */
    uint32_t blockDimZ;
    /* Whether this grid's launch was blocking
     * This field is semantically boolean.
     */
    uint32_t attrLaunchBlocking;
    /* Thread ID of the host thread that launched this grid */
    uint32_t attrHostTid;

    /* ================================== */
    /* ===== Since CUDA Driver r525 ===== */
    /* ================================== */

    /* Cluster dimension X */
    uint32_t clusterDimX;
    /* Cluster dimension Y */
    uint32_t clusterDimY;
    /* Cluster dimension Z */
    uint32_t clusterDimZ;
    /* Padding, ignore */
    uint32_t padding0;

    /* ================================== */
    /* ===== Since CUDA Driver r565 ===== */
    /* ================================== */

    /* Preferred cluster dimension X */
    uint32_t preferredClusterDimX;
    /* Preferred cluster dimension Y */
    uint32_t preferredClusterDimY;
    /* Preferred cluster dimension Z */
    uint32_t preferredClusterDimZ;
    /* Padding, ignore */
    uint32_t padding1;
} CudbgGridTableEntry;

/*
 * CUDA constbank information, per grid
 *
 * Since CUDA Driver r550.
 *
 * Contains information about all constbanks for a given grid.
 *
 * Section name format: sprintf("%s.dev%u.grid%u", CUDBG_SHNAME_CBTABLE, devIdx, gridIdx)
 * Section header type: CUDBG_SHT_CB_TABLE
 * Section element type: CudbgConstBankTableEntry
 * Section link: section header index of the corresponding CUDA grid table
 * Section info: gridIdx
 */
typedef struct {
    /* Global address of this constbank's start */
    uint64_t addr;
    /* Size of this constbank in bytes */
    uint32_t size;
    /* ID (number) of this constbank */
    uint32_t bankId;
} CudbgConstBankTableEntry;

/*
 * SM information, per device
 *
 * Contains information about all SMs for a given device. SM index in this
 * table is used in other section names, will be referred to as <smIdx> below.
 *
 * Section name format: sprintf("%s.dev%d", CUDBG_SHNAME_SMTABLE, devIdx)
 * Section header type: CUDBG_SHT_SM_TABLE
 * Section element type: CudbgSmTableEntry
 * Section link: section header index of the (unique) CUDA device table
 * Section info: devIdx
 *
 * Related sections:
 *
 * 1. Block (CTA) information, per SM - see below
 */
typedef struct {
    /* ID (number) of this SM */
    uint32_t smId;
    /* Padding, ignore */
    uint32_t padding0;

    /* ================================== */
    /* ===== Since CUDA Driver r555 ===== */
    /* ================================== */

    /* Exception that occurred in any of the warps.
     * This is useful when all faulted warps for an SM have
     * exited before an exception was reported.
     * This field is of type CUDBGException_t, see cudadebugger.h.
     */
    uint32_t exception;
    /* If non-zero, the following errorPC of the exception is valid.
     * Semantically boolean.
     */
    uint32_t errorPCValid;
    /* PC where an error occurred in any of the warps.
     * This is useful when all faulted warps for an SM have
     * exited before an exception was reported.
     */
    uint64_t errorPC;

    /* If non-zero, the following clusterExceptionTargetBlockIdx of the
     * exception is valid. Semantically boolean.
     */
    uint32_t clusterExceptionTargetBlockIdxValid;
    /* For cluster exceptions, following x,y,z fields represent the target block
     * index handling cluster requests.
     * Block index, X */
    uint32_t clusterExceptionTargetBlockIdxX;
    /* Block index, Y */
    uint32_t clusterExceptionTargetBlockIdxY;
    /* Block index, Z */
    uint32_t clusterExceptionTargetBlockIdxZ;

    /* ================================== */
    /* ===== Since CUDA Driver r580 ===== */
    /* ================================== */

    /* Additional information about the exception */
    uint64_t exceptionString;
} CudbgSmTableEntry;

/*
 * Block (CTA) information, per SM
 *
 * Contains information about all blocks for a given SM. Block index in this
 * table is used in other section names, will be referred to as <ctaIdx> below.
 *
 * Section name format: sprintf("%s.dev%d.sm%d", CUDBG_SHNAME_CTATABLE, devIdx, smIdx)
 * Section header type: CUDBG_SHT_CTA_TABLE
 * Section element type: CudbgCTATableEntry
 * Section link: section header index of corresponding SM table
 * Section info: smIdx
 *
 * Related sections:
 *
 * 1. Block shared memory, per block
 *
 * Contains raw block shared memory (does not have a separate struct
 * in this file). Not present if dumping memory is disabled.
 *
 * Section name format: sprintf("%s.dev%d.sm%d.cta%d", CUDBG_SHNAME_SHARED, devIdx, smIdx, ctaIdx)
 * Section header type: CUDBG_SHT_SHARED_MEM
 * Section element type: byte data
 * Section link: section header index of the corresponding block table
 * Section info: ctaIdx
 *
 * 2. Warp information, per block (CTA) - see below
 */
typedef struct {
    /* Grid ID of the grid containing this block */
    uint64_t gridId64;
    /* Block index, X */
    uint32_t blockIdxX;
    /* Block index, Y */
    uint32_t blockIdxY;
    /* Block index, Z */
    uint32_t blockIdxZ;
    /* Padding, ignore */
    uint32_t padding0;

    /* ================================== */
    /* ===== Since CUDA Driver r525 ===== */
    /* ================================== */

    /* Cluster index, X */
    uint32_t clusterIdxX;
    /* Cluster index, Y */
    uint32_t clusterIdxY;
    /* Cluster index, Z */
    uint32_t clusterIdxZ;
    /* Padding, ignore */
    uint32_t padding1;

    /* ================================== */
    /* ===== Since CUDA Driver r565 ===== */
    /* ================================== */

    /* Cluster dimension X */
    uint32_t clusterDimX;
    /* Cluster dimension Y */
    uint32_t clusterDimY;
    /* Cluster dimension Z */
    uint32_t clusterDimZ;
    /* Padding, ignore */
    uint32_t padding2;
} CudbgCTATableEntry;

/*
 * Warp information, per block (CTA)
 *
 * Contains information about all warps for a given CTA. Warp index in this
 * table is used in other section names, will be referred to as <warpIdx> below.
 *
 * Section name format: sprintf("%s.dev%d.sm%d.cta%d", CUDBG_SHNAME_WPTABLE, devIdx, smIdx, ctaIdx)
 * Section header type: CUDBG_SHT_WP_TABLE
 * Section element type: CudbgWarpTableEntry
 * Section link: section header index of corresponding block table
 * Section info: ctaIdx
 *
 * Related sections:
 *
 * 1. Uniform registers, per warp
 *
 * Since CUDA Driver r400.
 *
 * Contains raw uniform registers memory (does not have a separate struct
 * in this file). Not present if the device doesn't have uniform registers.
 *
 * Section name format:
 *     sprintf("%s.dev%d.sm%d.cta%d.wp%d", CUDBG_SHNAME_UREGS, devIdx, smIdx, ctaIdx, warpIdx)
 * Section header type: CUDBG_SHT_DEV_UREGS
 * Section element type: uint32_t data
 * Section link: section header index of the corresponding warp table
 * Section info: warpIdx
 *
 * 2. Uniform predicates, per warp
 *
 * Since CUDA Driver r400.
 *
 * Contains raw uniform predicates (does not have a separate struct in this
 * file). Not present if the device doesn't have uniform predicates.
 *
 * Section name format:
 *     sprintf("%s.dev%d.sm%d.cta%d.wp%d", CUDBG_SHNAME_UPRED, devIdx, smIdx, ctaIdx, warpIdx)
 * Section header type: CUDBG_SHT_DEV_UPRED
 * Section element type: uint32_t data (each value is semantically boolean)
 * Section link: section header index of the corresponding warp table
 * Section info: warpIdx
 *
 * 3. Convergence barrier masks, per warp
 *
 * Since CUDA Driver r575.
 *
 * Contains warp convergence barrier masks stored as uint32_t values.
 * Not present if the device doesn't have convergence barriers.
 *
 * Section name format:
 *     sprintf("%s.dev%d.sm%d.cta%d.wp%d", CUDBG_SHNAME_CBU_BAR, devIdx, smIdx, ctaIdx, warpIdx)
 * Section header type: CUDBG_SHT_CBU_BAR
 * Section element type: uint32_t data
 * Section link: section header index of corresponding warp table
 * Section info: warpIdx
 *
 * 4. Thread information, per warp - see below
 */
typedef struct {
    /* PC which has triggered a warp error
     * This field is only valid if errorPCValid is non-zero.
     */
    uint64_t errorPC;
    /* ID (number) of this warp */
    uint32_t warpId;
    /* Mask of the valid lanes */
    uint32_t validLanesMask;
    /* Mask of the active (non-diverged) lanes */
    uint32_t activeLanesMask;
    /* Indicates whether this warp has hit a breakpoint
     * This field is semantically boolean.
     */
    uint32_t isWarpBroken;
    /* Indicates whether the errorPC field is valid
     * This field is semantically boolean.
     */
    uint32_t errorPCValid;
    /* Padding, ignore */
    uint32_t padding0;

    /* ================================== */
    /* ===== Since CUDA Driver r525 ===== */
    /* ================================== */

    /* Number of registers used by this warp */
    uint32_t numRegs;
    /* Padding, ignore */
    uint32_t padding1;

    /* ================================== */
    /* ===== Since CUDA Driver r570 ===== */
    /* ================================== */

    /* Shared memory size */
    uint32_t sharedMemSize;
    /* Padding, ignore */
    uint32_t padding2;

    /* ================================== */
    /* ===== Since CUDA Driver r575 ===== */
    /* ================================== */

    /* Mask of lanes stopped in a syscall */
    uint32_t inSyscallLanesMask;
    /* Mask of lanes marked as active in CBU */
    uint32_t cbuActiveLanesMask;
    /* Mask of lanes marked as exited in CBU */
    uint32_t cbuExitedLanesMask;
    /* Mask of lanes participating in a collective region */
    uint32_t cbuCollectiveLanesMask;

    /* ================================== */
    /* ===== Since CUDA Driver r590 ===== */
    /* ================================== */

    /* This field is of type CUDBGBarrierScope */
    uint32_t barrierScope;
    /* Padding, ignore */
    uint32_t padding3;
    /* This field is an index into the string table. */
    uint64_t additionalBarrierInfo;
} CudbgWarpTableEntry;

/*
 * Thread information, per warp
 *
 * Contains information about all threads for a given warp. Thread index in this
 * table is used in other section names, will be referred to as <laneIdx> below.
 *
 * Section name format:
 *   sprintf("%s.dev%d.sm%d.cta%d.wp%d", CUDBG_SHNAME_LNTABLE, devIdx, smIdx, ctaIdx, warpIdx)
 * Section header type: CUDBG_SHT_LN_TABLE
 * Section element type: CudbgThreadTableEntry
 * Section link: section header index of corresponding warp table
 * Section info: warpIdx
 *
 * Related sections:
 *
 * 1. Thread local memory, per thread
 *
 * Contains raw thread local memory (does not have a separate struct in this
 * file). Not present if dumping memory is disabled.
 *
 * Section name format:
 *   sprintf("%s.dev%d.sm%d.cta%d.wp%d.ln%d", CUDBG_SHNAME_LOCAL, devIdx, smIdx, ctaIdx, warpIdx, laneIdx)
 * Section header type: CUDBG_SHT_LOCAL_MEM
 * Section element type: byte data
 * Section addr: local memory address of the start of the dumped local memory
 * Section link: section header index of the corresponding thread table
 * Section info: laneIdx
 *
 * 2. Thread registers
 *
 * Contains raw registers memory (does not have a separate struct in this file).
 *
 * Section name format:
 *   sprintf("%s.dev%d.sm%d.cta%d.wp%d.ln%d", CUDBG_SHNAME_REGS, devIdx, smIdx, ctaIdx, warpIdx, laneIdx)
 * Section header type: CUDBG_SHT_DEV_REGS
 * Section element type: uint32_t data
 * Section link: section header index of the corresponding thread table
 * Section info: laneIdx
 *
 * 3. Thread predicates
 *
 * Contains raw predicates memory (does not have a separate struct in this file).
 *
 * Section name format:
 *   sprintf("%s.dev%d.sm%d.cta%d.wp%d.ln%d", CUDBG_SHNAME_PRED, devIdx, smIdx, ctaIdx, warpIdx, laneIdx)
 * Section header type: CUDBG_SHT_DEV_PRED
 * Section element type: uint32_t data (each value is semantically boolean)
 * Section link: section header index of the corresponding thread table
 * Section info: laneIdx
 *
 * 4. Thread call stack, per thread - see below
 */
typedef struct {
    /* PC of this thread
     * In coredumps generated with the new unified backend this will be
     * a global address in GPU memory corresponding to the physical PC.
     * In coredumps generated with the older classic backend this will be
     * a global address in CPU memory where a copy of the GPU code is stored.
     */
    uint64_t virtualPC;
    /* Offset of this thread's PC from the start of the currently executing function */
    uint64_t physPC;
    /* Lane ID (number) of this thread */
    uint32_t ln;
    /* Thread index, X */
    uint32_t threadIdxX;
    /* Thread index, Y */
    uint32_t threadIdxY;
    /* Thread index, Z */
    uint32_t threadIdxZ;
    /* Exception hit by this thread, if any
     * This field is of type CUDBGException_t, see cudadebugger.h.
     */
    uint32_t exception;
    /* Call depth of the call stack of this thread
     * This field's value includes the number of frames that are in syscall.
     */
    uint32_t callDepth;
    /* Number of call stack frames that are in a syscall */
    uint32_t syscallCallDepth;
    /* Special CC register, only available on pre-Volta GPUs */
    uint32_t ccRegister;

    /* ================================== */
    /* ===== Since CUDA Driver r575 ===== */
    /* ================================== */

    /* CBU thread state
     * This field is of type CUDBGCBUThreadState, see cudadebugger.h.
     */
    uint32_t cbuThreadState;
    /* Padding, ignore */
    uint32_t padding0;

    /* ================================== */
    /* ===== Since CUDA Driver r615 ===== */
    /* ================================== */

    /* RPC.LO register value */
    uint32_t rpcLo;
    /* RPC.HI register value */
    uint32_t rpcHi;
} CudbgThreadTableEntry;

/*
 * Thread call stack, per thread
 *
 * Contains information about the stack frames for a given thread.
 *
 * Section name format:
 *   sprintf("%s.dev%d.sm%d.cta%d.wp%d.ln%d", CUDBG_SHNAME_BT, devIdx, smIdx, ctaIdx, warpIdx, laneIdx)
 * Section header type: CUDBG_SHT_BT
 * Section element type: CudbgBacktraceTableEntry
 * Section link: section header index of corresponding thread table
 * Section info: laneIdx
 */
typedef struct {
    /* Offset of the return address from the start of the caller function */
    uint64_t returnAddress;
    /* Return address of this call stack frame
     * In coredumps generated with the new unified backend this will be
     * a global address in GPU memory containing the code to return to.
     * In coredumps generated with the older classic backend this will be
     * a global address in CPU memory containing a copy of the GPU code.
     */
    uint64_t virtualReturnAddress;
    /* Stack frame level */
    uint32_t level;
    /* Padding, ignore */
    uint32_t pad;
} CudbgBacktraceTableEntry;

/*
 * CUDA coredump metadata.
 *
 * Since CUDA Driver r565.
 *
 * Contains additional information about the coredump.
 *
 * Section name format: CUDBG_SHNAME_META_DATA
 * Section header type: CUDBG_SHT_META_DATA
 * Section element type: CudbgMetaDataEntry
 * Section link: 0
 * Section info: 0
 */
typedef struct {
    /* Identifier for the generator of the coredump.
     * This field is an index into the string table.
     */
    uint64_t generatorName;
    /* The version of the GPU driver as reported by NVML API. Not set on Tegra. */
    uint32_t driverVersionMajor;
    uint32_t driverVersionMinor;
    /* The version of the CUDA driver as reported by the driver API (e.g. 12/7) */
    uint32_t cudaDriverVersionMajor;
    uint32_t cudaDriverVersionMinor;
    /* Flags used to generate the coredump (CUDBGCoredumpGenerationFlags) */
    uint32_t flags;
    /* Timestamp of this coredump, in seconds since the UNIX Epoch */
    uint32_t timestamp;
} CudbgMetaDataEntry;

#endif // __CUDACOREDUMP_H__
