//===-- DisassemblerIntelGT.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DisassemblerIntelGT.h"
#include "Plugins/Platform/IntelGT/PlatformIntelGT.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

#include <cinttypes>
#include <cstring>
#include <dlfcn.h>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(DisassemblerIntelGT)

// ---------------------------------------------------------------------------
// IGA constants
// Local copies of iga.h values so this file has no header dependency.
// ---------------------------------------------------------------------------

#define XE_VER(MAJ, MIN) (((MAJ) << 24) | (MIN))
static constexpr int IGA_SUCCESS = 0;
static constexpr int IGA_XE_HPG = XE_VER(1, 2);
static constexpr int IGA_XE_HPC = XE_VER(1, 4);
static constexpr int IGA_XE2 = XE_VER(2, 0);

// Layout must match iga_context_options_t.
struct IgaContextOptions {
  size_t cb;
  int gen;
};

// Layout must match iga_disassemble_options_t.
struct IgaDisasmOptions {
  uint32_t cb;
  uint32_t formatting_opts;
  uint32_t _reserved0;
  uint32_t _reserved1;
  uint32_t _reserved2;
  uint32_t decoder_opts;
};

// ---------------------------------------------------------------------------
// Instruction class
// ---------------------------------------------------------------------------

class DisassemblerIntelGT::InstructionIntelGT : public Instruction {
public:
  InstructionIntelGT(const Address &addr, uint32_t size, const char *text,
                     const uint8_t *bytes)
      : Instruction(addr, AddressClass::eCode), m_size(size) {
    if (text)
      m_text = text;
    // Raw opcode bytes so LLDB flags the instruction as valid.
    m_opcode.SetOpcodeBytes(bytes, size);
  }

  void CalculateMnemonicOperandsAndComment(const ExecutionContext *) override {
    // IGA returns the whole instruction as a single string.
    m_opcode_name = m_text;
    m_markup_opcode_name = m_text;
  }

  bool DoesBranch() override { return false; }
  bool HasDelaySlot() override { return false; }
  bool IsLoad() override { return false; }
  bool IsAuthenticated() override { return false; }
  size_t Decode(const Disassembler &, const DataExtractor &,
                lldb::offset_t) override {
    return m_size;
  }

private:
  uint32_t m_size;
  std::string m_text;
};

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

void DisassemblerIntelGT::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Intel GT EU ISA disassembler (via libiga64)",
                                CreateInstance);
}

void DisassemblerIntelGT::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::DisassemblerSP DisassemblerIntelGT::CreateInstance(const ArchSpec &arch,
                                                         const char *,
                                                         const char *,
                                                         const char *) {
  Log *log = GetLog(LLDBLog::Disassembler);
  LLDB_LOGF(log,
            "[IntelGT] CreateInstance: arch='%s' triple='%s' "
            "archKind=%d spirv64=%d",
            arch.GetArchitectureName(), arch.GetTriple().getTriple().c_str(),
            (int)arch.GetTriple().getArch(), (int)llvm::Triple::spirv64);
  if (arch.GetTriple().getArch() != llvm::Triple::spirv64)
    return {};
  auto sp = std::make_shared<DisassemblerIntelGT>(arch);
  if (!sp->m_iga_ctx)
    return {}; // libiga64 not available
  return sp;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

DisassemblerIntelGT::DisassemblerIntelGT(const ArchSpec &arch)
    : Disassembler(arch, nullptr) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // Honor plugin.platform.intelgt.libiga-path if set.
  FileSpec libiga_setting =
      platform_IntelGT::PlatformIntelGT::GetGlobalProperties().GetLibIgaPath();
  if (!libiga_setting.GetPath().empty()) {
    const std::string libiga_path = libiga_setting.GetPath();
    LLDB_LOGF(log, "[IntelGT] using libiga64 path from settings: %s",
              libiga_path.c_str());
    m_iga.lib_handle = dlopen(libiga_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!m_iga.lib_handle)
      LLDB_LOGF(log, "[IntelGT] dlopen('%s') failed: %s", libiga_path.c_str(),
                dlerror());
  } else {
    // Fall back to SONAMEs resolved via LD_LIBRARY_PATH / ld.so cache.
    static const char *lib_names[] = {
        "libiga64.so",
        "libiga64.so.2",
        nullptr,
    };
    for (const char **name = lib_names; *name; ++name) {
      m_iga.lib_handle = dlopen(*name, RTLD_NOW | RTLD_GLOBAL);
      if (m_iga.lib_handle)
        break;
    }
  }
  if (!m_iga.lib_handle) {
    LLDB_LOGF(log, "[IntelGT] libiga64 not found. Set "
                   "plugin.platform.intelgt.libiga-path or make libiga64.so "
                   "resolvable via LD_LIBRARY_PATH.");
    return;
  }
  LLDB_LOGF(log, "[IntelGT] dlopen libiga64 OK: %p", m_iga.lib_handle);

  m_iga.context_create = reinterpret_cast<decltype(m_iga.context_create)>(
      dlsym(m_iga.lib_handle, "iga_context_create"));
  m_iga.context_release = reinterpret_cast<decltype(m_iga.context_release)>(
      dlsym(m_iga.lib_handle, "iga_context_release"));
  m_iga.disassemble_instruction =
      reinterpret_cast<decltype(m_iga.disassemble_instruction)>(
          dlsym(m_iga.lib_handle, "iga_context_disassemble_instruction"));

  LLDB_LOGF(log, "[IntelGT] dlsym: create=%p release=%p disasm=%p",
            (void *)m_iga.context_create, (void *)m_iga.context_release,
            (void *)m_iga.disassemble_instruction);
  if (!m_iga.context_create || !m_iga.context_release ||
      !m_iga.disassemble_instruction)
    return;

  // TODO: pass the actual device ID via ArchSpec or settings.
  IgaContextOptions opts = {sizeof(opts), IGA_XE_HPG};
  int rc = m_iga.context_create(&opts, &m_iga_ctx);
  LLDB_LOGF(log, "[IntelGT] iga_context_create(XE_HPG=%d) -> rc=%d ctx=%p",
            IGA_XE_HPG, rc, m_iga_ctx);
  if (rc != IGA_SUCCESS)
    m_iga_ctx = nullptr;
}

DisassemblerIntelGT::~DisassemblerIntelGT() {
  if (m_iga_ctx && m_iga.context_release)
    m_iga.context_release(m_iga_ctx);
  if (m_iga.lib_handle)
    dlclose(m_iga.lib_handle);
}

// ---------------------------------------------------------------------------
// DecodeInstructions
// ---------------------------------------------------------------------------

size_t DisassemblerIntelGT::DecodeInstructions(const Address &base_addr,
                                               const DataExtractor &data,
                                               lldb::offset_t data_offset,
                                               size_t num_instructions,
                                               bool append,
                                               bool data_from_file) {
  Log *log = GetLog(LLDBLog::Disassembler);
  if (!append)
    m_instruction_list.Clear();
  if (!m_iga_ctx)
    return 0;

  const uint8_t *bytes = data.GetDataStart() + data_offset;
  size_t bytes_left = data.GetByteSize() - data_offset;
  size_t parsed = 0;
  Address inst_addr(base_addr);

  while (parsed < num_instructions && bytes_left >= 8) {
    // Detect compact (8B) vs full (16B) via CmptCtrl bit 29.
    bool compact = (bytes[3] & 0x20) != 0;
    uint32_t inst_len = compact ? 8 : 16;
    if (bytes_left < inst_len)
      break;

    char *text = nullptr;
    IgaDisasmOptions dopts = {};
    dopts.cb = sizeof(dopts);
    int status = m_iga.disassemble_instruction(m_iga_ctx, &dopts, bytes,
                                               nullptr, nullptr, &text);

    if (parsed < 3) // Log first few
      LLDB_LOGF(log,
                "[IntelGT] insn @+%zu: status=%d compact=%d "
                "bytes=%02x%02x%02x%02x %02x%02x%02x%02x text='%s'",
                (size_t)(bytes - (data.GetDataStart() + data_offset)), status,
                (int)compact, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4],
                bytes[5], bytes[6], bytes[7], text ? text : "(null)");

    const char *mnemonic = (status == IGA_SUCCESS && text) ? text : "???";
    InstructionSP inst_sp(
        new InstructionIntelGT(inst_addr, inst_len, mnemonic, bytes));
    m_instruction_list.Append(inst_sp);

    bytes += inst_len;
    bytes_left -= inst_len;
    inst_addr.Slide(inst_len);
    ++parsed;
  }

  return parsed;
}
