//===-- DisassemblerIntelGT.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DISASSEMBLER_INTELGT_DISASSEMBLERINTELGT_H
#define LLDB_SOURCE_PLUGINS_DISASSEMBLER_INTELGT_DISASSEMBLERINTELGT_H

#include "lldb/Core/Disassembler.h"

namespace lldb_private {

class DisassemblerIntelGT : public Disassembler {
public:
  DisassemblerIntelGT(const ArchSpec &arch);
  ~DisassemblerIntelGT() override;

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "intelgt"; }
  static lldb::DisassemblerSP CreateInstance(const ArchSpec &arch,
                                             const char *flavor,
                                             const char *cpu,
                                             const char *features);

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  size_t DecodeInstructions(const Address &base_addr, const DataExtractor &data,
                            lldb::offset_t data_offset, size_t num_instructions,
                            bool append, bool data_from_file) override;

  bool FlavorValidForArchSpec(const ArchSpec &arch,
                              const char *flavor) override {
    return true;
  }

  // dlopen'd IGA function pointers.
  struct IGAFunctions {
    void *lib_handle = nullptr;
    int (*context_create)(const void *opts, void **ctx) = nullptr;
    void (*context_release)(void *ctx) = nullptr;
    int (*disassemble_instruction)(void *ctx, const void *dopts,
                                   const void *input,
                                   const char *(*)(int32_t, void *), void *,
                                   char **) = nullptr;
  };

private:
  class InstructionIntelGT;

  IGAFunctions m_iga;
  void *m_iga_ctx = nullptr;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_DISASSEMBLER_INTELGT_DISASSEMBLERINTELGT_H
