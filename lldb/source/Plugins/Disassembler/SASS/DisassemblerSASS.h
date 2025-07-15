//===-- DisassemblerSASS.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASS_H
#define LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASS_H

#include <memory>
#include <string>

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/Error.h"

class InstructionSASS;

class DisassemblerSASS : public lldb_private::Disassembler {
public:
  explicit DisassemblerSASS(const lldb_private::ArchSpec &arch,
                            const char *flavor = nullptr,
                            const char *cpu = nullptr,
                            const char *features = nullptr);

  ~DisassemblerSASS() override;

  // Non-copyable and non-movable
  DisassemblerSASS(const DisassemblerSASS &) = delete;
  DisassemblerSASS &operator=(const DisassemblerSASS &) = delete;
  DisassemblerSASS(DisassemblerSASS &&) = delete;
  DisassemblerSASS &operator=(DisassemblerSASS &&) = delete;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "nvptx-nvdisasm"; }

  static lldb::DisassemblerSP CreateInstance(const lldb_private::ArchSpec &arch,
                                             const char *flavor,
                                             const char *cpu,
                                             const char *features);

  size_t DecodeInstructions(const lldb_private::Address &base_addr,
                            const lldb_private::DataExtractor &data,
                            lldb::offset_t data_offset, size_t num_instructions,
                            bool append, bool data_from_file) override;

protected:
  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  friend class InstructionSASS;

  bool FlavorValidForArchSpec(const lldb_private::ArchSpec &arch,
                              const char *flavor) override;

  bool IsValid() const;

private:
  /// Find the nvdisasm executable (cached after first call)
  /// \return FileSpec for nvdisasm executable, or error if not found
  llvm::Expected<lldb_private::FileSpec> FindNvdisasm();

  /// Run nvdisasm on the provided binary data and parse the output
  /// \param[in] data The binary data to disassemble
  /// \param[in] base_addr The base address for the instructions
  /// \param[in] max_instructions Maximum number of instructions to decode
  /// \return Number of instructions successfully decoded, or error
  llvm::Expected<size_t>
  DisassembleWithNvdisasm(const lldb_private::DataExtractor &data,
                          const lldb_private::Address &base_addr,
                          size_t max_instructions);

  /// Parse nvdisasm JSON output and create instructions
  /// \param[in] json_output JSON output from nvdisasm
  /// \param[in] base_addr Base address for calculating instruction addresses
  /// \param[in] max_instructions Maximum number of instructions to parse
  /// \return Number of instructions successfully parsed, or error
  llvm::Expected<size_t> ParseJsonOutput(const std::string &json_output,
                                         const lldb_private::Address &base_addr,
                                         size_t max_instructions);

  /// Extract CUDA SM architecture from the module's .note.nv.cuinfo section
  /// \param[in] base_addr Address to get the module from
  /// \return SM architecture string (e.g., "SM86") or error if extraction fails
  llvm::Expected<std::string>
  ExtractSmArchFromModule(const lldb_private::Address &base_addr);

  lldb_private::FileSpec m_nvdisasm_path;
  bool m_valid;
};

#endif // LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASS_H
