//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_INSTRUCTIONSASS_H
#define LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_INSTRUCTIONSASS_H

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

/// SASS Instruction implementation with nvdisasm JSON schema support
/// Provides accurate instruction classification using schema attributes when
/// available, with comprehensive fallback pattern implementation for maximum
/// compatibility across different nvdisasm versions and CUDA architectures.
class InstructionSASS : public lldb_private::Instruction {
public:
  InstructionSASS(const lldb_private::Address &address,
                  const std::string &opcode, const std::string &operands,
                  lldb_private::AddressClass addr_class);

  InstructionSASS(const lldb_private::Address &address,
                  const std::string &opcode, const std::string &operands,
                  const std::string &predicate, const std::string &extra,
                  const std::map<std::string, std::string> &other_attributes,
                  const std::vector<std::string> &other_flags,
                  lldb_private::AddressClass addr_class);

  bool DoesBranch() override;

  bool HasDelaySlot() override {
    // SASS doesn't have delay slots
    return false;
  }

  bool IsLoad() override;

  bool IsAuthenticated() override {
    // SASS doesn't have authenticated instructions
    return false;
  }

  bool IsCall() override;

  bool IsBarrier();

  const std::string &GetPredicate() const { return m_predicate; }
  const std::string &GetExtra() const { return m_extra; }
  const std::map<std::string, std::string> &GetOtherAttributes() const {
    return m_other_attributes;
  }
  const std::vector<std::string> &GetOtherFlags() const {
    return m_other_flags;
  }

  void CalculateMnemonicOperandsAndComment(
      const lldb_private::ExecutionContext *exe_ctx) override {
    // Already calculated in constructor, nothing to do
  }

  size_t Decode(const lldb_private::Disassembler &disassembler,
                const lldb_private::DataExtractor &data,
                lldb::offset_t data_offset) override;

  void SetOpcode(const void *opcode_data, size_t opcode_data_len);

  /// \return
  ///     The byte size of a SASS instruction.  SM versions >= 70 have 16 byte
  ///     instructions, while older versions have 8.  We don't support the older
  ///     versions.
  static size_t GetInstructionByteSize() { return 16; }

private:
  std::string m_predicate;
  std::string m_extra;
  std::map<std::string, std::string> m_other_attributes;
  std::vector<std::string> m_other_flags;

  // Cached attribute flags to avoid repeated lookups
  std::optional<bool> m_is_control_flow;
  std::optional<bool> m_is_subroutine_call;
  std::optional<bool> m_is_barrier;
  std::optional<bool> m_is_load;
};

#endif // LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_INSTRUCTIONSASS_H
