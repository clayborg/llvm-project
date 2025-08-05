//===-- InstructionSASS.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstructionSASS.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

static std::string ColorizeString(const std::string &text,
                                  llvm::raw_ostream::Colors color) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream.enable_colors(true);
  stream.changeColor(color);
  stream << text;
  stream.resetColor();
  return stream.str();
}

template <typename T, typename E>
static bool IsASubstringOfAny(const E &element, std::initializer_list<T> set) {
  for (const T &v : set)
    if (element.find(v) != std::string::npos)
      return true;
  return false;
}

static bool
CheckBooleanAttribute(const std::map<std::string, std::string> &attributes,
                      const std::string &key) {
  auto it = attributes.find(key);
  if (it == attributes.end())
    return false;
  return llvm::is_contained({"True", "true"}, it->second);
}

InstructionSASS::InstructionSASS(const lldb_private::Address &address,
                                 const std::string &opcode,
                                 const std::string &operands,
                                 AddressClass addr_class)
    : InstructionSASS(address, opcode, operands, "", "", {}, {}, addr_class) {}

InstructionSASS::InstructionSASS(
    const lldb_private::Address &address, const std::string &opcode,
    const std::string &operands, const std::string &predicate,
    const std::string &extra,
    const std::map<std::string, std::string> &other_attributes,
    const std::vector<std::string> &other_flags, AddressClass addr_class)
    : Instruction(address, addr_class), m_predicate(predicate), m_extra(extra),
      m_other_attributes(other_attributes), m_other_flags(other_flags) {

  // Cache attribute lookups to avoid repeated map searches
  m_is_control_flow = CheckBooleanAttribute(other_attributes, "control-flow");
  m_is_subroutine_call =
      CheckBooleanAttribute(other_attributes, "subroutine-call");
  m_is_barrier = CheckBooleanAttribute(other_attributes, "barrier");
  m_is_load = CheckBooleanAttribute(other_attributes, "load");

  // Set the protected members from the parent class directly
  m_opcode_name = opcode;
  m_mnemonics = operands;
  m_comment = extra; // Use extra field as comment

  // Set markup members with colors
  m_markup_opcode_name =
      ColorizeString(opcode, llvm::raw_ostream::Colors::GREEN);
  m_markup_mnemonics =
      ColorizeString(" " + operands, llvm::raw_ostream::Colors::CYAN);

  // Mark that we've already calculated the strings
  m_calculated_strings = true;
}

bool InstructionSASS::DoesBranch() {
  // Use cached schema attributes if available. Otherwise, fallback to pattern
  // matching.
  if (m_is_control_flow.has_value())
    return *m_is_control_flow;

  // Comprehensive branch detection based on CUDA instruction patterns
  // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

  // Turing+ control flow instructions
  // BRA - Branch
  // BRX - Branch indirect
  // JMP - Jump
  // JMX - Jump indirect
  // RET - Return
  // BRK - Break
  // CONT - Continue
  // SSY - Set Synchronization Point
  // BPT - Breakpoint
  // EXIT - Thread exit
  // SYNC - Synchronize
  // BREAK - Break
  // KILL - Kill thread
  // NANOSLEEP - Nanosleep
  // RTT - Return to top
  // WARPSYNC - Warp sync
  // YIELD - Yield
  // BMOV - Branch move
  // RPCMOV - RPC move
  // ACQBULK - Acquire bulk
  // ENDCOLLECTIVE - End collective
  if (IsASubstringOfAny(
          m_opcode_name,
          {"BRA",    "BRX",       "JMP",          "JMX",      "RET",   "BRK",
           "CONT",   "SSY",       "BPT",          "EXIT",     "SYNC",  "BREAK",
           "KILL",   "NANOSLEEP", "RTT",          "WARPSYNC", "YIELD", "BMOV",
           "RPCMOV", "ACQBULK",   "ENDCOLLECTIVE"})) {
    m_is_control_flow = true;
  } else {
    m_is_control_flow = false;
  }

  return *m_is_control_flow;
}

bool InstructionSASS::IsLoad() {
  // Use cached schema attributes if available. Otherwise, fallback to pattern
  // matching.
  if (m_is_load.has_value())
    return *m_is_load;

  // Comprehensive load detection based on CUDA instruction patterns
  // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

  // Standard load instructions
  // LD - Load
  // LDU - Load uniform
  // LDC - Load constant
  // LDS - Load shared
  // LDG - Load global
  // LDL - Load local
  if (IsASubstringOfAny(m_opcode_name,
                        {"LD", "LDU", "LDC", "LDS", "LDG", "LDL"})) {
    m_is_load = true;
  } else {
    m_is_load = false;
  }

  return *m_is_load;
}

bool InstructionSASS::IsCall() {
  // Use cached schema attributes if available. Otherwise, fallback to pattern
  // matching.
  if (m_is_subroutine_call.has_value())
    return *m_is_subroutine_call;

  // Comprehensive call detection based on CUDA instruction patterns
  // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

  // CALL - Call subroutine
  if (IsASubstringOfAny(m_opcode_name, {"CALL"}))
    m_is_subroutine_call = true;
  else
    m_is_subroutine_call = false;

  return *m_is_subroutine_call;
}

bool InstructionSASS::IsBarrier() {
  // Use cached schema attributes if available. Otherwise, fallback to pattern
  // matching.
  if (m_is_barrier.has_value())
    return *m_is_barrier;

  // Comprehensive barrier detection based on CUDA instruction patterns
  // Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/

  // MEMBAR, DEPBAR, UCGABAR_* - all covered with BAR pattern
  // BAR - Memory barrier (covers MEMBAR, DEPBAR, UCGABAR_*)
  if (IsASubstringOfAny(m_opcode_name, {"BAR"}))
    m_is_barrier = true;
  else
    m_is_barrier = false;

  return *m_is_barrier;
}

size_t InstructionSASS::Decode(const Disassembler &disassembler,
                               const DataExtractor &data,
                               lldb::offset_t data_offset) {
  // For SASS instructions parsed from nvdisasm, we don't need to decode
  // the bytes ourselves - the parsing was already done
  return GetOpcode().GetByteSize();
}

void InstructionSASS::SetOpcode(const void *opcode_data,
                                size_t opcode_data_len) {
  m_opcode.SetOpcodeBytes(opcode_data, opcode_data_len);
}
