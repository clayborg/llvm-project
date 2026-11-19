//===-- AmdDbgApiRegisterTypeParser.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_AMDDBGAPIREGISTERTYPEPARSER_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_AMDDBGAPIREGISTERTYPEPARSER_H

#include "lldb/Target/RegisterFlags.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace lldb_private {

/// How to show a register, derived from its amd-dbgapi type.
struct AmdDbgApiRegisterType {
  lldb::Encoding encoding = lldb::eEncodingUint;
  lldb::Format format = lldb::eFormatHex;
  /// The register's fields, for a flags type. Owned by the parser.
  const RegisterFlags *flags = nullptr;
};

/// Parses the type strings amd-dbgapi returns for
/// AMD_DBGAPI_REGISTER_INFO_TYPE, such as "int32_t[64]" or
/// "flags32_t trapsts { bool EXCP.MEM_VIOL @8; enum dp_rate { FULL = 4 }
/// DP_RATE @29-31; }", and owns the RegisterFlags and FieldEnum objects built
/// from flags types.
///
/// Use one instance for all the registers of an architecture, and keep it
/// alive as long as their RegisterInfos: flags_type points into it.
///
/// Type names must be unique within the instance. A second definition with the
/// same name is an error, as in ROCgdb. The target XML refers to types by name,
/// so a duplicate would be shown with the first definition's fields.
class AmdDbgApiRegisterTypeParser {
public:
  /// Parse \p type for a register of \p byte_size bytes.
  ///
  /// \return An error if \p type is malformed, unsupported, or does not match
  /// \p byte_size.
  llvm::Expected<AmdDbgApiRegisterType> Parse(llvm::StringRef type,
                                              unsigned byte_size);

private:
  llvm::Expected<const RegisterFlags *>
  ParseFlags(llvm::StringRef type, unsigned flags_size, unsigned byte_size);
  llvm::Expected<const FieldEnum *> ParseEnum(llvm::StringRef &type);
  llvm::Expected<RegisterFlags::Field> ParseField(llvm::StringRef &type,
                                                  unsigned register_bits);

  /// Enums defined so far, by name, for references such as "enum fp_round".
  llvm::StringMap<std::unique_ptr<FieldEnum>> m_enums;
  /// Flags types defined so far, by name, for references such as
  /// "flags32_t mode".
  llvm::StringMap<std::unique_ptr<RegisterFlags>> m_flags;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_AMDDBGAPIREGISTERTYPEPARSER_H
