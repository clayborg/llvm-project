//===-- AmdDbgApiRegisterTypeParser.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AmdDbgApiRegisterTypeParser.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
#include <utility>

using namespace lldb_private;

// amd-dbgapi describes each register with a C-like type string. Most are a
// scalar or an array, such as "int32_t" for an SGPR, "int32_t[64]" for a
// VGPR or "void (*)()" for pc. Status, mode and trapsts are flags types that
// name their bit fields. For example, shortened from gfx942's mode, with one
// field per line:
//
//   flags32_t mode {
//     enum fp_round { NEAREST_EVEN = 0, PLUS_INF = 1 } FP_ROUND.32 @0-1;
//     enum fp_round FP_ROUND.64_16 @2-3;
//     bool IEEE @9;
//     uint32_t CSP @29-31;
//   }
//
// Parse reads the whole string. ParseFlags reads the flags type after
// "flags32_t": its name, "mode", and its fields. ParseField reads one field,
// such as "bool IEEE @9;": a type, a name and its bits. ParseEnum reads an
// enum type after "enum": its name, "fp_round", and its enumerators, such as
// "NEAREST_EVEN = 0". A type without a '{...}' body, like the second
// fp_round, refers to an earlier definition.
//
// The strings Parse accepts, where NAME is [A-Za-z0-9_.]+, N is a decimal
// number, [x] is optional and x* / x+ repeat:
//
//   type   := scalar | scalar '[' N ']' | 'void (*)()' | 'void(void)' | flags
//   scalar := 'int32_t' | 'uint32_t' | 'int64_t' | 'uint64_t' | 'float'
//           | 'double'
//   flags  := ('flags32_t' | 'flags64_t') NAME ['{' field+ '}']
//   field  := 'bool' NAME '@' N ';'
//           | (int | enum) NAME '@' N '-' N ';'
//   int    := 'uint32_t' in flags32_t, 'uint64_t' in flags64_t
//   enum   := 'enum' NAME ['{' (NAME '=' N [','])* '}']
//
// The void forms must match exactly. Elsewhere, whitespace separates adjacent
// words and is optional around punctuation.
//
// See the documentation of AMD_DBGAPI_REGISTER_INFO_TYPE in amd-dbgapi.h for
// the format:
// https://github.com/ROCm/rocm-systems/blob/529f578/projects/rocdbgapi/include/amd-dbgapi.h.in#L5396
// amd-dbgapi also returns int32_t, which that grammar omits, and "void (*)()"
// where it says "void(void)". ROCgdb parses the same strings in
// parse_amd_dbgapi_register_type:
// https://github.com/ROCm/ROCgdb/blob/4d8ffef/gdb/amdgpu-tdep.c#L1007

// Skips whitespace, then removes and returns the next word: a keyword, a name
// such as "EXCP.DIV0", or a number. Returns "" if punctuation is next.
static llvm::StringRef ConsumeToken(llvm::StringRef &type) {
  type = type.ltrim();
  llvm::StringRef token = type.take_while(
      [](char c) { return llvm::isAlnum(c) || c == '_' || c == '.'; });
  type = type.drop_front(token.size());
  return token;
}

// Skips whitespace, then removes c if it is next. Returns whether it was.
static bool ConsumeChar(llvm::StringRef &type, char c) {
  type = type.ltrim();
  return type.consume_front(llvm::StringRef(&c, 1));
}

namespace {
// How to show a scalar type such as "int32_t", alone or in an array.
struct ScalarType {
  unsigned byte_size;
  lldb::Encoding encoding;
  lldb::Format format;
  // Format for an array of this type.
  lldb::Format vector_format;
};
} // namespace

// Looks up a scalar type name such as "int32_t" or "float".
//
// amd-dbgapi types SGPRs and VGPR lanes as int32_t, but they often hold
// addresses and masks, so integers show as unsigned hex. Unsigned also lets
// "register write" take any bit pattern.
static std::optional<ScalarType> LookupScalar(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ScalarType>>(name)
      .Cases({"int32_t", "uint32_t"},
             ScalarType{4, lldb::eEncodingUint, lldb::eFormatHex,
                        lldb::eFormatVectorOfUInt32})
      .Cases({"int64_t", "uint64_t"},
             ScalarType{8, lldb::eEncodingUint, lldb::eFormatHex,
                        lldb::eFormatVectorOfUInt64})
      .Case("float", ScalarType{4, lldb::eEncodingIEEE754, lldb::eFormatFloat,
                                lldb::eFormatVectorOfFloat32})
      .Case("double", ScalarType{8, lldb::eEncodingIEEE754, lldb::eFormatFloat,
                                 lldb::eFormatVectorOfFloat64})
      .Default(std::nullopt);
}

// Parses an enum type after the "enum" keyword and returns its FieldEnum. In
// the example at the top of the file, this reads
// "fp_round { NEAREST_EVEN = 0, PLUS_INF = 1 }" in the first field and just
// "fp_round" in the second. As in ROCgdb, the ',' after an enumerator is
// optional.
llvm::Expected<const FieldEnum *>
AmdDbgApiRegisterTypeParser::ParseEnum(llvm::StringRef &type) {
  llvm::StringRef name = ConsumeToken(type);
  if (name.empty())
    return llvm::createStringError(
        llvm::formatv("expected an enum name before \"{0}\"", type));

  auto existing = m_enums.find(name);
  if (!ConsumeChar(type, '{')) {
    if (existing == m_enums.end())
      return llvm::createStringError(
          llvm::formatv("reference to unknown enum \"{0}\"", name));
    return existing->second.get();
  }
  if (existing != m_enums.end())
    return llvm::createStringError(
        llvm::formatv("redefinition of enum \"{0}\"", name));

  // Enumerators: "NAME = N", each optionally followed by ','.
  FieldEnum::Enumerators enumerators;
  while (!ConsumeChar(type, '}')) {
    llvm::StringRef enumerator = ConsumeToken(type);
    uint64_t value;
    if (enumerator.empty() || !ConsumeChar(type, '=') ||
        !llvm::to_integer(ConsumeToken(type), value, 10))
      return llvm::createStringError(
          llvm::formatv("malformed enumerator at \"{0}\"", type));
    enumerators.emplace_back(value, enumerator.str());
    ConsumeChar(type, ',');
  }

  auto inserted = m_enums.try_emplace(
      name, std::make_unique<FieldEnum>(name.str(), enumerators));
  return inserted.first->second.get();
}

// Parses one field of a flags type, such as "bool IEEE @9;" or
// "enum fp_round FP_ROUND.64_16 @2-3;": its type, name and bits, up to and
// including the ';'. register_bits is the size of the flags type in bits.
llvm::Expected<RegisterFlags::Field>
AmdDbgApiRegisterTypeParser::ParseField(llvm::StringRef &type,
                                        unsigned register_bits) {
  // Type: "bool", an integer the size of the flags type, or an enum.
  llvm::StringRef field_type = ConsumeToken(type);
  bool is_bool = field_type == "bool";
  const FieldEnum *enum_type = nullptr;
  if (field_type == "enum") {
    llvm::Expected<const FieldEnum *> parsed = ParseEnum(type);
    if (!parsed)
      return parsed.takeError();
    enum_type = *parsed;
  } else if (!is_bool &&
             field_type != (register_bits == 32 ? "uint32_t" : "uint64_t")) {
    // Integer fields must be as wide as the flags type, as in ROCgdb.
    return llvm::createStringError(
        llvm::formatv("unsupported field type \"{0}\"", field_type));
  }

  // Name.
  llvm::StringRef name = ConsumeToken(type);
  if (name.empty())
    return llvm::createStringError(
        llvm::formatv("expected a field name at \"{0}\"", type));

  // Bits: "@start" for a bool, "@start-end" otherwise.
  uint64_t start = 0;
  if (!ConsumeChar(type, '@') ||
      !llvm::to_integer(ConsumeToken(type), start, 10))
    return llvm::createStringError(
        llvm::formatv("expected '@' and a start bit at \"{0}\"", type));
  uint64_t end = start;
  if (!is_bool && (!ConsumeChar(type, '-') ||
                   !llvm::to_integer(ConsumeToken(type), end, 10)))
    return llvm::createStringError(
        llvm::formatv("expected '-' and an end bit at \"{0}\"", type));

  if (!ConsumeChar(type, ';'))
    return llvm::createStringError(
        llvm::formatv("expected ';' at \"{0}\"", type));

  if (start > end || end >= register_bits)
    return llvm::createStringError(llvm::formatv(
        "field \"{0}\" has invalid bit range {1}-{2}", name, start, end));
  if (enum_type) {
    // Like ProcessGDBRemote's XML parser, keep the field but drop an enum that
    // does not fit it.
    uint64_t max_value = RegisterFlags::Field::GetMaxValue(start, end);
    for (const FieldEnum::Enumerator &enumerator :
         enum_type->GetEnumerators()) {
      if (enumerator.m_value > max_value) {
        LLDB_LOG(GetLog(LLDBLog::Thread),
                 "Ignoring enum {0} of field {1}: enumerator {2} does not fit",
                 enum_type->GetID(), name, enumerator.m_name);
        enum_type = nullptr;
        break;
      }
    }
  }
  return RegisterFlags::Field(name.str(), start, end, enum_type);
}

// Parses a flags type after "flags32_t" or "flags64_t", such as
// "mode { ... }" in the example at the top of the file, and returns its
// RegisterFlags. A reference, such as "flags32_t mode", has no body.
// flags_size and byte_size are the sizes of the flags type and of the
// register, in bytes.
llvm::Expected<const RegisterFlags *> AmdDbgApiRegisterTypeParser::ParseFlags(
    llvm::StringRef rest, unsigned flags_size, unsigned byte_size) {
  if (flags_size != byte_size)
    return llvm::createStringError(
        llvm::formatv("flags type is {0} bytes but the register is {1} bytes",
                      flags_size, byte_size));

  llvm::StringRef name = ConsumeToken(rest);
  if (name.empty())
    return llvm::createStringError(
        llvm::formatv("expected a flags type name before \"{0}\"", rest));

  auto existing = m_flags.find(name);
  // No body: a reference such as "flags32_t mode".
  if (!ConsumeChar(rest, '{')) {
    if (!rest.trim().empty())
      return llvm::createStringError(llvm::formatv("unexpected \"{0}\"", rest));
    if (existing == m_flags.end())
      return llvm::createStringError(
          llvm::formatv("reference to unknown flags type \"{0}\"", name));
    if (existing->second->GetSize() != byte_size)
      return llvm::createStringError(
          llvm::formatv("flags type \"{0}\" has a different size", name));
    return existing->second.get();
  }
  if (existing != m_flags.end())
    return llvm::createStringError(
        llvm::formatv("redefinition of flags type \"{0}\"", name));

  std::vector<RegisterFlags::Field> fields;
  while (!ConsumeChar(rest, '}')) {
    if (rest.trim().empty())
      return llvm::createStringError("missing '}'");
    llvm::Expected<RegisterFlags::Field> field =
        ParseField(rest, flags_size * 8);
    if (!field)
      return field.takeError();
    fields.push_back(std::move(*field));
  }
  if (!rest.trim().empty())
    return llvm::createStringError(llvm::formatv("unexpected \"{0}\"", rest));
  if (fields.empty())
    return llvm::createStringError(
        llvm::formatv("flags type \"{0}\" has no fields", name));

  // RegisterFlags wants fields from the most significant bit down, and
  // asserts that they do not overlap.
  llvm::sort(fields,
             [](const auto &lhs, const auto &rhs) { return rhs < lhs; });
  for (size_t i = 1; i < fields.size(); ++i)
    if (fields[i].Overlaps(fields[i - 1]))
      return llvm::createStringError(
          llvm::formatv("fields \"{0}\" and \"{1}\" overlap",
                        fields[i].GetName(), fields[i - 1].GetName()));

  auto inserted = m_flags.try_emplace(
      name, std::make_unique<RegisterFlags>(name.str(), flags_size, fields));
  return inserted.first->second.get();
}

// Works out how to show a register from its type string.
llvm::Expected<AmdDbgApiRegisterType>
AmdDbgApiRegisterTypeParser::Parse(llvm::StringRef type, unsigned byte_size) {
  // A flags type, such as the mode example at the top of the file.
  llvm::StringRef rest = type;
  llvm::StringRef keyword = ConsumeToken(rest);
  if (keyword == "flags32_t" || keyword == "flags64_t") {
    llvm::Expected<const RegisterFlags *> flags =
        ParseFlags(rest, keyword == "flags32_t" ? 4 : 8, byte_size);
    if (!flags)
      return flags.takeError();
    return AmdDbgApiRegisterType{lldb::eEncodingUint, lldb::eFormatHex, *flags};
  }

  // The code pointer type of pc.
  llvm::StringRef trimmed = type.trim();
  if (trimmed == "void (*)()" || trimmed == "void(void)")
    return AmdDbgApiRegisterType{lldb::eEncodingUint, lldb::eFormatHex};

  // An array such as "int32_t[64]".
  if (trimmed.consume_back("]")) {
    auto [element, count_str] = trimmed.rsplit('[');
    std::optional<ScalarType> scalar = LookupScalar(element.trim());
    uint64_t count;
    if (!scalar || !llvm::to_integer(count_str.trim(), count, 10))
      return llvm::createStringError(
          llvm::formatv("unsupported type \"{0}\"", type));
    if (scalar->byte_size * count != byte_size)
      return llvm::createStringError(
          llvm::formatv("\"{0}\" does not match the register size of {1} bytes",
                        type, byte_size));
    return AmdDbgApiRegisterType{lldb::eEncodingVector, scalar->vector_format};
  }

  // A single value, such as "uint64_t" or "float".
  std::optional<ScalarType> scalar = LookupScalar(trimmed);
  if (!scalar)
    return llvm::createStringError(
        llvm::formatv("unsupported type \"{0}\"", type));
  if (scalar->byte_size != byte_size)
    return llvm::createStringError(
        llvm::formatv("\"{0}\" does not match the register size of {1} bytes",
                      type, byte_size));
  return AmdDbgApiRegisterType{scalar->encoding, scalar->format};
}
