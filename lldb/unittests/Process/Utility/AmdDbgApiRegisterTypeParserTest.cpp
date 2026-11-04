//===-- AmdDbgApiRegisterTypeParserTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/AmdDbgApiRegisterTypeParser.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <tuple>

using namespace lldb_private;

// Type strings returned by ROCm 7.0 for gfx942. For reference:
// https://github.com/ROCm/rocm-systems/blob/529f578/projects/rocdbgapi/src/architecture.cpp#L2996
static constexpr const char *k_gfx942_trapsts =
    "flags32_t trapsts {  bool EXCP.INVALID @0;  bool EXCP.DENORM @1;"
    "  bool EXCP.DIV0 @2;  bool EXCP.OVERFLOW @3;  bool EXCP.UNDERFLOW @4;"
    "  bool EXCP.INEXACT @5;  bool EXCP.INT_DIV0 @6;"
    "  bool EXCP.ADDR_WATCH @7;  bool EXCP.MEM_VIOL @8;  bool SAVE_CTX @10;"
    "  bool ILLEGAL_INST @11;  bool EXCP_HI.ADDR_WATCH1 @12;"
    "  bool EXCP_HI.ADDR_WATCH2 @13;  bool EXCP_HI.ADDR_WATCH3 @14;"
    "  uint32_t EXCP_CYCLE @16-21;  bool HOST_TRAP @22;  bool WAVE_BEGIN @23;"
    "  bool WAVE_END @24;  bool TRAP_AFTER_INST @25;  bool PERF_SNAPSHOT @26;"
    "  bool XNACK_ERROR @28;  enum dp_rate {    NONE    = 0,    QUARTER = 1,"
    "    FULL    = 4  } DP_RATE @29-31;}";

static constexpr const char *k_gfx942_mode =
    "flags32_t mode {  enum fp_round {    NEAREST_EVEN = 0,    PLUS_INF  = 1,"
    "    MINUS_INF = 2,    ZERO      = 3  } FP_ROUND.32 @0-1;"
    "  enum fp_round FP_ROUND.64_16 @2-3;  enum fp_denorm {"
    "    FLUSH_SRC_DST = 0,    FLUSH_DST     = 1,    FLUSH_SRC     = 2,"
    "    FLUSH_NONE    = 3  } FP_DENORM.32 @4-5;"
    "  enum fp_denorm FP_DENORM.64_16 @6-7;  bool DX10_CLAMP @8;"
    "  bool IEEE @9;  bool LOD_CLAMPED @10;  bool DEBUG_EN @11;"
    "  bool EXCP_EN.INVALID @12;  bool EXCP_EN.DENORM @13;"
    "  bool EXCP_EN.DIV0 @14;  bool EXCP_EN.OVERFLOW @15;"
    "  bool EXCP_EN.UNDERFLOW @16;  bool EXCP_EN.INEXACT @17;"
    "  bool EXCP_EN.INT_DIV0 @18;  bool EXCP_EN.ADDR_WATCH @19;"
    "  bool TRAP_ON_END @21;  bool FP16_OVFL @23;  bool POPS_PACKER0 @24;"
    "  bool POPS_PACKER1 @25;  bool DISABLE_PERF @26;  bool GPR_IDX_EN @27;"
    "  bool VSKIP @28;  uint32_t CSP @29-31;}";

// The flags of a flags type, or nullptr for any other type.
static llvm::Expected<const RegisterFlags *>
ParseFlags(AmdDbgApiRegisterTypeParser &parser, llvm::StringRef type,
           unsigned byte_size) {
  llvm::Expected<AmdDbgApiRegisterType> parsed = parser.Parse(type, byte_size);
  if (!parsed)
    return parsed.takeError();
  return parsed->flags;
}

using FieldTuple = std::tuple<std::string, unsigned, unsigned>;

// Fields as (name, start, end), most significant first, without the unnamed
// padding fields RegisterFlags adds.
static std::vector<FieldTuple> NamedFields(const RegisterFlags &flags) {
  std::vector<FieldTuple> fields;
  for (const RegisterFlags::Field &field : flags.GetFields())
    if (!field.GetName().empty())
      fields.emplace_back(field.GetName(), field.GetStart(), field.GetEnd());
  return fields;
}

static const RegisterFlags::Field &FieldNamed(const RegisterFlags &flags,
                                              llvm::StringRef name) {
  for (const RegisterFlags::Field &field : flags.GetFields())
    if (field.GetName() == name)
      return field;
  ADD_FAILURE() << "no field named " << name.str();
  static const RegisterFlags::Field missing("", 0);
  return missing;
}

TEST(AmdDbgApiRegisterTypeParserTest, Trapsts) {
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<const RegisterFlags *> flags =
      ParseFlags(parser, k_gfx942_trapsts, 4);
  ASSERT_THAT_EXPECTED(flags, llvm::Succeeded());
  ASSERT_NE(*flags, nullptr);

  EXPECT_EQ((*flags)->GetID(), "trapsts");
  EXPECT_EQ((*flags)->GetSize(), 4u);
  EXPECT_THAT(
      NamedFields(**flags),
      testing::ElementsAre(
          FieldTuple("DP_RATE", 29, 31), FieldTuple("XNACK_ERROR", 28, 28),
          FieldTuple("PERF_SNAPSHOT", 26, 26),
          FieldTuple("TRAP_AFTER_INST", 25, 25), FieldTuple("WAVE_END", 24, 24),
          FieldTuple("WAVE_BEGIN", 23, 23), FieldTuple("HOST_TRAP", 22, 22),
          FieldTuple("EXCP_CYCLE", 16, 21),
          FieldTuple("EXCP_HI.ADDR_WATCH3", 14, 14),
          FieldTuple("EXCP_HI.ADDR_WATCH2", 13, 13),
          FieldTuple("EXCP_HI.ADDR_WATCH1", 12, 12),
          FieldTuple("ILLEGAL_INST", 11, 11), FieldTuple("SAVE_CTX", 10, 10),
          FieldTuple("EXCP.MEM_VIOL", 8, 8),
          FieldTuple("EXCP.ADDR_WATCH", 7, 7),
          FieldTuple("EXCP.INT_DIV0", 6, 6), FieldTuple("EXCP.INEXACT", 5, 5),
          FieldTuple("EXCP.UNDERFLOW", 4, 4), FieldTuple("EXCP.OVERFLOW", 3, 3),
          FieldTuple("EXCP.DIV0", 2, 2), FieldTuple("EXCP.DENORM", 1, 1),
          FieldTuple("EXCP.INVALID", 0, 0)));

  const FieldEnum *dp_rate = FieldNamed(**flags, "DP_RATE").GetEnum();
  ASSERT_NE(dp_rate, nullptr);
  EXPECT_EQ(dp_rate->GetID(), "dp_rate");
  std::vector<std::pair<uint64_t, std::string>> enumerators;
  for (const FieldEnum::Enumerator &enumerator : dp_rate->GetEnumerators())
    enumerators.emplace_back(enumerator.m_value, enumerator.m_name);
  EXPECT_THAT(enumerators, testing::ElementsAre(
                               std::pair<uint64_t, std::string>(0, "NONE"),
                               std::pair<uint64_t, std::string>(1, "QUARTER"),
                               std::pair<uint64_t, std::string>(4, "FULL")));
  EXPECT_EQ(FieldNamed(**flags, "EXCP_CYCLE").GetEnum(), nullptr);
}

TEST(AmdDbgApiRegisterTypeParserTest, EnumReferencedByName) {
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<const RegisterFlags *> flags =
      ParseFlags(parser, k_gfx942_mode, 4);
  ASSERT_THAT_EXPECTED(flags, llvm::Succeeded());
  ASSERT_NE(*flags, nullptr);

  const FieldEnum *fp_round = FieldNamed(**flags, "FP_ROUND.32").GetEnum();
  ASSERT_NE(fp_round, nullptr);
  EXPECT_EQ(FieldNamed(**flags, "FP_ROUND.64_16").GetEnum(), fp_round);
  const FieldEnum *fp_denorm = FieldNamed(**flags, "FP_DENORM.32").GetEnum();
  ASSERT_NE(fp_denorm, nullptr);
  EXPECT_EQ(FieldNamed(**flags, "FP_DENORM.64_16").GetEnum(), fp_denorm);
  EXPECT_EQ(fp_denorm->GetEnumerators().back().m_name, "FLUSH_NONE");

  const RegisterFlags::Field &csp = FieldNamed(**flags, "CSP");
  EXPECT_EQ(csp.GetStart(), 29u);
  EXPECT_EQ(csp.GetEnd(), 31u);
}

TEST(AmdDbgApiRegisterTypeParserTest, FlagsReferencedByName) {
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<const RegisterFlags *> defined =
      ParseFlags(parser, "flags32_t f { bool A @0; }", 4);
  ASSERT_THAT_EXPECTED(defined, llvm::Succeeded());
  EXPECT_THAT_EXPECTED(ParseFlags(parser, "flags32_t f", 4),
                       llvm::HasValue(*defined));
  EXPECT_THAT_EXPECTED(ParseFlags(parser, "flags32_t g", 4), llvm::Failed());
}

TEST(AmdDbgApiRegisterTypeParserTest, OtherTypes) {
  struct Case {
    const char *type;
    unsigned byte_size;
    lldb::Encoding encoding;
    lldb::Format format;
  };
  // The first four are every non-flags type amd-dbgapi 0.77.4 returns for
  // gfx942.
  for (const Case &c : {
           Case{"int32_t", 4, lldb::eEncodingUint, lldb::eFormatHex},
           Case{"uint64_t", 8, lldb::eEncodingUint, lldb::eFormatHex},
           Case{"int32_t[64]", 256, lldb::eEncodingVector,
                lldb::eFormatVectorOfUInt32},
           Case{"void (*)()", 8, lldb::eEncodingUint, lldb::eFormatHex},
           Case{"uint32_t", 4, lldb::eEncodingUint, lldb::eFormatHex},
           Case{"int32_t[32]", 128, lldb::eEncodingVector,
                lldb::eFormatVectorOfUInt32},
           Case{"float", 4, lldb::eEncodingIEEE754, lldb::eFormatFloat},
           Case{"double[2]", 16, lldb::eEncodingVector,
                lldb::eFormatVectorOfFloat64},
           Case{"void(void)", 8, lldb::eEncodingUint, lldb::eFormatHex},
       }) {
    AmdDbgApiRegisterTypeParser parser;
    llvm::Expected<AmdDbgApiRegisterType> parsed =
        parser.Parse(c.type, c.byte_size);
    ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded()) << c.type;
    EXPECT_EQ(parsed->encoding, c.encoding) << c.type;
    EXPECT_EQ(parsed->format, c.format) << c.type;
    EXPECT_EQ(parsed->flags, nullptr) << c.type;
  }
}

TEST(AmdDbgApiRegisterTypeParserTest, FlagsTypeDisplay) {
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<AmdDbgApiRegisterType> parsed =
      parser.Parse(k_gfx942_trapsts, 4);
  ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded());
  EXPECT_EQ(parsed->encoding, lldb::eEncodingUint);
  EXPECT_EQ(parsed->format, lldb::eFormatHex);
  EXPECT_NE(parsed->flags, nullptr);
}

TEST(AmdDbgApiRegisterTypeParserTest, OtherTypeErrors) {
  for (const auto &[type, byte_size, error] : {
           // Sizes that do not match the register.
           std::tuple<const char *, unsigned, const char *>{
               "int32_t[64]", 128, "does not match the register size"},
           {"uint32_t", 8, "does not match the register size"},
           {"double", 4, "does not match the register size"},
           // Unsupported or malformed types.
           {"int8_t", 1, "unsupported type"},
           {"int32_t[x]", 4, "unsupported type"},
           {"bool[4]", 4, "unsupported type"},
       }) {
    AmdDbgApiRegisterTypeParser parser;
    EXPECT_THAT_EXPECTED(parser.Parse(type, byte_size),
                         llvm::FailedWithMessage(testing::HasSubstr(error)))
        << type;
  }
}

TEST(AmdDbgApiRegisterTypeParserTest, Flags64) {
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<const RegisterFlags *> flags =
      ParseFlags(parser, "flags64_t f { uint64_t HI @32-63; bool LO @0; }", 8);
  ASSERT_THAT_EXPECTED(flags, llvm::Succeeded());
  EXPECT_EQ((*flags)->GetSize(), 8u);
  EXPECT_THAT(
      NamedFields(**flags),
      testing::ElementsAre(FieldTuple("HI", 32, 63), FieldTuple("LO", 0, 0)));

  EXPECT_THAT_EXPECTED(ParseFlags(parser, "flags64_t g { bool A @63; }", 8),
                       llvm::Succeeded());
  EXPECT_THAT_EXPECTED(
      ParseFlags(parser, "flags64_t h { bool A @64; }", 8),
      llvm::FailedWithMessage(testing::HasSubstr("invalid bit range")));
  // An integer field must have the size of its flags type.
  EXPECT_THAT_EXPECTED(
      ParseFlags(parser, "flags64_t i { uint32_t A @0-1; }", 8),
      llvm::FailedWithMessage(testing::HasSubstr("unsupported field type")));
}

TEST(AmdDbgApiRegisterTypeParserTest, Errors) {
  for (const auto &[type, error] : {
           // Size does not match the 4-byte register.
           std::pair<const char *, const char *>{"flags64_t f { bool A @0; }",
                                                 "but the register is"},
           // Bit outside the register.
           {"flags32_t f { bool A @32; }", "invalid bit range"},
           // Start after end.
           {"flags32_t f { uint32_t A @3-2; }", "invalid bit range"},
           // Multi-bit field without an end.
           {"flags32_t f { uint32_t A @3; }", "expected '-' and an end bit"},
           {"flags32_t f { uint32_t A @0-3; bool B @2; }", "overlap"},
           // Enum used before it is defined.
           {"flags32_t f { enum e A @0-1; }", "reference to unknown enum"},
           {"flags32_t f { enum { X = 0 } A @0-1; }", "expected an enum name"},
           {"flags32_t f { enum e { X 0 } A @0-1; }", "malformed enumerator"},
           {"flags32_t f { enum e { X = 0x1 } A @0-1; }",
            "malformed enumerator"},
           {"flags32_t f { int8_t A @0-1; }", "unsupported field type"},
           // An integer field must have the size of its flags type.
           {"flags32_t f { uint64_t A @0-1; }", "unsupported field type"},
           {"flags32_t f { bool @0; }", "expected a field name"},
           {"flags32_t f { bool A 0; }", "expected '@' and a start bit"},
           {"flags32_t f { bool A @0 }", "expected ';'"},
           {"flags32_t f { bool A @0;", "missing '}'"},
           {"flags32_t f { }", "has no fields"},
           {"flags32_t f { bool A @0; } trailing", "unexpected"},
           {"flags32_t f junk", "unexpected"},
           {"flags32_t f", "reference to unknown flags type"},
           {"flags32_t { bool A @0; }", "expected a flags type name"},
       }) {
    AmdDbgApiRegisterTypeParser parser;
    EXPECT_THAT_EXPECTED(ParseFlags(parser, type, 4),
                         llvm::FailedWithMessage(testing::HasSubstr(error)))
        << type;
  }
}

TEST(AmdDbgApiRegisterTypeParserTest, FlagsReferenceSizeMismatch) {
  AmdDbgApiRegisterTypeParser parser;
  ASSERT_THAT_EXPECTED(ParseFlags(parser, "flags64_t f { bool A @0; }", 8),
                       llvm::Succeeded());
  EXPECT_THAT_EXPECTED(
      ParseFlags(parser, "flags32_t f", 4),
      llvm::FailedWithMessage(testing::HasSubstr("has a different size")));
}

TEST(AmdDbgApiRegisterTypeParserTest, EnumTooWideForField) {
  // The field is kept without the enum, as ProcessGDBRemote does.
  AmdDbgApiRegisterTypeParser parser;
  llvm::Expected<const RegisterFlags *> flags = ParseFlags(
      parser, "flags32_t f { enum e { X = 0, Y = 4 } A @0-1; bool B @5; }", 4);
  ASSERT_THAT_EXPECTED(flags, llvm::Succeeded());
  EXPECT_THAT(
      NamedFields(**flags),
      testing::ElementsAre(FieldTuple("B", 5, 5), FieldTuple("A", 0, 1)));
  EXPECT_EQ(FieldNamed(**flags, "A").GetEnum(), nullptr);
}

TEST(AmdDbgApiRegisterTypeParserTest, Redefinitions) {
  // As in ROCgdb, a second definition is an error even if it is identical.
  AmdDbgApiRegisterTypeParser parser;
  ASSERT_THAT_EXPECTED(
      ParseFlags(parser, "flags32_t f { enum e { X = 0, Y = 1 } A @0-1; }", 4),
      llvm::Succeeded());
  EXPECT_THAT_EXPECTED(
      ParseFlags(parser, "flags32_t f { enum e { X = 0, Y = 1 } A @0-1; }", 4),
      llvm::Failed());
  EXPECT_THAT_EXPECTED(
      ParseFlags(parser, "flags32_t g { enum e { X = 0, Y = 1 } A @0-1; }", 4),
      llvm::Failed());
  // Referring to the enum by name is fine.
  EXPECT_THAT_EXPECTED(ParseFlags(parser, "flags32_t h { enum e A @0-1; }", 4),
                       llvm::Succeeded());
}

TEST(AmdDbgApiRegisterTypeParserTest, EnumCommas) {
  // As in ROCgdb, the ',' after an enumerator is optional.
  for (const char *type : {
           "flags32_t f { enum e { X = 0, Y = 1, } A @0-1; }",
           "flags32_t f { enum e { X = 0 Y = 1 } A @0-1; }",
       }) {
    AmdDbgApiRegisterTypeParser parser;
    llvm::Expected<const RegisterFlags *> flags = ParseFlags(parser, type, 4);
    ASSERT_THAT_EXPECTED(flags, llvm::Succeeded()) << type;
    const FieldEnum *e = FieldNamed(**flags, "A").GetEnum();
    ASSERT_NE(e, nullptr) << type;
    EXPECT_EQ(e->GetEnumerators().size(), 2u) << type;
  }
}
